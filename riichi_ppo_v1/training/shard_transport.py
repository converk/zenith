"""B2:DDP shard 的共享内存传输(driver → learner rank,省去 pickle/管道拷贝)。

现状(pickle 传输,mp.Queue):driver 侧 shard select gather 拷贝后,feeder
线程 pickle(整块拷贝)→ 管道 memcpy → learner unpickle(再拷贝),2.5GB 分
片共 4 次内存拷贝,且与 12 个 rollout worker 争内存带宽。

共享内存路径:driver 把 shard 的 SoA 数组逐一写入一个 ``SharedMemory`` 块
(select gather 之外的唯一一次拷贝),命令队列只传元数据(字段布局);learner
按元数据以 ``np.ndarray(buffer=...)`` 零拷贝视图重建 ``RolloutBuffer``。数组
逐位一致(★ 语义),``update_timeout_s`` 与错误传播语义不变。

生命周期:driver 创建并 unlink-负责(收到该 rank 结果或异常后统一
``close()+unlink()``);learner attach 后只在 update 期间持有映射,update
结束(含异常)即 ``close()``,不 unlink。
"""

from __future__ import annotations

from multiprocessing import shared_memory
from typing import Any

import numpy as np

from .rollout_buffer import RolloutBuffer

# 共享内存块命名前缀(清扫与命名共用单一来源)。
SHARD_SHM_PREFIX = "riichi-ppo-shard-"


def sweep_stale_shard_blocks() -> int:
    """清理已消亡进程遗留的 shard 共享内存块,返回清理数量。

    driver 异常消亡时 finally 的 unlink 可能不执行,tmpfs 块会滞留 /dev/shm
    并占用内存;块名内嵌创建者 pid,本进程启动时对「pid 已不存在」的块做
    一次尽力清扫(存活 pid 的块属于并行运行的其他训练,绝不动)。
    """
    import glob
    import os

    removed = 0
    for path in glob.glob(f"/dev/shm/{SHARD_SHM_PREFIX}*"):
        parts = os.path.basename(path).split("-")
        try:
            # 命名:riichi-ppo-shard-<pid>-<seq>-<rank>。
            pid = int(parts[3])
        except (IndexError, ValueError):
            continue
        if os.path.exists(f"/proc/{pid}"):
            continue
        try:
            os.unlink(path)
            removed += 1
        except OSError:
            pass
    return removed


def shard_field_arrays(buffer: RolloutBuffer) -> dict[str, np.ndarray]:
    """收集 shard 传输所需的全部 SoA 数组(与 select 产物字段一致)。"""
    fields: dict[str, np.ndarray] = {}
    for name in buffer._FIXED_FIELDS:  # noqa: SLF001 - 同包内受控访问
        fields[name] = getattr(buffer, name)
    for offsets_name, flat_name in buffer._VARIABLE_FIELDS:  # noqa: SLF001
        fields[offsets_name] = getattr(buffer, offsets_name)
        fields[flat_name] = getattr(buffer, flat_name)
    return fields


class ShardShmWriter:
    """把一个 rank 分片的数组布局写入新建的共享内存块。

    ``write`` 计算各数组在块内的偏移(8 字节对齐),``np.copyto`` 逐一写入,
    返回可经 pickle 小队列传递的元数据 dict。分片体积约 2.5GB,/dev/shm
    需容纳 world_size 份(world_size=2 时约 5GB,临时存在一个 update 周期)。
    """

    def __init__(self, fields: dict[str, np.ndarray], name: str) -> None:
        self._name = name
        layout: list[tuple[str, tuple[int, ...], str, int]] = []
        cursor = 0
        for field_name, array in fields.items():
            contiguous = np.ascontiguousarray(array)
            start = (cursor + 7) // 8 * 8
            layout.append((
                field_name,
                tuple(int(size) for size in contiguous.shape),
                contiguous.dtype.str,
                start,
            ))
            cursor = start + int(contiguous.nbytes)
        if cursor <= 0:
            raise ValueError("shard shm payload must contain at least one array")
        self._shm = shared_memory.SharedMemory(create=True, size=cursor, name=name)
        self._layout = layout
        self._fields = fields
        self._written = False

    def write(self) -> dict[str, Any]:
        """把数组写入共享内存并返回元数据(含块名与逐字段布局)。"""
        if self._written:
            raise RuntimeError("shard shm writer must write exactly once")
        self._written = True
        for field_name, shape, dtype_str, offset in self._layout:
            array = np.ascontiguousarray(self._fields[field_name])
            view = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=self._shm.buf, offset=offset)
            np.copyto(view, array)
        return {
            "shm_name": self._shm.name,
            "total_bytes": self._shm.size,
            "layout": self._layout,
        }

    def release(self) -> None:
        """driver 侧释放:close + unlink(幂等)。"""
        shm = getattr(self, "_shm", None)
        if shm is not None:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            self._shm = None


class ShardShmView:
    """learner 侧的共享内存分片视图:update 期间持有,结束后 close。

    ``buffer`` 是零拷贝重建的 ``RolloutBuffer``(数组直接指向共享内存),
    数值与 driver 写入的 select 分片逐位一致。
    """

    def __init__(self, meta: dict[str, Any]) -> None:
        self._shm = shared_memory.SharedMemory(name=meta["shm_name"])
        fields: dict[str, np.ndarray] = {}
        for field_name, shape, dtype_str, offset in meta["layout"]:
            fields[field_name] = np.ndarray(
                tuple(shape), dtype=np.dtype(dtype_str),
                buffer=self._shm.buf, offset=offset,
            )
        self.buffer = RolloutBuffer.from_field_map(fields)

    def close(self) -> None:
        """解除本进程映射(不 unlink,所有权在 driver)。"""
        shm = getattr(self, "_shm", None)
        if shm is not None:
            shm.close()
            self._shm = None
