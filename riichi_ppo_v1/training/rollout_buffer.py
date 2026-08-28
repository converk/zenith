"""PPO rollout SoA(Structure-of-Arrays)紧凑缓冲:一次性物化 + 跨 epoch 复用。

worker 在 GAE 完成后把 ``list[Transition]`` 一次性物化成扁平 SoA 数组(变长字段
使用 offset 索引),driver、learner 与 DDP 分片全程只传递该结构。``collate``
把 minibatch 下标向量化 gather 成 V18 padded host 张量。

边界说明:
- 所有变长字段都存成「flat 数组 + offset(长度 N+1)」;collate 用一次性 gather
  替代逐 transition 的 ``torch.tensor`` 拷贝。
- ``legal_mask`` 固定为 241 维,不涉及 padding。
- ``critic_factors`` 在长度为 0 时为空数组;collate 处理 max_critic==0 情形。
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import torch

from ..model.encoding_protocol import (
    QUERY_ROW_WIDTH,
    TOKEN_NUMERIC_WIDTH,
    TOKEN_ROW_WIDTH,
    SEGMENT_SHARED,
)
from ..model.schema import NUM_ACTIONS
from .trajectory import Transition, transition_sequence_length

# V18 当前局面行宽与 critic 私有行宽共享 token schema。
_ACTOR_W = TOKEN_ROW_WIDTH
_NUMERIC_W = TOKEN_NUMERIC_WIDTH
_QUERY_W = QUERY_ROW_WIDTH
_CRITIC_W = TOKEN_ROW_WIDTH


def _gather_padded(
    flat: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    max_len: int,
    default: int | float,
    *,
    width: int | None = None,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """把 flat 数组按 ``(starts, lengths)`` gather 成 ``[B, max_len(, width)]``。

    向量化版本,避免逐 transition 循环。``flat`` 是按行拼接的连续存储,
    ``starts``/``lengths`` 指向每行在 flat 中的起点与长度;超出长度的位置填
    ``default``。``width`` 为 None 时返回 2D,否则返回 3D。

    优化:单次分配目标 dtype 的输出并只写合法区,替代「advanced-index 拷贝 +
    ``np.where`` 再拷贝 + ``.long()`` 再拷贝」的三次全量复制;输出值与旧实现
    逐位一致(``dtype`` 缺省时与 ``flat`` 相同)。
    """
    batch = int(len(lengths))
    if max_len == 0:
        if width is None:
            return np.zeros((batch, 0), dtype=dtype or flat.dtype)
        return np.zeros((batch, 0, width), dtype=dtype or flat.dtype)
    positions = np.arange(max_len, dtype=np.int64)[None, :] + starts[:, None]
    valid = positions < (starts + lengths)[:, None]
    safe = np.where(valid, positions, 0)
    target_dtype = np.dtype(dtype if dtype is not None else flat.dtype)
    if width is None:
        out = np.empty((batch, max_len), dtype=target_dtype)
        out[valid] = flat[safe[valid]]
        out[~valid] = default
    else:
        out = np.empty((batch, max_len, width), dtype=target_dtype)
        out[valid] = flat[safe[valid]]
        out[~valid] = default
    return out


def _compact_factor_flat(flat: np.ndarray, name: str) -> np.ndarray:
    """把 V18 token 因子行压成 uint8(所有字段值 < 256,见 schema 单源)。

    存储与 Ray/多进程传输体积缩小 4 倍(实测 512 半庄全量缓冲 ~8GB → ~2GB),
    ``collate`` 输出仍按模型需求一次性转 int64,数值逐位一致。超出 uint8
    域时 fail-closed,避免静默回绕。
    """
    if flat.size == 0:
        return flat.astype(np.uint8, copy=False)
    if int(flat.max()) > 255:
        raise ValueError(f"{name} factor exceeds uint8 range (max={int(flat.max())})")
    return flat.astype(np.uint8, copy=False)


class RolloutBuffer:
    """一次物化整个 rollout 的 SoA 缓冲。

    只保留训练所需字段;reward/done/advantage 等标量也一并抽取,供 learner
    复用(不再遍历 ``Transition`` 对象)。构造时只做一次 O(N) 的数组拼接与
    标量抽取,之后的 ``collate`` 都是纯 numpy gather + 一次 torch 转换。
    """

    def __init__(self, transitions: Sequence[Transition]) -> None:
        count = len(transitions)
        self.size = count

        self.actor_lengths = np.array(
            [int(item.actor_length) for item in transitions], dtype=np.int64
        )
        self.query_pair_counts = np.array(
            [int(item.query_pair_counts) for item in transitions], dtype=np.int64
        )
        self.critic_lengths = np.array(
            [int(item.critic_length) for item in transitions], dtype=np.int64
        )
        self.sequence_lengths = np.array(
            [transition_sequence_length(item) for item in transitions], dtype=np.int64
        )
        self.actions = np.array([int(item.action) for item in transitions], dtype=np.int64)
        self.old_logprobs = np.array(
            [float(item.logprob) for item in transitions], dtype=np.float32
        )
        self.values = np.array([float(item.value) for item in transitions], dtype=np.float32)
        self.rewards = np.array([float(item.reward) for item in transitions], dtype=np.float32)
        self.kyoku_rewards = np.array(
            [float(item.kyoku_reward) for item in transitions], dtype=np.float32
        )
        self.done = np.array([bool(item.done) for item in transitions], dtype=np.bool_)
        self.advantages = np.array(
            [float(item.advantage) for item in transitions], dtype=np.float32
        )
        self.legal_mask = np.stack([item.legal_mask for item in transitions], axis=0).astype(
            np.bool_
        )

        # ---- 可变长字段:flat + offset ----
        self.actor_offsets, actor_factors_flat, _ = self._concat_var(
            transitions,
            lambda item: item.actor_factors.astype(np.int32, copy=False),
        )
        self.actor_factors_flat = _compact_factor_flat(
            actor_factors_flat, "actor_factors",
        )
        (
            self.actor_numeric_offsets,
            self.actor_numeric_flat,
            _,
        ) = self._concat_var(
            transitions, lambda item: item.actor_numeric.astype(np.float32, copy=False)
        )
        (
            self.query_rows_offsets,
            query_rows_flat,
            _,
        ) = self._concat_var(
            transitions, lambda item: item.query_rows.astype(np.int32, copy=False)
        )
        self.query_rows_flat = _compact_factor_flat(query_rows_flat, "query_rows")
        (
            self.query_ids_offsets,
            query_ids_flat,
            _,
        ) = self._concat_var(
            transitions, lambda item: item.query_action_ids.astype(np.int32, copy=False)
        )
        self.query_ids_flat = _compact_factor_flat(query_ids_flat, "query_action_ids")

        # critic 是可选字段;长度为 0 时为空 array。
        critic_parts = []
        critic_offsets = []
        cursor = 0
        for item in transitions:
            critic_offsets.append(cursor)
            if int(item.critic_length) > 0:
                arr = np.asarray(item.critic_factors, dtype=np.uint8)
                critic_parts.append(arr)
                cursor += arr.shape[0]
        critic_offsets.append(cursor)
        self.critic_offsets = np.asarray(critic_offsets, dtype=np.int64)
        self.critic_factors_flat = (
            np.concatenate(critic_parts, axis=0) if critic_parts else np.zeros((0, _CRITIC_W), dtype=np.uint8)
        )

    def __len__(self) -> int:
        return self.size

    def arrays(self) -> Iterator[np.ndarray]:
        """迭代实际参与序列化的连续数组,供传输 profiling 使用。"""
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                yield value

    def payload_stats(self) -> tuple[int, int]:
        """返回 ndarray 数量与数组有效字节数。"""
        arrays = tuple(self.arrays())
        return len(arrays), sum(int(array.nbytes) for array in arrays)

    @staticmethod
    def _offsets(lengths: np.ndarray) -> np.ndarray:
        offsets = np.empty(len(lengths) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])
        return offsets

    @classmethod
    def concatenate(cls, buffers: Sequence["RolloutBuffer"]) -> "RolloutBuffer":
        """按 worker 顺序合并 SoA shard,不恢复百万级 Transition 对象。"""
        if not buffers:
            raise ValueError("cannot concatenate an empty rollout buffer list")
        result = cls.__new__(cls)
        result.size = sum(len(buffer) for buffer in buffers)
        fixed_fields = (
            "actor_lengths", "query_pair_counts",
            "critic_lengths", "sequence_lengths", "actions", "old_logprobs",
            "values", "rewards", "kyoku_rewards", "done", "advantages",
            "legal_mask",
        )
        for name in fixed_fields:
            setattr(result, name, np.concatenate([
                np.asarray(getattr(buffer, name)) for buffer in buffers
            ], axis=0))
        variable_fields = (
            ("actor_offsets", "actor_factors_flat", result.actor_lengths),
            ("actor_numeric_offsets", "actor_numeric_flat", result.actor_lengths),
            ("query_rows_offsets", "query_rows_flat", 2 * result.query_pair_counts),
            ("query_ids_offsets", "query_ids_flat", result.query_pair_counts),
            ("critic_offsets", "critic_factors_flat", result.critic_lengths),
        )
        for offsets_name, flat_name, lengths in variable_fields:
            setattr(result, offsets_name, cls._offsets(lengths))
            setattr(result, flat_name, np.concatenate([
                np.asarray(getattr(buffer, flat_name)) for buffer in buffers
            ], axis=0))
        return result

    @staticmethod
    def _select_flat(
        flat: np.ndarray,
        offsets: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """按任意下标重排 flat+offsets,支持 DDP 补齐产生的重复下标。"""
        starts = offsets[indices]
        lengths = offsets[indices + 1] - starts
        selected_offsets = RolloutBuffer._offsets(lengths)
        total = int(selected_offsets[-1])
        if total == 0:
            return selected_offsets, np.empty((0, *flat.shape[1:]), dtype=flat.dtype)
        # 每行源起点减目标起点后 repeat,一次生成连续 gather 下标。
        bases = np.repeat(starts - selected_offsets[:-1], lengths)
        source_indices = bases + np.arange(total, dtype=np.int64)
        return selected_offsets, np.ascontiguousarray(flat[source_indices])

    def select(self, indices: Sequence[int]) -> "RolloutBuffer":
        """构造训练分片;字段与顺序严格遵循给定下标。"""
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1 or not len(idx):
            raise ValueError("rollout buffer selection must be a non-empty 1D index array")
        if int(idx.min()) < 0 or int(idx.max()) >= self.size:
            raise IndexError("rollout buffer selection index is out of range")
        result = self.__class__.__new__(self.__class__)
        result.size = len(idx)
        fixed_fields = (
            "actor_lengths", "query_pair_counts",
            "critic_lengths", "sequence_lengths", "actions", "old_logprobs",
            "values", "rewards", "kyoku_rewards", "done", "advantages",
            "legal_mask",
        )
        for name in fixed_fields:
            setattr(result, name, np.ascontiguousarray(getattr(self, name)[idx]))
        variable_fields = (
            ("actor_offsets", "actor_factors_flat"),
            ("actor_numeric_offsets", "actor_numeric_flat"),
            ("query_rows_offsets", "query_rows_flat"),
            ("query_ids_offsets", "query_ids_flat"),
            ("critic_offsets", "critic_factors_flat"),
        )
        for offsets_name, flat_name in variable_fields:
            selected_offsets, selected_flat = self._select_flat(
                getattr(self, flat_name), getattr(self, offsets_name), idx,
            )
            setattr(result, offsets_name, selected_offsets)
            setattr(result, flat_name, selected_flat)
        return result

    @staticmethod
    def _concat_var(
        transitions: Sequence[Transition],
        extract,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按行拼接可变长数组,返回 (offsets[N+1], flat, per-row width)。

        ``extract`` 返回每个 transition 的 2D array(第一维为 token 数)。
        """
        parts = []
        offsets = []
        rows = 0
        for item in transitions:
            offsets.append(rows)
            arr = np.asarray(extract(item))
            parts.append(arr)
            rows += int(arr.shape[0])
        offsets.append(rows)
        flat = np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)
        return np.asarray(offsets, dtype=np.int64), flat, (flat.shape[1] if flat.ndim >= 2 else 0)

    def collate(
        self,
        indices: Sequence[int],
        *,
        include_query_rows: bool = True,
    ) -> dict[str, torch.Tensor]:
        """把一个 minibatch 的下标数组 gather 成 V18 padded host 张量(CPU)。

        ``include_query_rows=False`` 时跳过 ``query_rows`` 的 gather 与输出:
        该字段仅作离线一致性校验,模型 forward 不消费,learner 侧传 False
        省去每 minibatch 一次无用的拼装/拷贝;默认 True 保持其余调用方
        (worker/inference/测试)的历史行为不变。

        额外输出 host 侧标量 ``shared_capacity``/``critic_total_capacity``
        (numpy 预计算,与 forward 内 GPU 推导逐位同义):learner 侧透传给
        forward 以消除 ``max().item()`` 同步;调用方不消费时忽略即可。
        """
        idx = np.asarray([int(index) for index in indices], dtype=np.int64)
        if len(idx) == 0:
            raise ValueError("cannot collate an empty minibatch")
        batch = int(len(idx))

        actor_lens = self.actor_lengths[idx]
        pair_counts = self.query_pair_counts[idx]
        critic_lens = self.critic_lengths[idx]

        max_actor = int(actor_lens.max(initial=0))
        max_pairs = int(pair_counts.max(initial=0))
        max_critic = int(critic_lens.max(initial=0))

        actor_factors = _gather_padded(
            self.actor_factors_flat,
            self.actor_offsets[idx],
            actor_lens,
            max_actor,
            0,
            width=_ACTOR_W,
            dtype=np.int64,
        )
        actor_numeric = _gather_padded(
            self.actor_numeric_flat,
            self.actor_numeric_offsets[idx],
            actor_lens,
            max_actor,
            0.0,
            width=_NUMERIC_W,
        )
        query_rows = None
        if include_query_rows:
            query_rows = _gather_padded(
                self.query_rows_flat,
                self.query_rows_offsets[idx],
                2 * pair_counts,
                2 * max_pairs,
                0,
                width=_QUERY_W,
                dtype=np.int32,
            )
        query_action_ids = _gather_padded(
            self.query_ids_flat,
            self.query_ids_offsets[idx],
            pair_counts,
            max_pairs,
            0,
            dtype=np.int64,
        )
        if max_critic > 0:
            critic_factors = _gather_padded(
                self.critic_factors_flat,
                self.critic_offsets[idx],
                critic_lens,
                max_critic,
                0,
                width=_CRITIC_W,
                dtype=np.int64,
            )
        else:
            critic_factors = np.zeros((batch, 0, _CRITIC_W), dtype=np.int64)

        # 标量长度/动作使用 long,legal mask 使用 bool。
        actor_lengths = torch.from_numpy(actor_lens)
        query_pair_counts = torch.from_numpy(pair_counts)
        critic_lengths = torch.from_numpy(critic_lens)
        actions = torch.from_numpy(self.actions[idx])
        old_logprobs = torch.from_numpy(self.old_logprobs[idx])
        advantages = torch.from_numpy(self.advantages[idx])
        legal = torch.from_numpy(self.legal_mask[idx])

        # host 侧容量预计算:shared 行数 = segment==SHARED 的有效行数(padding
        # 行 segment 为 0 不污染计数);critic 总长 = shared + critic + value 行。
        # 语义与 architecture.forward 内 GPU 推导逐位同义,经单测 torch.equal 对照。
        shared_per_row = (actor_factors[..., 0] == SEGMENT_SHARED).sum(-1)
        shared_capacity = int(shared_per_row.max(initial=0))
        critic_total_capacity = int(
            (shared_per_row + critic_lens + 1).max(initial=0)
        )

        # 键顺序与历史实现保持一致(query_rows 缺省时原位跳过)。
        batch_tensors = {
            "actor_factors": torch.from_numpy(np.ascontiguousarray(actor_factors)),
            "actor_numeric": torch.from_numpy(np.ascontiguousarray(actor_numeric)),
            "actor_lengths": actor_lengths,
        }
        if include_query_rows:
            batch_tensors["query_rows"] = torch.from_numpy(
                np.ascontiguousarray(query_rows)
            )
        batch_tensors.update({
            "query_action_ids": torch.from_numpy(np.ascontiguousarray(query_action_ids)),
            "query_pair_counts": query_pair_counts,
            "legal_mask": legal,
            "critic_factors": torch.from_numpy(np.ascontiguousarray(critic_factors)),
            "critic_lengths": critic_lengths,
            "actions": actions,
            "old_logprobs": old_logprobs,
            "advantages": advantages,
            # host 侧标量放末尾;非张量,learner 侧 pop 后透传 forward。
            "shared_capacity": shared_capacity,
            "critic_total_capacity": critic_total_capacity,
        })
        return batch_tensors

    def bucketed_minibatches(
        self,
        minibatch_size: int,
        rng: np.random.Generator | None = None,
        *,
        bucket_window_multiplier: int = 1,
    ) -> tuple[np.ndarray, ...]:
        """按粗粒度长度窗口构造 minibatch,兼顾随机性与 padding 成本。"""
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        window_multiplier = max(1, int(bucket_window_multiplier))
        count = self.size
        if count == 0:
            raise ValueError("cannot bucket an empty rollout")
        lengths = self.sequence_lengths
        permutation = rng.permutation if rng is not None else np.random.permutation
        shuffled = permutation(count)
        sorted_indices = shuffled[np.argsort(lengths[shuffled], kind="stable")]
        window_size = max(minibatch_size, minibatch_size * window_multiplier)
        windowed = sorted_indices.copy()
        for start in range(0, count, window_size):
            stop = min(start + window_size, count)
            windowed[start:stop] = windowed[start:stop][permutation(stop - start)]
        batches = tuple(
            windowed[start : start + minibatch_size]
            for start in range(0, count, minibatch_size)
        )
        batch_order = permutation(len(batches))
        return tuple(batches[index] for index in batch_order)
