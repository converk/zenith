"""双卡 DDP PPO learner:两个常驻 worker 进程 + NCCL 同步。

driver 侧只做编排(分片、汇总、checkpoint 落盘),不持有模型;每个 worker
持有自己的 ``PPOLearner``(rank 0/1)。update 时各自对本地分片跑 DDP
前向/反向,NCCL 平均梯度后同步执行优化器 step。

分片按轮询切分并把长度补齐到一致的本地 minibatch 数,保证两个 rank 的
collective 严格对齐(不依赖 ``model.join``);early-stop 与 loss 有限性判定
也通过进程组同步,避免单 rank 提前退出造成死锁。
"""

from __future__ import annotations

import ctypes
import datetime
import multiprocessing
import os
import queue as _queue
import random
import signal
import socket
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .learner import (
    PPOLearner,
    discounted_empirical_returns,
    rollout_target_metrics,
    rollout_update_targets,
)
from .rollout_buffer import RolloutBuffer
from .shard_transport import ShardShmView, ShardShmWriter, shard_field_arrays

_CMD_UPDATE = "update"
_CMD_WEIGHTS = "weights"
_CMD_RNG_STATE = "rng_state"
_CMD_SAVE = "save"
_CMD_SHUTDOWN = "shutdown"
_CMD_RELEASE_CACHE = "release_cache"


def _enable_parent_death_signal() -> None:
    """Linux 下给 DDP 子进程设置 PDEATHSIG=SIGKILL,作为孤儿兜底。

    driver 进程无论因何退出(包括无法捕获的 SIGKILL/段错误),内核都会立即
    杀掉本 rank,避免孤儿进程继续占用 GPU 显存;正常路径仍走 _CMD_SHUTDOWN
    优雅退出,此机制只兜底异常消亡。非 Linux 或 prctl 不可用时静默降级,
    依赖 driver 侧的 SIGTERM 处理器与 shutdown() 清理。
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        PR_SET_PDEATHSIG = 1  # noqa: N806 -- Linux prctl 平台常量,保留大写惯例
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    except Exception:
        pass


# 按 executed samples 加权平均的指标(每个 rank 处理样本数可能差一个 padding)。
_SAMPLE_WEIGHTED_KEYS = {
    "loss", "policy_loss", "value_loss", "value_loss_raw", "value_prediction",
    "entropy", "entropy_normalized",
    "sft_reference_kl", "approx_kl", "clipfrac", "ratio",
    "update/executed_transition_tokens_mean",
}
# 按 executed minibatches 加权平均的指标。
_STEP_WEIGHTED_KEYS = {
    "grad_norm", "grad_norm_post_clip",
    "grad_norm_actor", "grad_norm_critic", "grad_norm_shared",
    "grad_norm_actor_pre_clip", "grad_norm_actor_post_clip",
    "grad_norm_actor_clip_scale", "grad_norm_actor_clipped",
    "grad_norm_shared_pre_clip", "grad_norm_shared_post_clip",
    "grad_norm_shared_clip_scale", "grad_norm_shared_clipped",
    "grad_norm_critic_pre_clip", "grad_norm_critic_post_clip",
    "grad_norm_critic_clip_scale", "grad_norm_critic_clipped",
}
# 直接求和的指标。
_SUM_KEYS = {
    "transitions",
    "update/planned_minibatches", "update/executed_minibatches",
    "update/executed_transition_samples",
    "update/executed_padded_input_tokens", "update/executed_padding_input_tokens",
}
# 两个 rank 同步后取值必然一致的指标,直接取 rank 0。
_RANK0_KEYS = {
    "update/configured_epochs", "update/epochs_started", "update/epochs_completed",
    "update/early_stop",
    "system/learning_rate", "system/actor_learning_rate", "system/shared_learning_rate",
    "system/critic_learning_rate", "system/entropy_coef", "system/sft_kl_coef",
    "system/critic_public_grad_scale", "system/critic_private_embedding_grad_scale",
    "training/critic_bootstrap", "training/policy_update",
}


def learner_shard_indices(
    count: int,
    world_size: int,
    minibatch_size: int,
) -> list[list[int]]:
    """返回每个 rank 分片在完整 rollout 中的下标(含补齐)。"""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    if world_size == 1:
        return [list(range(count))]
    if count < world_size:
        raise ValueError(
            f"cannot split {count} transitions into {world_size} shards"
        )
    index_shards = [
        list(range(rank, count, world_size)) for rank in range(world_size)
    ]
    counts = [len(shard) for shard in index_shards]
    target_batches = max(
        (count + minibatch_size - 1) // minibatch_size for count in counts
    )
    padded: list[list[int]] = []
    for shard, shard_count in zip(index_shards, counts, strict=True):
        minimum = (target_batches - 1) * minibatch_size + 1
        needed = max(0, minimum - shard_count)
        if needed:
            filler = [shard[index % shard_count] for index in range(needed)]
            padded.append(shard + filler)
        else:
            padded.append(shard)
    return padded


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = float(sum(weights))
    if total <= 0.0:
        return float(np.mean(values))
    return float(sum(value * weight for value, weight in zip(values, weights)) / total)


def aggregate_learner_metrics(
    metrics_by_rank: list[dict[str, float]],
    transitions: RolloutBuffer,
    *,
    gamma: float = 1.0,
) -> dict[str, float]:
    """把两个 DDP rank 的本地指标汇总为一次 update 的全局指标。"""
    if not metrics_by_rank:
        raise ValueError("cannot aggregate an empty metrics list")
    sample_weights = [
        float(metrics.get("update/executed_transition_samples", 0.0))
        for metrics in metrics_by_rank
    ]
    step_weights = [
        float(metrics.get("update/executed_minibatches", 0.0))
        for metrics in metrics_by_rank
    ]
    names = {name for metrics in metrics_by_rank for name in metrics}
    result: dict[str, float] = {}
    for name in names:
        present = [
            index for index, metrics in enumerate(metrics_by_rank) if name in metrics
        ]
        values = [float(metrics_by_rank[index][name]) for index in present]
        if name in _SAMPLE_WEIGHTED_KEYS:
            weights = [sample_weights[index] for index in present]
            result[name] = _weighted_mean(values, weights)
        elif name in _STEP_WEIGHTED_KEYS:
            weights = [step_weights[index] for index in present]
            result[name] = _weighted_mean(values, weights)
        elif name in _SUM_KEYS:
            result[name] = float(sum(values))
        elif name in _RANK0_KEYS:
            result[name] = float(metrics_by_rank[0][name])
        elif name.startswith("timing/"):
            if name.endswith("/count"):
                result[name] = float(sum(values))
            elif name.endswith("/total_s") or name.endswith("/max_ms"):
                result[name] = float(max(values))
            elif name.endswith("/mean_ms"):
                # 占位,待 total_s/count 全部汇总后统一重算。
                result[name] = 0.0
            else:
                result[name] = float(metrics_by_rank[0][name])
        elif name.startswith("gpu/"):
            # 保留 rank 0 的显存占用;峰值统计由调用方使用逐 rank 列表。
            result[name] = float(metrics_by_rank[0][name])
        else:
            result[name] = float(metrics_by_rank[0][name])
    for name in list(result):
        if name.startswith("timing/") and name.endswith("/mean_ms"):
            total = result.get(name.removesuffix("/mean_ms") + "/total_s", 0.0)
            count = result.get(name.removesuffix("/mean_ms") + "/count", 0.0)
            result[name] = total * 1000.0 / max(count, 1)
    # buffer/Q 统计基于完整 rollout 在 host 侧精确重算,覆盖各 rank 分片统计。
    result.update(rollout_target_metrics(transitions))
    advantages, lambda_returns, _lambda_mean, _lambda_std = rollout_update_targets(
        transitions,
    )
    mc_returns = discounted_empirical_returns(transitions, gamma=gamma).astype(np.float64)
    raw_advantages = np.asarray(transitions.advantages, dtype=np.float64)
    raw_values = np.asarray(transitions.values, dtype=np.float64)
    lambda_targets = np.asarray(lambda_returns, dtype=np.float64)
    lambda_var = float(np.var(lambda_targets))
    mc_var = float(np.var(mc_returns))
    lambda_ev = (
        0.0
        if lambda_var <= 1e-12
        else 1.0 - float(np.var(lambda_targets - raw_values)) / lambda_var
    )
    mc_ev = (
        0.0
        if mc_var <= 1e-12
        else 1.0 - float(np.var(mc_returns - raw_values)) / mc_var
    )
    result.update({
        "value_explained_variance": lambda_ev,
        "value_explained_variance_lambda": lambda_ev,
        "value_explained_variance_mc": mc_ev,
        "raw_advantage_mean": float(raw_advantages.mean()) if len(raw_advantages) else 0.0,
        "raw_advantage_std": float(raw_advantages.std()) if len(raw_advantages) else 0.0,
        "normalized_advantage_mean": (
            float(np.asarray(advantages, dtype=np.float64).mean()) if len(transitions) else 0.0
        ),
        "normalized_advantage_std": float(np.asarray(advantages, dtype=np.float64).std()) if len(transitions) else 0.0,
        "lambda_return_mean": float(lambda_targets.mean()) if len(lambda_targets) else 0.0,
        "lambda_return_std": float(lambda_targets.std()) if len(lambda_targets) else 0.0,
        "mc_return_mean": float(mc_returns.mean()) if len(mc_returns) else 0.0,
        "mc_return_std": float(mc_returns.std()) if len(mc_returns) else 0.0,
    })
    padded = result["update/executed_padded_input_tokens"]
    tokens = (
        result["update/executed_transition_tokens_mean"]
        * result["update/executed_transition_samples"]
    )
    result["update/executed_padding_fraction_of_padded_input_tokens"] = (
        (padded - tokens) / max(padded, 1.0)
    )
    return result


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _learner_worker(
    rank: int,
    world_size: int,
    model_size: str,
    config: dict[str, Any],
    resume_path: str | None,
    init_model_path: str | None,
    command_queue: Any,
    result_queue: Any,
) -> None:
    """单个 DDP rank 的常驻循环:初始化进程组后按命令执行 update/保存。"""
    # 父进程(driver)异常退出时由内核兜底清理,避免孤儿进程占用显存。
    _enable_parent_death_signal()
    try:
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        seed = int(config["seed"]) + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # 长训练/机器高负载时 NCCL 心跳与 collective 可能被争抢拖慢,默认
        # 超时(30min)可能误判为死局;放宽到 2h 只影响等待判定,不改训练语义。
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            device_id=device,
            timeout=datetime.timedelta(seconds=7200),
        )
        learner_hp = {
            key: value
            for key, value in config.items()
            if key not in {"model_size", "device"}
        }
        learner = PPOLearner(
            model_size,
            f"cuda:{rank}",
            rank=rank,
            world_size=world_size,
            **learner_hp,
        )
        if resume_path:
            learner.load(resume_path)
        elif init_model_path:
            learner.load_model_weights(init_model_path)
        else:
            raise ValueError(
                "V18 PPO DDP learner requires --init-model or a resume checkpoint"
            )
        result_queue.put({"kind": "ready", "iteration": learner.iteration})
        while True:
            command = command_queue.get()
            kind = command["kind"]
            if kind == _CMD_UPDATE:
                # B2:shm 传输时按元数据零拷贝重建分片视图,update 结束
                # (含异常)即 close 本进程映射;unlink 由 driver 统一负责。
                view: ShardShmView | None = None
                try:
                    if "shard_shm" in command:
                        view = ShardShmView(command["shard_shm"])
                        transitions = view.buffer
                    else:
                        transitions = command["transitions"]
                    metrics = learner.update(
                        transitions,
                        shuffle_seed=command["shuffle_seed"],
                        advantages=command["advantages"],
                        returns=command["returns"],
                    )
                finally:
                    if view is not None:
                        view.close()
                result_queue.put({
                    "kind": "result",
                    "metrics": metrics,
                    "iteration": learner.iteration,
                })
            elif kind == _CMD_WEIGHTS:
                result_queue.put({"kind": "result", "weights": learner.weights()})
            elif kind == _CMD_RNG_STATE:
                result_queue.put({"kind": "result", "rng_state": learner.rng_state()})
            elif kind == _CMD_RELEASE_CACHE:
                learner.release_cache()
                result_queue.put({"kind": "result", "ok": True})
            elif kind == _CMD_SAVE:
                learner.save(
                    command["path"],
                    command["train_config"],
                    command["extra_state"],
                )
                result_queue.put({"kind": "result", "ok": True})
            elif kind == _CMD_SHUTDOWN:
                result_queue.put({"kind": "result", "ok": True})
                break
            else:
                raise ValueError(f"unknown learner command {kind!r}")
    except Exception as exc:
        try:
            result_queue.put({
                "kind": "error",
                "rank": rank,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class LearnerDDP:
    """driver 侧双卡 DDP learner 管理器(进程编排 + 指标汇总)。"""

    def __init__(
        self,
        model_size: str,
        device: str,
        world_size: int,
        *,
        config: dict[str, Any],
        resume: str | None = None,
        init_model: str | None = None,
    ) -> None:
        if world_size < 2:
            raise ValueError("LearnerDDP requires world_size >= 2")
        if not str(device).startswith("cuda"):
            raise ValueError("LearnerDDP requires a CUDA device")
        if resume and init_model:
            raise ValueError("resume and init_model are mutually exclusive")
        self.world_size = int(world_size)
        self.iteration = 0
        self._minibatch_size = int(config["minibatch_size"])
        self._gamma = float(config.get("gamma", 1.0))
        # update 时长 ∝ games×epochs,与 minibatch 大小无关;固定 600s 会
        # 误杀 2048 局配置的首个全量更新(基线实测 512 局 166-184s,
        # 外推 2048 局 665-735s > 600s),默认放宽到 1800s,可由配置
        # update_timeout_s 覆盖。启动就绪等待仍用 120s。
        self._update_timeout = float(config.get("update_timeout_s", 1800.0))
        if self._update_timeout <= 0:
            raise ValueError("update_timeout_s must be positive")
        # B2:shard IPC 传输方式。``pickle`` 为历史路径(mp.Queue feeder 线程
        # pickle → 管道 memcpy → learner unpickle,2.5GB 分片共 4 次拷贝);
        # ``shm`` 经 /dev/shm 传递 SoA 数组(driver 写入 1 次拷贝,learner
        # 零拷贝视图),数组逐位一致。默认 pickle,消融后定默认。
        self._shard_transport = str(config.get("learner_shard_transport", "pickle"))
        if self._shard_transport not in {"pickle", "shm"}:
            raise ValueError("learner_shard_transport must be 'pickle' or 'shm'")
        self._shm_seq = 0
        self._context = multiprocessing.get_context("spawn")
        self._command_queues = [
            self._context.Queue() for _ in range(self.world_size)
        ]
        self._result_queues = [
            self._context.Queue() for _ in range(self.world_size)
        ]
        self._processes: list[multiprocessing.Process] = []
        # 两个 spawn 子进程继承环境变量,经 127.0.0.1 + 动态端口建立 NCCL 组。
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_free_port())
        try:
            for rank in range(self.world_size):
                process = self._context.Process(
                    target=_learner_worker,
                    args=(
                        rank,
                        self.world_size,
                        model_size,
                        config,
                        resume,
                        init_model,
                        self._command_queues[rank],
                        self._result_queues[rank],
                    ),
                    name=f"riichi-ppo-ddp-learner-{rank}",
                )
                process.start()
                self._processes.append(process)
            ready = [self._recv(rank, timeout=120.0) for rank in range(self.world_size)]
            for message in ready:
                if message.get("kind") != "ready":
                    raise RuntimeError(
                        f"learner rank failed to become ready: {message.get('error')}"
                    )
            self.iteration = int(ready[0]["iteration"])
        except Exception:
            self.shutdown()
            raise

    def _recv(self, rank: int, timeout: float = 600.0) -> dict[str, Any]:
        queue = self._result_queues[rank]
        try:
            message = queue.get(timeout=timeout)
        except _queue.Empty as exc:
            raise RuntimeError(
                f"learner rank {rank} did not respond within {timeout:.0f}s"
            ) from exc
        if message.get("kind") == "error":
            raise RuntimeError(
                f"learner rank {rank} failed: {message.get('error')}\n"
                f"{message.get('traceback')}"
            )
        return message

    def _send_all(self, command: dict[str, Any]) -> None:
        for queue in self._command_queues:
            queue.put(command)

    def update(
        self,
        transitions: RolloutBuffer,
        *,
        shuffle_seed: int | None = None,
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        """分片更新;返回 ``(全局汇总指标, 逐 rank 指标)``。"""
        if not isinstance(transitions, RolloutBuffer):
            raise TypeError("LearnerDDP.update requires a RolloutBuffer")
        # B1 插计:driver 侧 update 未插计段分解(★ 纯计时)。四段先后串行:
        # 分片准备(learner_shard_indices + 全量 buffer 的 select gather 拷贝
        # + advantage/return 全量数学)→ 命令队列 put(shard pickle 传输的
        # 入队段;feeder 线程的序列化与管道写与 recv 等待重叠)→ 等待两个
        # rank 的结果(含 shard 传输 + rank 侧 learner_wall + 结果回传)→
        # 全量指标汇总(在关键路径上)。四段合计 ≈ update_wall 中 learner
        # 之外的 driver 侧开销。
        shard_started = time.perf_counter()
        index_shards = learner_shard_indices(
            len(transitions), self.world_size, self._minibatch_size,
        )
        shards = [transitions.select(shard) for shard in index_shards]
        advantages, returns, _return_mean, _return_std = rollout_update_targets(
            transitions,
        )
        # B2:shm 传输时把分片写入 /dev/shm(每 rank 一块),命令队列只传
        # 元数据;finally 统一 release(收到结果或任何异常路径都安全),
        # learner 侧在 update 结束后 close 自身映射。
        use_shm = self._shard_transport == "shm"
        writers: list[ShardShmWriter] = []
        try:
            if use_shm:
                shard_metas: list[dict[str, Any]] = []
                for rank, shard in enumerate(shards):
                    writer = ShardShmWriter(
                        shard_field_arrays(shard),
                        name=f"riichi-ppo-shard-{os.getpid()}-{self._shm_seq}-{rank}",
                    )
                    writers.append(writer)
                    shard_metas.append(writer.write())
                self._shm_seq += 1
            shard_select_s = time.perf_counter() - shard_started
            put_started = time.perf_counter()
            for rank in range(self.world_size):
                command: dict[str, Any] = {
                    "kind": _CMD_UPDATE,
                    "shuffle_seed": shuffle_seed,
                    "advantages": advantages[index_shards[rank]],
                    "returns": returns[index_shards[rank]],
                }
                if use_shm:
                    command["shard_shm"] = shard_metas[rank]
                else:
                    command["transitions"] = shards[rank]
                self._command_queues[rank].put(command)
            shard_put_s = time.perf_counter() - put_started
            recv_started = time.perf_counter()
            messages = [
                self._recv(rank, timeout=self._update_timeout)
                for rank in range(self.world_size)
            ]
            learner_recv_wait_s = time.perf_counter() - recv_started
            per_rank = [messages[rank]["metrics"] for rank in range(self.world_size)]
            self.iteration = int(messages[0]["iteration"])
            aggregate_started = time.perf_counter()
            metrics = aggregate_learner_metrics(
                per_rank, transitions, gamma=self._gamma,
            )
            aggregate_metrics_s = time.perf_counter() - aggregate_started
        finally:
            for writer in writers:
                writer.release()
        metrics.update({
            "update/shard_select_s": shard_select_s,
            "update/shard_put_s": shard_put_s,
            "update/learner_recv_wait_s": learner_recv_wait_s,
            "update/aggregate_metrics_s": aggregate_metrics_s,
        })
        return metrics, per_rank

    def release_cache(self) -> None:
        """评测前释放全部 rank 的缓存显存(见 ``PPOLearner.release_cache``)。"""
        self._send_all({"kind": _CMD_RELEASE_CACHE})
        for rank in range(self.world_size):
            self._recv(rank)

    def weights(self) -> dict[str, torch.Tensor]:
        self._command_queues[0].put({"kind": _CMD_WEIGHTS})
        message = self._recv(0)
        return message["weights"]

    def rng_states(self) -> list[dict[str, Any]]:
        self._send_all({"kind": _CMD_RNG_STATE})
        messages = [self._recv(rank) for rank in range(self.world_size)]
        return [messages[rank]["rng_state"] for rank in range(self.world_size)]

    def save(
        self,
        path: str,
        train_config: dict[str, Any],
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        merged = dict(extra_state or {})
        merged["rank_rng_states"] = self.rng_states()
        self._command_queues[0].put({
            "kind": _CMD_SAVE,
            "path": str(path),
            "train_config": train_config,
            "extra_state": merged,
        })
        self._recv(0)

    def shutdown(self) -> None:
        for queue in self._command_queues:
            try:
                queue.put({"kind": _CMD_SHUTDOWN}, timeout=5.0)
            except Exception:
                pass
        for process in self._processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for queue in (*self._command_queues, *self._result_queues):
            queue.close()
