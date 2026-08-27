"""双卡 DDP learner:分片补齐、指标汇总与参数校验测试。"""

from __future__ import annotations

import ctypes
import multiprocessing
import numpy as np
import pytest
import signal
import sys

from riichi_ppo_v1.training.learner_ddp import (
    LearnerDDP,
    aggregate_learner_metrics,
    learner_shard_indices,
)
from riichi_ppo_v1.training.rollout_buffer import RolloutBuffer
from riichi_ppo_v1.training.trajectory import transition_sequence_length

from .test_rollout_buffer import _random_transition


def transition(reward: float, action: int = 0):
    item = _random_transition(np.random.default_rng(action + 17))
    item.reward = float(reward)
    item.advantage = float(reward)
    item.action = int(item.query_action_ids[0])
    return item


def _pdeathsig_probe(queue) -> None:
    """在独立子进程里设置并回读 PDEATHSIG,避免影响 pytest 进程自身。"""
    from riichi_ppo_v1.training.learner_ddp import _enable_parent_death_signal

    _enable_parent_death_signal()
    libc = ctypes.CDLL(None, use_errno=True)
    value = ctypes.c_ulong()
    ret = libc.prctl(2, ctypes.byref(value))  # PR_GET_PDEATHSIG
    queue.put((int(ret), int(value.value)))


def test_partition_keeps_all_rows_and_equal_batch_counts() -> None:
    rows = [transition(0.0) for _ in range(1025)]
    shards = learner_shard_indices(len(rows), world_size=2, minibatch_size=512)

    assert len(shards) == 2
    counts = [len(shard) for shard in shards]
    batch_counts = [
        (count + 511) // 512 for count in counts
    ]
    assert batch_counts[0] == batch_counts[1]
    assert sum(counts) - len(rows) <= 511
    originals = [index for shard in shards for index in shard]
    # 轮询切分:每个原始下标至少出现一次,补齐只重复已有下标。
    assert set(originals) == set(range(len(rows)))
    assert all(len(shard) >= 1 for shard in shards)


def test_partition_single_rank_returns_input_unchanged() -> None:
    assert learner_shard_indices(3, world_size=1, minibatch_size=2) == [[0, 1, 2]]


def test_partition_rejects_too_few_rows_or_bad_sizes() -> None:
    with pytest.raises(ValueError, match="cannot split"):
        learner_shard_indices(1, world_size=2, minibatch_size=2)
    with pytest.raises(ValueError, match="minibatch_size"):
        learner_shard_indices(1, world_size=1, minibatch_size=0)


def test_aggregate_metrics_weights_samples_and_steps() -> None:
    rows = [transition(0.2, action=0), transition(-0.1, action=5)]
    per_rank = [
        {
            "loss": 1.0,
            "grad_norm": 0.5,
            "update/executed_transition_samples": 10.0,
            "update/executed_minibatches": 2.0,
            "update/executed_transition_tokens_mean": 8.0,
            "update/executed_padded_input_tokens": 90.0,
            "update/executed_padding_input_tokens": 10.0,
            "update/configured_epochs": 4.0,
            "timing/update/model_forward/total_s": 1.0,
            "timing/update/model_forward/count": 2.0,
            "timing/update/model_forward/mean_ms": 500.0,
            "gpu/torch_memory_peak_allocated_mb": 100.0,
        },
        {
            "loss": 3.0,
            "grad_norm": 0.7,
            "update/executed_transition_samples": 30.0,
            "update/executed_minibatches": 6.0,
            "update/executed_transition_tokens_mean": 12.0,
            "update/executed_padded_input_tokens": 390.0,
            "update/executed_padding_input_tokens": 30.0,
            "update/configured_epochs": 4.0,
            "timing/update/model_forward/total_s": 1.4,
            "timing/update/model_forward/count": 6.0,
            "timing/update/model_forward/mean_ms": 233.333333,
            "gpu/torch_memory_peak_allocated_mb": 200.0,
        },
    ]
    aggregated = aggregate_learner_metrics(per_rank, RolloutBuffer(rows))

    assert aggregated["loss"] == pytest.approx(2.5)  # (10*1 + 30*3) / 40
    assert aggregated["grad_norm"] == pytest.approx(0.65)  # (2*0.5 + 6*0.7) / 8
    assert aggregated["update/executed_transition_samples"] == 40.0
    assert aggregated["update/executed_transition_tokens_mean"] == pytest.approx(11.0)
    assert aggregated["update/executed_padded_input_tokens"] == 480.0
    assert aggregated["timing/update/model_forward/total_s"] == pytest.approx(1.4)
    assert aggregated["timing/update/model_forward/count"] == 8.0
    assert aggregated["timing/update/model_forward/mean_ms"] == pytest.approx(175.0)
    assert aggregated["update/configured_epochs"] == 4.0
    assert aggregated["gpu/torch_memory_peak_allocated_mb"] == 100.0
    # buffer/advantage 统计按完整 rollout 精确重算。
    assert aggregated["advantage_mean"] == pytest.approx(0.05)
    assert aggregated["update/buffer_transition_tokens_mean"] == pytest.approx(
        np.mean([transition_sequence_length(row) for row in rows])
    )
    assert "value_explained_variance_lambda" in aggregated
    assert "mc_return_mean" in aggregated


def test_learner_ddp_rejects_non_cuda_or_small_world_size() -> None:
    config = {"minibatch_size": 512, "seed": 1}
    with pytest.raises(ValueError, match="world_size"):
        LearnerDDP("v18", "cuda", 1, config=config)
    with pytest.raises(ValueError, match="CUDA"):
        LearnerDDP("v18", "cpu", 2, config=config)


def test_enable_parent_death_signal_sets_pdeathsig() -> None:
    """DDP 子进程必须设置 PDEATHSIG,driver 异常消亡时由内核兜底清理。"""
    if not sys.platform.startswith("linux"):
        pytest.skip("PDEATHSIG 仅 Linux 可用")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_pdeathsig_probe, args=(queue,))
    process.start()
    process.join(timeout=30)
    assert process.exitcode == 0
    ret, pdeathsig = queue.get(timeout=10)
    assert ret == 0
    assert pdeathsig == signal.SIGKILL


def test_termination_handler_raises_system_exit() -> None:
    """外部 SIGTERM 必须转成 SystemExit,让 run() 的 finally 执行清理。"""
    from riichi_ppo_v1.training.train import _termination_handler

    with pytest.raises(SystemExit) as excinfo:
        _termination_handler(signal.SIGTERM, None)
    assert excinfo.value.code == 128 + signal.SIGTERM
