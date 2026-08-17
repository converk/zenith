"""双卡 DDP learner:分片补齐、指标汇总与参数校验测试。"""

from __future__ import annotations

import numpy as np
import pytest

from riichi_ppo_v1.training.learner_ddp import (
    LearnerDDP,
    aggregate_learner_metrics,
    partition_learner_shards,
)

from .test_v16_learner import transition


def test_partition_keeps_all_rows_and_equal_batch_counts() -> None:
    rows = [transition(0.0) for _ in range(1025)]
    shards = partition_learner_shards(rows, world_size=2, minibatch_size=512)

    assert len(shards) == 2
    counts = [len(shard) for shard in shards]
    batch_counts = [
        (count + 511) // 512 for count in counts
    ]
    assert batch_counts[0] == batch_counts[1]
    assert sum(counts) - len(rows) <= 511
    originals = [row for shard in shards for row in shard]
    # 轮询切分:每个原始行恰好由某一个分片持有,补齐只复制已有对象。
    assert len({id(row) for row in originals}) == len(rows)
    assert all(len(shard) >= 1 for shard in shards)


def test_partition_single_rank_returns_input_unchanged() -> None:
    rows = [transition(0.0) for _ in range(3)]
    assert partition_learner_shards(rows, world_size=1, minibatch_size=2) == [rows]


def test_partition_rejects_too_few_rows_or_bad_sizes() -> None:
    with pytest.raises(ValueError, match="cannot split"):
        partition_learner_shards([transition(0.0)], world_size=2, minibatch_size=2)
    with pytest.raises(ValueError, match="minibatch_size"):
        partition_learner_shards([transition(0.0)], world_size=1, minibatch_size=0)


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
    aggregated = aggregate_learner_metrics(per_rank, rows)

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
    # buffer/Q 统计按完整 rollout 精确重算。
    assert aggregated["q_target_mean"] == pytest.approx(0.05)
    assert aggregated["update/buffer_transition_tokens_mean"] == 8.0


def test_learner_ddp_rejects_non_cuda_or_small_world_size() -> None:
    config = {"minibatch_size": 512, "seed": 1}
    with pytest.raises(ValueError, match="world_size"):
        LearnerDDP("v16", "cuda", 1, config=config)
    with pytest.raises(ValueError, match="CUDA"):
        LearnerDDP("v16", "cpu", 2, config=config)
