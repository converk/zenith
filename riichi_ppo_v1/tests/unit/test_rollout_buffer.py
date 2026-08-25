"""RolloutBuffer 单路径的物化、分片、GAE 与 learner 契约测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from riichi_ppo_v1.training.learner import PPOLearner, rollout_update_targets
from riichi_ppo_v1.training.rollout_buffer import RolloutBuffer
from riichi_ppo_v1.training.trajectory import Transition


def _random_transition(rng: np.random.Generator) -> Transition:
    history_length = int(rng.integers(20, 60))
    snapshot_length = int(rng.integers(2, 8))
    pair_count = int(rng.integers(1, 8))
    critic_length = int(rng.integers(0, 30))
    action_ids = rng.integers(0, 10, size=(pair_count,)).astype(np.int32)
    query_rows = np.zeros((2 * pair_count, 15), dtype=np.int32)
    for pair, action_id in enumerate(action_ids):
        query_rows[2 * pair, 0] = 1
        query_rows[2 * pair + 1, 0] = 2
        query_rows[2 * pair, 1] = int(action_id)
        query_rows[2 * pair + 1, 1] = int(action_id)
        query_rows[2 * pair, 5:] = rng.integers(0, 6, size=(10,))
        query_rows[2 * pair + 1, 5:] = rng.integers(0, 6, size=(10,))
    legal_mask = np.zeros(241, dtype=np.bool_)
    legal_mask[action_ids] = True
    return Transition(
        history_factors=rng.integers(0, 4, (history_length, 10)).astype(np.uint8),
        history_numeric=rng.random((history_length, 8)).astype(np.float32),
        history_length=history_length,
        snapshot_kinds=rng.integers(0, 4, (snapshot_length,)).astype(np.uint8),
        snapshot_cat=rng.integers(0, 4, (snapshot_length, 4)).astype(np.uint8),
        snapshot_num=rng.random((snapshot_length, 7)).astype(np.float32),
        snapshot_length=snapshot_length,
        query_rows=query_rows,
        query_action_ids=action_ids,
        query_pair_counts=pair_count,
        legal_mask=legal_mask,
        action=int(action_ids[0]),
        logprob=float(np.float32(rng.random())),
        value=float(np.float32(rng.random())),
        reward=float(np.float32(rng.random())),
        kyoku_reward=float(np.float32(rng.random())),
        done=bool(rng.integers(0, 2)),
        advantage=float(np.float32(rng.random())),
        critic_factors=(
            rng.integers(0, 4, (critic_length, 10)).astype(np.uint8)
            if critic_length
            else None
        ),
        critic_length=critic_length,
    )


def _transitions(rng: np.random.Generator, count: int) -> list[Transition]:
    return [_random_transition(rng) for _ in range(count)]


def _assert_buffers_equal(left: RolloutBuffer, right: RolloutBuffer) -> None:
    assert left.size == right.size
    left_arrays = {
        name: value for name, value in vars(left).items()
        if isinstance(value, np.ndarray)
    }
    right_arrays = {
        name: value for name, value in vars(right).items()
        if isinstance(value, np.ndarray)
    }
    assert set(left_arrays) == set(right_arrays)
    for name in left_arrays:
        np.testing.assert_array_equal(left_arrays[name], right_arrays[name], err_msg=name)


def _assert_row_prefix_and_zero_padding(
    actual: torch.Tensor,
    row: int,
    expected: np.ndarray,
) -> None:
    length = expected.shape[0]
    torch.testing.assert_close(actual[row, :length], torch.from_numpy(expected))
    if length < actual.shape[1]:
        assert torch.count_nonzero(actual[row, length:]) == 0


def test_collate_preserves_all_segments_and_padding() -> None:
    transitions = _transitions(np.random.default_rng(0), 32)
    buffer = RolloutBuffer(transitions)
    indices = np.random.default_rng(1).permutation(len(transitions))[:16]
    batch = buffer.collate(indices)

    for row, index in enumerate(indices):
        item = transitions[int(index)]
        _assert_row_prefix_and_zero_padding(batch["history_factors"], row, item.history_factors)
        _assert_row_prefix_and_zero_padding(batch["history_numeric"], row, item.history_numeric)
        _assert_row_prefix_and_zero_padding(batch["snapshot_kinds"], row, item.snapshot_kinds)
        _assert_row_prefix_and_zero_padding(batch["snapshot_cat"], row, item.snapshot_cat)
        _assert_row_prefix_and_zero_padding(batch["snapshot_num"], row, item.snapshot_num)
        _assert_row_prefix_and_zero_padding(batch["query_rows"], row, item.query_rows)
        _assert_row_prefix_and_zero_padding(
            batch["query_action_ids"], row, item.query_action_ids,
        )
        expected_critic = (
            item.critic_factors
            if item.critic_factors is not None
            else np.zeros((0, 10), dtype=np.uint8)
        )
        _assert_row_prefix_and_zero_padding(batch["critic_factors"], row, expected_critic)
        torch.testing.assert_close(batch["legal_mask"][row], torch.from_numpy(item.legal_mask))
        assert batch["history_lengths"][row] == item.history_length
        assert batch["snapshot_lengths"][row] == item.snapshot_length
        assert batch["query_pair_counts"][row] == item.query_pair_counts
        assert batch["critic_lengths"][row] == item.critic_length
        assert batch["actions"][row] == item.action
        assert batch["old_logprobs"][row] == np.float32(item.logprob)
        assert batch["advantages"][row] == np.float32(item.advantage)


def test_bucketed_minibatches_are_deterministic_and_cover_all_rows() -> None:
    buffer = RolloutBuffer(_transitions(np.random.default_rng(3), 1500))
    left = buffer.bucketed_minibatches(512, rng=np.random.default_rng(42))
    right = buffer.bucketed_minibatches(512, rng=np.random.default_rng(42))
    assert len(left) == len(right) == 3
    assert sorted(int(index) for batch in left for index in batch) == list(range(len(buffer)))
    for first, second in zip(left, right, strict=True):
        np.testing.assert_array_equal(first, second)
        assert len(first) <= 512
        assert np.all(np.diff(buffer.sequence_lengths[first]) >= 0)


def test_concatenate_and_select_are_elementwise_exact() -> None:
    transitions = _transitions(np.random.default_rng(19), 37)
    merged = RolloutBuffer.concatenate([
        RolloutBuffer(transitions[:17]), RolloutBuffer(transitions[17:]),
    ])
    _assert_buffers_equal(merged, RolloutBuffer(transitions))
    array_count, array_bytes = merged.payload_stats()
    assert array_count < len(transitions)
    assert array_bytes > 0

    indices = [36, 0, 18, 18, 7, 29]
    _assert_buffers_equal(
        merged.select(indices),
        RolloutBuffer([transitions[index] for index in indices]),
    )


def test_rollout_target_math_matches_frozen_formula() -> None:
    transitions = _transitions(np.random.default_rng(23), 31)
    buffer = RolloutBuffer(transitions)
    advantages, returns, return_mean, return_std = rollout_update_targets(buffer, gamma=0.97)

    raw_advantages = np.asarray([item.advantage for item in transitions], dtype=np.float32)
    expected_advantages = (
        (raw_advantages - raw_advantages.mean(dtype=np.float64))
        / (raw_advantages.std(dtype=np.float64) + 1e-8)
    ).astype(np.float32)
    expected_returns = np.zeros(len(transitions), dtype=np.float32)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        if transitions[index].done:
            running = 0.0
        running = float(transitions[index].reward) + 0.97 * running
        expected_returns[index] = np.float32(running)
    np.testing.assert_array_equal(advantages, expected_advantages)
    np.testing.assert_array_equal(returns, expected_returns)
    assert return_mean == float(expected_returns.mean(dtype=np.float64))
    assert return_std == float(expected_returns.std(dtype=np.float64))


def test_learner_accepts_only_rollout_buffer() -> None:
    transitions = _transitions(np.random.default_rng(7), 9)
    kwargs = {
        "learning_rate": 1e-4,
        "profile_enabled": False,
        "profile_cuda_sync": False,
        "update_epochs": 1,
        "minibatch_size": 3,
        "ppo_clip": 0.2,
        "value_coef": 0.5,
        "entropy_start": 0.01,
        "entropy_end": 0.001,
        "max_grad_norm": 0.5,
        "target_kl": 0.0,
        "total_updates": 50,
        "warmup_fraction": 0.02,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-5,
        "weight_decay": 0.01,
        "value_loss": "huber",
        "value_target_normalization": "batch_std",
        "value_target_std_floor": 1e-2,
        "gamma": 1.0,
        "critic_bootstrap_updates": 0,
    }
    learner = PPOLearner("v16", "cpu", **kwargs)
    metrics = learner.update(RolloutBuffer(transitions), shuffle_seed=123)
    for name in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl"):
        assert np.isfinite(metrics[name]), name
    assert metrics["update/executed_minibatches"] == 3.0
    assert metrics["update/executed_transition_samples"] == 9.0
    with pytest.raises(TypeError, match="RolloutBuffer"):
        learner.update(transitions, shuffle_seed=123)  # type: ignore[arg-type]
