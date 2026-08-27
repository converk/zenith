"""RolloutBuffer 单路径的物化、分片、GAE 与 learner 契约测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from riichi_ppo_v1.tests.v18_fixtures import actor_inputs, critic_inputs
from riichi_ppo_v1.training.learner import PPOLearner, rollout_update_targets
from riichi_ppo_v1.training.rollout_buffer import RolloutBuffer
from riichi_ppo_v1.training.trajectory import Transition


def _random_transition(rng: np.random.Generator) -> Transition:
    pair_count = int(rng.integers(1, 8))
    action_ids = tuple(
        sorted(
            int(value)
            for value in rng.choice(np.arange(1, 80), size=(pair_count,), replace=False)
        )
    )
    actor = actor_inputs(batch=1, action_ids=action_ids)
    critic = critic_inputs(batch=1)
    actor_length = int(actor["actor_lengths"][0])
    critic_length = int(critic["critic_lengths"][0])
    return Transition(
        actor_factors=actor["actor_factors"][0, :actor_length].numpy().astype(np.int32),
        actor_numeric=actor["actor_numeric"][0, :actor_length].numpy().astype(np.float32),
        actor_length=actor_length,
        query_rows=actor["query_rows"][0, : 2 * pair_count].numpy().astype(np.int32),
        query_action_ids=actor["action_ids"][0, :pair_count].numpy().astype(np.int32),
        query_pair_counts=pair_count,
        legal_mask=actor["legal_mask"][0].numpy().astype(np.bool_),
        action=int(action_ids[0]),
        logprob=float(np.float32(rng.random())),
        value=float(np.float32(rng.random())),
        reward=float(np.float32(rng.random())),
        kyoku_reward=float(np.float32(rng.random())),
        done=bool(rng.integers(0, 2)),
        advantage=float(np.float32(rng.random())),
        critic_factors=critic["critic_factors"][0, :critic_length].numpy().astype(np.uint8),
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
    expected_tensor = torch.from_numpy(expected).to(dtype=actual.dtype)
    torch.testing.assert_close(actual[row, :length], expected_tensor)
    if length < actual.shape[1]:
        assert torch.count_nonzero(actual[row, length:]) == 0


def test_collate_preserves_all_segments_and_padding() -> None:
    transitions = _transitions(np.random.default_rng(0), 32)
    buffer = RolloutBuffer(transitions)
    indices = np.random.default_rng(1).permutation(len(transitions))[:16]
    batch = buffer.collate(indices)

    for row, index in enumerate(indices):
        item = transitions[int(index)]
        _assert_row_prefix_and_zero_padding(batch["actor_factors"], row, item.actor_factors)
        _assert_row_prefix_and_zero_padding(batch["actor_numeric"], row, item.actor_numeric)
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
        assert batch["actor_lengths"][row] == item.actor_length
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
    learner = PPOLearner("v18", "cpu", **kwargs)
    metrics = learner.update(RolloutBuffer(transitions), shuffle_seed=123)
    for name in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl"):
        assert np.isfinite(metrics[name]), name
    assert metrics["update/executed_minibatches"] == 3.0
    assert metrics["update/executed_transition_samples"] == 9.0
    with pytest.raises(TypeError, match="RolloutBuffer"):
        learner.update(transitions, shuffle_seed=123)  # type: ignore[arg-type]
