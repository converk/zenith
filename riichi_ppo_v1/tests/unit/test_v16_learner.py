"""V16 PPO learner:调度、V16 collate、更新与检查点契约测试。"""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import pytest

from riichi_ppo_v1.model.schema import TOKEN_SCHEMA_VERSION
from riichi_ppo_v1.training.learner import (
    PPOLearner,
    approximate_kl_values,
    branch_grad_norms,
    collate,
    discounted_empirical_returns,
    length_bucketed_minibatches,
    materialize_host_batch,
    normalize_value_targets,
    scheduled_entropy_coefficient,
    scheduled_learning_rate,
    transition_length_metrics,
    transfer_batch_to_device,
    value_loss_values,
)
from riichi_ppo_v1.training.trajectory import Transition


def transition(value: float, *, action: int = 0, pairs: int = 2) -> Transition:
    """构造一个满足 forward_v16 形状约束的最小 V16 决策点。"""
    history_factors = np.zeros((2, 10), dtype=np.uint8)
    history_factors[0, 0] = 1
    snapshot_kinds = np.zeros(2, dtype=np.uint8)
    query_rows = np.zeros((2 * pairs, 15), dtype=np.int32)
    action_ids = np.zeros(pairs, dtype=np.int32)
    for pair, action_id in enumerate((0, 5)):
        query_rows[2 * pair, 0] = 1
        query_rows[2 * pair + 1, 0] = 2
        query_rows[2 * pair, 1] = action_id
        query_rows[2 * pair + 1, 1] = action_id
        action_ids[pair] = action_id
    legal = np.zeros(241, dtype=np.bool_)
    legal[[0, 5]] = True
    item = Transition(
        history_factors,
        np.zeros((2, 8), dtype=np.float32),
        2,
        snapshot_kinds,
        np.zeros((2, 4), dtype=np.uint8),
        np.zeros((2, 7), dtype=np.float32),
        2,
        query_rows,
        action_ids,
        pairs,
        legal,
        action,
        0.0,
        value,
        0.0,
        expected_q=value,
    )
    item.advantage = value
    item.q_target = value
    return item


def learner_kwargs(**overrides):
    defaults = dict(
        learning_rate=1e-4,
        profile_enabled=False,
        profile_cuda_sync=False,
        update_epochs=1,
        minibatch_size=1,
        ppo_clip=0.2,
        value_coef=0.5,
        q_coef=1.0,
        entropy_start=0.01,
        entropy_end=0.001,
        max_grad_norm=0.5,
        target_kl=0.0,
        total_updates=100,
        warmup_fraction=0.02,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-5,
        weight_decay=0.01,
        value_loss="huber",
        value_target_normalization="batch_std",
        value_target_std_floor=1e-2,
        gamma=1.0,
    )
    defaults.update(overrides)
    return defaults


def test_exp_style_learning_rate_schedule() -> None:
    assert np.isclose(scheduled_learning_rate(3e-4, update=1, total_updates=100, warmup_fraction=0.02), 1.5e-4)
    assert np.isclose(scheduled_learning_rate(3e-4, update=2, total_updates=100, warmup_fraction=0.02), 3e-4)
    assert np.isclose(scheduled_learning_rate(3e-4, update=100, total_updates=100, warmup_fraction=0.02), 3e-4 / 98)


def test_entropy_schedule_linearly_anneals() -> None:
    assert np.isclose(scheduled_entropy_coefficient(0.01, 0.001, update=0, total_updates=100), 0.01)
    assert np.isclose(scheduled_entropy_coefficient(0.01, 0.001, update=50, total_updates=100), 0.0055)
    assert np.isclose(scheduled_entropy_coefficient(0.01, 0.001, update=100, total_updates=100), 0.001)


def test_value_loss_supports_huber_and_mse() -> None:
    predicted = torch.tensor([0.0, 3.0])
    returns = torch.tensor([0.0, 0.0])
    torch.testing.assert_close(value_loss_values(predicted, returns, "mse"), torch.tensor([0.0, 9.0]))
    torch.testing.assert_close(value_loss_values(predicted, returns, "huber"), torch.tensor([0.0, 2.5]))


def test_batch_std_value_target_normalization_scales_prediction_and_target_together() -> None:
    predicted = torch.tensor([1.0, 5.0])
    returns = torch.tensor([3.0, 7.0])
    normalized_predicted, normalized_returns = normalize_value_targets(
        predicted, returns, mode="batch_std", mean=5.0, std=2.0, std_floor=0.01,
    )
    torch.testing.assert_close(normalized_predicted, torch.tensor([-2.0, 0.0]))
    torch.testing.assert_close(normalized_returns, torch.tensor([-1.0, 1.0]))


def test_discounted_empirical_returns_reset_at_kyoku_boundaries() -> None:
    rows = [transition(0.0) for _ in range(4)]
    for item, reward in zip(rows, (1.0, 2.0, 3.0, 4.0), strict=True):
        item.reward = reward
    rows[1].done = True
    rows[3].done = True
    np.testing.assert_allclose(
        discounted_empirical_returns(rows, 0.5),
        [2.0, 2.0, 5.0, 4.0],
    )


def test_approximate_kl_matches_exp_formula() -> None:
    old_logprob = torch.tensor([0.0, -0.5])
    new_logprob = torch.tensor([0.0, -0.25])
    log_ratio = new_logprob - old_logprob
    expected = (log_ratio.exp() - 1.0) - log_ratio
    torch.testing.assert_close(approximate_kl_values(new_logprob, old_logprob), expected)


def test_transition_length_metrics_report_v16_sequence_tokens() -> None:
    first = transition(0.0)
    second = transition(0.0)
    metrics = transition_length_metrics([first, second])
    length = 2 + 2 + 4
    assert metrics["update/buffer_transition_tokens_mean"] == length
    assert metrics["update/buffer_transition_input_tokens_max"] == length
    assert metrics["update/buffer_effective_input_tokens"] == 2 * length
    assert metrics["update/buffer_global_padding_input_tokens"] == 0.0


def test_length_bucketing_covers_all_rows_with_homogeneous_batches() -> None:
    transitions = [transition(0.0) for _ in range(8)]
    for index, item in enumerate(transitions):
        if index % 2:
            item.history_length = 7
    np.random.seed(7)
    batches = length_bucketed_minibatches(transitions, minibatch_size=2)

    assert sorted(index for batch in batches for index in batch) == list(range(len(transitions)))
    assert all(len(batch) <= 2 for batch in batches)
    assert all(
        len({transitions[int(index)].history_length for index in batch}) == 1
        for batch in batches
    )


def test_split_collate_transfers_all_v16_segments() -> None:
    transitions = [transition(0.2), transition(-0.1)]
    advantages = np.asarray([1.5, -0.5], dtype=np.float32)
    host = materialize_host_batch(transitions, advantages=advantages)
    split = transfer_batch_to_device(host, torch.device("cpu"))
    legacy = collate(transitions, torch.device("cpu"), advantages=advantages)
    assert set(split) == set(legacy)
    for name in split:
        torch.testing.assert_close(split[name], legacy[name])
    torch.testing.assert_close(split["advantages"], torch.tensor([1.5, -0.5]))


def test_v16_learner_rejects_legacy_model_sizes() -> None:
    with pytest.raises(ValueError, match="v16"):
        PPOLearner("mid", "cpu", **learner_kwargs())


def test_adamw_parameters_are_read_from_config() -> None:
    learner = PPOLearner(
        "v16",
        "cpu",
        **learner_kwargs(
            learning_rate=2e-4,
            adam_beta1=0.8,
            adam_beta2=0.95,
            adam_epsilon=1e-6,
            weight_decay=0.2,
        ),
    )
    group = learner.optimizer.param_groups[0]
    assert group["lr"] == 2e-4
    assert group["betas"] == (0.8, 0.95)
    assert group["eps"] == 1e-6
    assert group["weight_decay"] == 0.2


def test_update_batch_mode_validation() -> None:
    with pytest.raises(ValueError, match="update_batch_mode"):
        PPOLearner("v16", "cpu", **learner_kwargs(update_batch_mode="gpu_cache"))


def test_branch_learning_rates_are_scheduled_independently() -> None:
    learner = PPOLearner(
        "v16",
        "cpu",
        **learner_kwargs(
            actor_learning_rate=2e-5,
            shared_learning_rate=5e-6,
            critic_learning_rate=4e-5,
        ),
    )
    metrics = learner.update([transition(0.2), transition(-0.1)], shuffle_seed=3)
    assert metrics["system/actor_learning_rate"] == 1e-5
    assert metrics["system/shared_learning_rate"] == 2.5e-6
    assert metrics["system/critic_learning_rate"] == 2e-5


def test_cpu_update_completes_epochs_and_keeps_finite_metrics() -> None:
    learner = PPOLearner(
        "v16",
        "cpu",
        **learner_kwargs(update_epochs=2, minibatch_size=2, target_kl=0.0),
    )
    metrics = learner.update(
        [transition(0.2, action=0), transition(-0.1, action=5), transition(0.3, action=0)],
        shuffle_seed=7,
    )
    assert metrics["update/early_stop"] == 0.0
    assert metrics["update/configured_epochs"] == 2.0
    assert metrics["update/epochs_completed"] == 2.0
    assert metrics["update/executed_minibatches"] == 4.0
    assert metrics["update/executed_transition_samples"] == 6.0
    for name in ("loss", "policy_loss", "value_loss", "q_loss", "entropy"):
        assert np.isfinite(metrics[name]), name
    assert metrics["q_loss"] >= 0.0
    assert metrics["value_loss"] >= 0.0
    assert metrics["grad_norm_post_clip"] <= learner.hp["max_grad_norm"]
    assert {"grad_norm_actor", "grad_norm_critic", "grad_norm_shared"}.issubset(metrics)
    assert not learner.use_bf16


def test_checkpoint_records_v16_contract_and_restores_state() -> None:
    learner = PPOLearner("v16", "cpu", **learner_kwargs())
    learner.iteration = 7
    with TemporaryDirectory() as directory:
        path = f"{directory}/checkpoint.pt"
        learner.save(path, {"seed": 1})
        payload = torch.load(path, weights_only=False)
        assert payload["ppo_format_version"] == 3
        assert payload["token_schema_version"] == TOKEN_SCHEMA_VERSION
        assert set(payload) == {
            "ppo_format_version", "model", "optimizer", "model_config",
            "train_config", "iteration", "token_schema_version", "torch_rng",
            "cuda_rng", "python_rng", "numpy_rng", "extra_state",
        }
        restored = PPOLearner("v16", "cpu", **learner_kwargs())
        restored.load(path)
        assert restored.iteration == 7
        for name, value in learner.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[name], value)


def test_checkpoint_save_is_atomic_without_tmp_leftovers() -> None:
    learner = PPOLearner("v16", "cpu", **learner_kwargs())
    learner.iteration = 3
    with TemporaryDirectory() as directory:
        path = f"{directory}/checkpoint_00030.pt"
        learner.save(path, {"seed": 1})
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")
        payload = torch.load(path, weights_only=False)
        assert payload["iteration"] == 3


def test_load_model_weights_requires_v16_sft_contract() -> None:
    source = PPOLearner("v16", "cpu", **learner_kwargs())
    with TemporaryDirectory() as directory:
        path = Path(directory) / "sft.pt"
        torch.save({
            "model": source.weights(),
            "model_config": asdict(source.config),
            "training_stage": "sft",
            "training_mode": "actor_only",
        }, path)
        target = PPOLearner("v16", "cpu", **learner_kwargs())
        target.iteration = 99
        target.load_model_weights(path)
        assert target.iteration == 0
        for name, value in source.model.state_dict().items():
            torch.testing.assert_close(target.model.state_dict()[name], value)

        wrong = path.with_name("v13.pt")
        torch.save({
            "model": source.weights(),
            "model_config": {**asdict(source.config), "policy_head_type": "isolated_action_query"},
            "training_stage": "sft",
        }, wrong)
        with pytest.raises(RuntimeError, match="symmetric_action_query"):
            target.load_model_weights(wrong)


def test_branch_grad_norms_include_every_v16_branch() -> None:
    learner = PPOLearner("v16", "cpu", **learner_kwargs())
    for index, parameter in enumerate(learner.model.parameters(), start=1):
        parameter.grad = torch.full_like(parameter, float(index))
    norms = branch_grad_norms(learner.model)
    assert set(norms) == {"actor", "critic", "shared"}
    for value in norms.values():
        assert float(value) > 0.0
