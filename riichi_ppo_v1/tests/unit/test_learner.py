import numpy as np
import torch
from tempfile import TemporaryDirectory

from riichi_ppo_v1.training.learner import (
    PPOLearner,
    approximate_kl_values,
    branch_grad_norms,
    collate,
    materialize_host_batch,
    normalize_value_targets,
    scheduled_entropy_coefficient,
    scheduled_learning_rate,
    transfer_batch_to_device,
    value_loss_values,
)
from riichi_ppo_v1.training.trajectory import Transition


def transition(value: float) -> Transition:
    item = Transition(
        np.asarray([[1, 1, 4, 1, 1, 1, 0, 0, 1, 1]], dtype=np.uint8),
        np.zeros((1, 8), dtype=np.float32), 1,
        np.eye(1, 241, 0, dtype=np.bool_)[0], 0, 0.0, value,
    )
    item.advantage = value
    item.return_ = value
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
        entropy_coef=0.01,
        max_grad_norm=0.5,
        target_kl=0.0,
        total_updates=100,
        warmup_fraction=0.02,
        entropy_start=0.01,
        entropy_end=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-5,
        weight_decay=0.01,
        value_loss="huber",
        value_target_normalization="batch_std",
        value_target_std_floor=1e-2,
    )
    defaults.update(overrides)
    return defaults


def test_exp_style_learning_rate_schedule() -> None:
    assert np.isclose(scheduled_learning_rate(3e-4, update=1, total_updates=100, warmup_fraction=0.02), 1.5e-4)
    assert np.isclose(scheduled_learning_rate(3e-4, update=2, total_updates=100, warmup_fraction=0.02), 3e-4)
    assert np.isclose(scheduled_learning_rate(3e-4, update=100, total_updates=100, warmup_fraction=0.02), 3e-4 / 98)


def test_learning_rate_schedule_without_warmup() -> None:
    assert np.isclose(scheduled_learning_rate(1.0, update=1, total_updates=4, warmup_fraction=0.0), 1.0)
    assert np.isclose(scheduled_learning_rate(1.0, update=4, total_updates=4, warmup_fraction=0.0), 0.25)
    assert np.isclose(scheduled_learning_rate(1.0, update=5, total_updates=4, warmup_fraction=0.0), 0.0)


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
    torch.testing.assert_close(
        value_loss_values(normalized_predicted, normalized_returns, "mse"), torch.tensor([1.0, 1.0]),
    )


def test_value_target_normalization_none_preserves_the_original_loss_space() -> None:
    predicted = torch.tensor([1.0, 5.0])
    returns = torch.tensor([3.0, 7.0])
    actual_predicted, actual_returns = normalize_value_targets(
        predicted, returns, mode="none", mean=5.0, std=2.0, std_floor=0.01,
    )
    torch.testing.assert_close(actual_predicted, predicted)
    torch.testing.assert_close(actual_returns, returns)


def test_value_target_normalization_uses_the_standard_deviation_floor() -> None:
    predicted = torch.tensor([1.0])
    returns = torch.tensor([1.0])
    normalized_predicted, normalized_returns = normalize_value_targets(
        predicted, returns, mode="batch_std", mean=1.0, std=0.0, std_floor=0.01,
    )
    assert torch.isfinite(normalized_predicted).all()
    assert torch.isfinite(normalized_returns).all()
    torch.testing.assert_close(normalized_predicted, torch.zeros(1))
    torch.testing.assert_close(normalized_returns, torch.zeros(1))


def test_branch_grad_norms_are_disjoint_and_include_every_model_branch() -> None:
    class BranchModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = torch.nn.Linear(1, 1, bias=False)
            self.public_backbone = torch.nn.Linear(1, 1, bias=False)
            self.actor_backbone = torch.nn.Linear(1, 1, bias=False)
            self.policy_head = torch.nn.Linear(1, 1, bias=False)
            self.query = torch.nn.Parameter(torch.zeros(1))
            self.critic_embedding = torch.nn.Linear(1, 1, bias=False)
            self.critic_backbone = torch.nn.Linear(1, 1, bias=False)
            self.value_head = torch.nn.Linear(1, 1, bias=False)
            self.value_query = torch.nn.Parameter(torch.zeros(1))

    model = BranchModel()
    for index, parameter in enumerate(model.parameters(), start=1):
        parameter.grad = torch.full_like(parameter, float(index))

    norms = branch_grad_norms(model)
    expected = {"actor": 0.0, "critic": 0.0, "shared": 0.0}
    for name, parameter in model.named_parameters():
        root = name.split(".", 1)[0]
        branch = "actor" if root in {"actor_backbone", "policy_head", "query"} else (
            "critic" if root in {"critic_embedding", "critic_backbone", "value_head", "value_query"} else "shared"
        )
        expected[branch] += float(parameter.grad.square().sum())
    for branch, total in expected.items():
        assert torch.isclose(norms[branch], torch.tensor(total).sqrt())


def test_approximate_kl_matches_exp_formula() -> None:
    old_logprob = torch.tensor([0.0, -0.5])
    new_logprob = torch.tensor([0.0, -0.25])
    log_ratio = new_logprob - old_logprob
    expected = (log_ratio.exp() - 1.0) - log_ratio
    torch.testing.assert_close(approximate_kl_values(new_logprob, old_logprob), expected)


def test_adamw_parameters_are_read_from_config() -> None:
    learner = PPOLearner(
        "mid",
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
    try:
        PPOLearner("mid", "cpu", **learner_kwargs(update_batch_mode="unknown"))
    except ValueError as exc:
        assert "update_batch_mode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid update_batch_mode must be rejected")


def test_split_collate_matches_legacy_collate_on_cpu() -> None:
    transitions = [transition(0.2), transition(-0.1)]
    advantages = np.asarray([1.5, -0.5], dtype=np.float32)
    host = materialize_host_batch(transitions, advantages=advantages)
    split = transfer_batch_to_device(host, torch.device("cpu"))
    legacy = collate(transitions, torch.device("cpu"), advantages=advantages)
    assert set(split) == set(legacy)
    for name in split:
        torch.testing.assert_close(split[name], legacy[name])
    torch.testing.assert_close(split["advantages"], torch.tensor([1.5, -0.5]))


def test_cpu_batch_modes_fall_back_to_streaming() -> None:
    modes = ("streaming", "prefetch", "gpu_cache", "auto")
    for mode in modes:
        torch.manual_seed(3)
        learner = PPOLearner(
            "mid",
            "cpu",
            **learner_kwargs(update_epochs=1, minibatch_size=2, update_batch_mode=mode),
        )
        metrics = learner.update([transition(0.2), transition(-0.1), transition(0.3)], shuffle_seed=7)
        assert metrics["update/batch_mode_id"] == 0.0
        assert metrics["update/executed_minibatches"] == 2.0
        assert metrics["value_loss_raw"] >= 0.0
        assert metrics["value_target_std"] >= 0.0
        assert metrics["value_prediction_std"] >= 0.0
        assert metrics["value_explained_variance"] == metrics["explained_variance"]
        assert metrics["grad_norm_post_clip"] <= learner.hp["max_grad_norm"]
        assert {"grad_norm_actor", "grad_norm_critic", "grad_norm_shared"}.issubset(metrics)


def test_target_kl_zero_completes_all_ppo_epochs() -> None:
    learner = PPOLearner("mid", "cpu", **learner_kwargs(update_epochs=2, minibatch_size=2, target_kl=0.0))
    metrics = learner.update([transition(0.2), transition(-0.1), transition(0.3)])
    assert metrics["update/early_stop"] == 0.0
    assert metrics["update/configured_epochs"] == 2.0
    assert metrics["update/epochs_completed"] == 2.0
    assert metrics["update/planned_minibatches"] == 4.0
    assert metrics["update/executed_minibatches"] == 4.0
    assert metrics["update/executed_transition_samples"] == 6.0


def test_cpu_update_keeps_fp32_parameters_and_disables_bf16_autocast() -> None:
    learner = PPOLearner("mid", "cpu", **learner_kwargs())
    assert not learner.use_bf16
    assert {parameter.dtype for parameter in learner.model.parameters()} == {torch.float32}


def test_checkpoint_records_v8_schema_metadata_and_restores_state() -> None:
    learner = PPOLearner("mid", "cpu", **learner_kwargs())
    learner.iteration = 7
    with TemporaryDirectory() as directory:
        path = f"{directory}/checkpoint.pt"
        learner.save(path, {"seed": 1}, {
            "reward_scale_controller": {"discard_weight": 0.6},
        })
        payload = torch.load(path, weights_only=False)
        assert set(payload) == {
            "model", "optimizer", "model_config", "train_config", "iteration",
            "torch_rng", "cuda_rng", "python_rng", "numpy_rng", "token_schema_version", "extra_state",
        }
        assert payload["token_schema_version"] == 8
        assert payload["extra_state"]["reward_scale_controller"]["discard_weight"] == 0.6

        restored = PPOLearner("mid", "cpu", **learner_kwargs())
        restored.load(path)
        assert restored.iteration == 7
        for name, value in learner.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[name], value)


def test_checkpoint_requires_a_complete_model_state() -> None:
    learner = PPOLearner("mid", "cpu", **learner_kwargs())
    with TemporaryDirectory() as directory:
        path = f"{directory}/incomplete.pt"
        torch.save({"model": {}}, path)
        try:
            learner.load(path)
        except RuntimeError:
            pass
        else:  # pragma: no cover
            raise AssertionError("incomplete model state must be rejected")
