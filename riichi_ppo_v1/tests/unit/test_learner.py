import numpy as np
import torch
from dataclasses import asdict
from riichi_ppo_v1.model.feature_schema import DECISION_ANALYSIS_VERSION, RUST_ANALYSIS_VERSION, feature_schema_sha256
from riichi_ppo_v1.model.schema import TOKEN_SCHEMA_VERSION
from tempfile import TemporaryDirectory

from riichi_ppo_v1.training.learner import (
    PPOLearner,
    approximate_kl_values,
    branch_grad_norms,
    categorical_kl_values,
    collate,
    discounted_empirical_returns,
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


def test_categorical_kl_matches_masked_distribution_and_is_zero_at_reference() -> None:
    policy = torch.tensor([[1.0, 0.0, float("-inf")]], requires_grad=True)
    reference = torch.tensor([[0.0, 1.0, float("-inf")]])
    actual = categorical_kl_values(policy, reference)
    probability = torch.softmax(policy[:, :2], dim=-1)
    expected = (
        probability
        * (
            torch.log_softmax(policy[:, :2], dim=-1)
            - torch.log_softmax(reference[:, :2], dim=-1)
        )
    ).sum(-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(categorical_kl_values(policy, policy), torch.zeros(1))
    actual.sum().backward()
    assert torch.isfinite(policy.grad).all()


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


def test_branch_learning_rates_are_scheduled_independently() -> None:
    learner = PPOLearner(
        "mid",
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


def test_critic_public_gradient_scale_validation() -> None:
    try:
        PPOLearner(
            "mid", "cpu", **learner_kwargs(critic_public_grad_scale=1.1)
        )
    except ValueError as exc:
        assert "critic_public_grad_scale" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid critic_public_grad_scale must be rejected")


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


def test_checkpoint_records_schema_metadata_and_restores_state() -> None:
    learner = PPOLearner("mid", "cpu", **learner_kwargs())
    learner.iteration = 7
    with TemporaryDirectory() as directory:
        path = f"{directory}/checkpoint.pt"
        learner.save(path, {"seed": 1})
        payload = torch.load(path, weights_only=False)
        assert set(payload) == {
            "model", "optimizer", "model_config", "train_config", "iteration",
            "torch_rng", "cuda_rng", "python_rng", "numpy_rng", "token_schema_version", "extra_state",
            "ppo_format_version", "feature_schema_sha256", "rust_analysis_version",
            "decision_analysis_version", "policy_head_type",
        }
        assert payload["token_schema_version"] == TOKEN_SCHEMA_VERSION
        assert payload["ppo_format_version"] == 2

        restored = PPOLearner("mid", "cpu", **learner_kwargs())
        restored.load(path)
        assert restored.iteration == 7
        for name, value in learner.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[name], value)


def test_model_initialization_starts_joint_ppo() -> None:
    kwargs = learner_kwargs(
        gamma=0.99,
        update_epochs=1,
        minibatch_size=2,
    )
    source = PPOLearner("mid", "cpu", **kwargs)
    with TemporaryDirectory() as directory:
        path = f"{directory}/actor_only.pt"
        torch.save({
            "model": source.weights(),
            "token_schema_version": TOKEN_SCHEMA_VERSION,
            "feature_schema_sha256": feature_schema_sha256(),
            "rust_analysis_version": RUST_ANALYSIS_VERSION,
            "decision_analysis_version": DECISION_ANALYSIS_VERSION,
            "training_stage": "sft",
            "training_mode": "actor_only",
            "model_config": asdict(source.config),
        }, path)
        learner = PPOLearner("mid", "cpu", **kwargs)
        learner.load_model_weights(path)
        before = {
            name: value.detach().clone()
            for name, value in learner.model.state_dict().items()
        }
        rows = [transition(0.2), transition(-0.1)]
        for row, reward in zip(rows, (1.0, -1.0), strict=True):
            row.legal_mask[:2] = True
            row.reward = reward
            row.done = True
        metrics = learner.update(rows, shuffle_seed=3)
        after_warmup = learner.model.state_dict()
        assert any(
            not torch.equal(value, before[name])
            for name, value in after_warmup.items()
        )
        resume_path = f"{directory}/resume.pt"
        learner.save(resume_path, {"seed": 1})
        resumed = PPOLearner("mid", "cpu", **kwargs)
        resumed.load(resume_path)
        assert resumed.iteration == 1
        actor_before_joint = resumed.model.policy_head.weight.detach().clone()
        joint_metrics = resumed.update(rows, shuffle_seed=4)
        assert joint_metrics["training/policy_update"] == 2.0
        assert not torch.equal(resumed.model.policy_head.weight, actor_before_joint)


def test_actor_only_sft_initialization_bootstraps_critic_before_policy() -> None:
    kwargs = learner_kwargs(
        critic_bootstrap_updates=1,
        critic_bootstrap_learning_rate=1e-4,
        sft_kl_coef_start=0.02,
        sft_kl_coef_end=0.0,
        sft_kl_anneal_updates=10,
        zero_value_head_on_sft_init=True,
        update_epochs=1,
        minibatch_size=2,
    )
    source = PPOLearner("mid", "cpu", **kwargs)
    with TemporaryDirectory() as directory:
        path = f"{directory}/actor_only.pt"
        torch.save({
            "model": source.weights(),
            "token_schema_version": TOKEN_SCHEMA_VERSION,
            "feature_schema_sha256": feature_schema_sha256(),
            "rust_analysis_version": RUST_ANALYSIS_VERSION,
            "decision_analysis_version": DECISION_ANALYSIS_VERSION,
            "training_stage": "sft",
            "training_mode": "actor_only",
            "model_config": asdict(source.config),
        }, path)
        learner = PPOLearner("mid", "cpu", **kwargs)
        learner.load_model_weights(path)
        assert learner.reference_model is not None
        assert not any(parameter.requires_grad for parameter in learner.reference_model.parameters())
        torch.testing.assert_close(
            learner.model.value_head.weight,
            torch.zeros_like(learner.model.value_head.weight),
        )
        actor_before = learner.model.policy_head.weight.detach().clone()
        shared_before = learner.model.token_embedding.table.weight.detach().clone()
        critic_before = learner.model.value_head.weight.detach().clone()
        rows = [transition(1.0), transition(-1.0)]
        for row in rows:
            row.legal_mask[:2] = True
        rows[1].action = 1
        metrics = learner.update(rows, shuffle_seed=3)
        assert metrics["training/critic_bootstrap"] == 1.0
        assert metrics["training/policy_update"] == 0.0
        assert metrics["system/sft_kl_coef"] == 0.0
        assert metrics["system/learning_rate"] == 0.0
        assert metrics["system/actor_learning_rate"] == 0.0
        assert metrics["system/shared_learning_rate"] == 0.0
        assert metrics["system/critic_learning_rate"] == 1e-4
        assert metrics["system/critic_public_grad_scale"] == 0.0
        torch.testing.assert_close(learner.model.policy_head.weight, actor_before)
        torch.testing.assert_close(learner.model.token_embedding.table.weight, shared_before)
        assert not torch.equal(learner.model.value_head.weight, critic_before)
        joint_metrics = learner.update(rows, shuffle_seed=4)
        assert joint_metrics["training/critic_bootstrap"] == 0.0
        assert joint_metrics["training/policy_update"] == 1.0
        assert joint_metrics["system/sft_kl_coef"] > 0.0
        assert not torch.equal(learner.model.policy_head.weight, actor_before)


def test_anchored_ppo_checkpoint_restores_frozen_sft_reference() -> None:
    kwargs = learner_kwargs(sft_kl_coef_start=0.02, sft_kl_coef_end=0.0)
    source = PPOLearner("mid", "cpu", **kwargs)
    with TemporaryDirectory() as directory:
        sft_path = f"{directory}/sft.pt"
        torch.save({
            "model": source.weights(),
            "token_schema_version": TOKEN_SCHEMA_VERSION,
            "feature_schema_sha256": feature_schema_sha256(),
            "rust_analysis_version": RUST_ANALYSIS_VERSION,
            "decision_analysis_version": DECISION_ANALYSIS_VERSION,
            "training_stage": "sft",
            "training_mode": "actor_only",
            "model_config": asdict(source.config),
        }, sft_path)
        source.load_model_weights(sft_path)
        checkpoint_path = f"{directory}/ppo.pt"
        source.save(checkpoint_path, {"seed": 1})
        restored = PPOLearner("mid", "cpu", **kwargs)
        restored.load(checkpoint_path)
        assert restored.reference_model is not None
        for name, value in source.reference_model.state_dict().items():
            torch.testing.assert_close(restored.reference_model.state_dict()[name], value)
        assert not any(parameter.requires_grad for parameter in restored.reference_model.parameters())


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
