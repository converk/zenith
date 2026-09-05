"""V19 PPO 信念联合损失单测:loss 前向/backward、actor 参数组与 scale 传参。"""

from __future__ import annotations

import numpy as np
import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.tests.v18_fixtures import actor_inputs, critic_inputs
from riichi_ppo_v1.training.learner import PPOLearner
from riichi_ppo_v1.training.rollout_buffer import RolloutBuffer
from riichi_ppo_v1.training.trajectory import Transition


def _transition(rng: np.random.Generator, row: int) -> Transition:
    action_ids = (1, 7, 12)
    actor = actor_inputs(batch=2, action_ids=action_ids)
    critic = critic_inputs(batch=2)
    actor_length = int(actor["actor_lengths"][0])
    critic_length = int(critic["critic_lengths"][0])
    return Transition(
        actor_factors=actor["actor_factors"][0, :actor_length].numpy().astype(np.int32),
        actor_numeric=actor["actor_numeric"][0, :actor_length].numpy().astype(np.float32),
        actor_length=actor_length,
        query_action_ids=actor["action_ids"][0, :len(action_ids)].numpy().astype(np.int32),
        query_pair_counts=len(action_ids),
        legal_mask=actor["legal_mask"][0].numpy().astype(np.bool_),
        action=int(action_ids[row % len(action_ids)]),
        logprob=float(np.float32(rng.random())),
        value=float(np.float32(rng.random())),
        reward=float(np.float32(rng.random())),
        kyoku_reward=float(np.float32(rng.random())),
        done=False,
        advantage=float(np.float32(rng.random())),
        critic_factors=critic["critic_factors"][0, :critic_length].numpy().astype(np.uint8),
        critic_length=critic_length,
        # 合法合成信念标签:hand 0..4、shanten 0..8、wait/danger 0/1、
        # loss 用 0..24000 的原始点数(训练侧归一化)。
        belief_hand=rng.integers(0, 5, size=102, dtype=np.uint8),
        belief_shanten=rng.integers(0, 9, size=3, dtype=np.uint8),
        belief_wait=rng.integers(0, 2, size=105, dtype=np.uint8),
        belief_danger=rng.integers(0, 2, size=102, dtype=np.uint8),
        belief_loss=(rng.random(102) * 24000.0).astype(np.float32),
    )


def _learner_kwargs() -> dict[str, object]:
    return {
        "learning_rate": 1e-4,
        "profile_enabled": False,
        "torch_compile": False,
        "update_epochs": 1,
        "minibatch_size": 3,
        "ppo_clip": 0.2,
        "value_coef": 0.5,
        "entropy_start": 0.01,
        "entropy_middle": 0.005,
        "entropy_end": 0.001,
        "entropy_middle_fraction": 0.5,
        "entropy_loss_mode": "normalized",
        "max_grad_norm": 0.5,
        "actor_max_grad_norm": 0.5,
        "shared_max_grad_norm": 0.5,
        "critic_max_grad_norm": 1.0,
        "target_kl": 0.0,
        "target_kl_check_interval": 8,
        "total_updates": 50,
        "warmup_fraction": 0.02,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-5,
        "weight_decay": 0.0,
        "value_loss": "huber",
        "value_target_normalization": "batch_std",
        "value_target_std_floor": 1e-2,
        "gamma": 1.0,
        "critic_bootstrap_updates": 0,
        "critic_private_embedding_grad_scale": 0.25,
        "belief_public_grad_scale": 0.25,
        "belief_head_weight_hand": 1.0,
        "belief_head_weight_shanten": 1.0,
        "belief_head_weight_wait": 1.0,
        "belief_head_weight_danger": 1.0,
        "belief_head_weight_loss": 1.0,
        "belief_wait_danger_weight": 0.05,
        "bucket_window_multiplier": 8,
    }


def test_belief_loss_forward_backward_finite_and_metrics() -> None:
    """合成标签下一次完整 update:信念损失有限、指标齐全、反向不崩溃。"""
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)
    transitions = [_transition(rng, row) for row in range(6)]
    buffer = RolloutBuffer(transitions)
    learner = PPOLearner("v19", "cpu", **_learner_kwargs())
    metrics = learner.update(buffer, shuffle_seed=7)
    for name in (
        "belief/total_loss", "belief/hand_accuracy", "belief/shanten_top1",
        "belief/wait_auc", "belief/wait_precision_at_5", "belief/danger_auc",
        "belief/loss_mae",
    ):
        assert name in metrics, name
        assert np.isfinite(metrics[name]), name
    assert metrics["belief/total_loss"] >= 0.0
    assert "loss" in metrics and np.isfinite(metrics["loss"])


def test_belief_parameters_in_actor_group() -> None:
    """信念网络/转换矩阵参数必须归入 actor 优化器组(不新增独立 belief LR)。"""
    learner = PPOLearner("v19", "cpu", **_learner_kwargs())
    actor_ids = {id(parameter) for parameter in learner.branch_parameters["actor"]}
    belief_parameters = [
        parameter
        for name, parameter in learner.model.named_parameters()
        if name.startswith("belief_network.")
    ]
    assert belief_parameters
    assert all(id(parameter) in actor_ids for parameter in belief_parameters)
    # 无独立 belief 学习率分支。
    assert not any(group.get("branch") == "belief" for group in learner.optimizer.param_groups)


def test_belief_public_grad_scale_is_forwarded(monkeypatch) -> None:
    """PPOLearner 把配置的 belief_public_grad_scale 透传给模型 forward。"""
    learner = PPOLearner("v19", "cpu", **_learner_kwargs())
    captured: dict[str, float | None] = {}

    original_forward = learner.model.forward

    def wrapper(*args, **kwargs):
        captured["scale"] = kwargs.get("belief_public_grad_scale")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(learner.model, "forward", wrapper)
    rng = np.random.default_rng(9)
    buffer = RolloutBuffer([_transition(rng, row) for row in range(6)])
    learner.update(buffer, shuffle_seed=1)
    assert captured["scale"] == 0.25
