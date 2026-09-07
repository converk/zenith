"""V19 PPO 熵地板屏障与逐损失项梯度归因诊断单测。

探索保持方案(2026-09-07)配套:entropy_floor_penalty 屏障数学、
loss_term_grad_norms 逐项归因,以及完整 update 下的指标接线。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from riichi_ppo_v1.tests.unit.test_v19_learner_belief_loss import (
    _learner_kwargs,
    _transition,
)
from riichi_ppo_v1.training.learner import (
    PPOLearner,
    entropy_floor_penalty,
    loss_term_grad_norms,
)
from riichi_ppo_v1.training.rollout_buffer import RolloutBuffer


def _buffer(seed: int, rows: int = 6) -> RolloutBuffer:
    rng = np.random.default_rng(seed)
    return RolloutBuffer([_transition(rng, row) for row in range(rows)])


def test_entropy_floor_penalty_gate_and_gradient() -> None:
    """高于地板屏障为常数 0 且梯度为零;跌破地板为正且梯度方向为升熵。"""
    logits = torch.tensor(
        [[2.0, 0.5, -0.3], [1.0, 0.2, -0.1], [0.3, 0.3, 0.3], [1.5, -0.5, 0.0]],
        requires_grad=True,
    )
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    entropy_values = -(log_probabilities * probabilities).sum(-1)
    entropy_mean = float(entropy_values.mean())
    assert entropy_mean > 0.0

    # 高于地板:惩罚恒 0,反向梯度为零。
    zero_penalty = entropy_floor_penalty(entropy_values, entropy_mean * 0.5)
    assert float(zero_penalty) == 0.0
    zero_penalty.backward(retain_graph=True)
    assert float(logits.grad.abs().sum()) == 0.0
    logits.grad = None

    # 跌破地板:惩罚 = floor - batch 均 raw 熵,沿梯度一步后熵上升。
    floor = entropy_mean + 0.5
    penalty = entropy_floor_penalty(entropy_values, floor)
    assert abs(float(penalty) - 0.5) < 1e-6
    penalty.backward()
    assert float(logits.grad.abs().sum()) > 0.0
    with torch.no_grad():
        stepped = logits - 0.01 * logits.grad
    stepped_log_probabilities = torch.log_softmax(stepped, dim=-1)
    stepped_probabilities = stepped_log_probabilities.exp()
    stepped_entropy = -(stepped_log_probabilities * stepped_probabilities).sum(-1)
    assert float(stepped_entropy.mean()) > entropy_mean


def test_loss_term_grad_norms_attribution_and_no_grad_pollution() -> None:
    """逐项归因:范数与手工对照一致、未用组不出现、param.grad 不被写入。"""
    shared = nn.Linear(4, 4)
    head_a = nn.Linear(4, 2)
    head_b = nn.Linear(4, 2)
    inputs = torch.randn(3, 4)
    loss_a = head_a(shared(inputs)).square().mean()
    loss_b = 0.5 * head_b(shared(inputs)).square().mean()
    groups = {
        "actor": list(head_a.parameters()),
        "belief": list(head_b.parameters()),
        "critic": [],
        "shared": list(shared.parameters()),
    }
    metrics = loss_term_grad_norms({"a": loss_a, "b": loss_b}, groups)
    assert set(metrics) == {
        "grad_term/a/actor", "grad_term/a/shared",
        "grad_term/b/belief", "grad_term/b/shared",
    }
    reference_a = torch.autograd.grad(
        loss_a, list(head_a.parameters()), retain_graph=True,
    )
    expected_a = float(torch.sqrt(sum(g.square().sum() for g in reference_a)))
    assert abs(metrics["grad_term/a/actor"] - expected_a) < 1e-6
    reference_b = torch.autograd.grad(
        loss_b, list(head_b.parameters()), retain_graph=True,
    )
    expected_b = float(torch.sqrt(sum(g.square().sum() for g in reference_b)))
    assert abs(metrics["grad_term/b/belief"] - expected_b) < 1e-6
    # 与参数无图连接的项被跳过,而不是报错。
    constant = torch.zeros((), requires_grad=False)
    assert "grad_term/constant/actor" not in loss_term_grad_norms(
        {"constant": constant}, groups,
    )
    # autograd.grad 不写入 param.grad。
    for module in (shared, head_a, head_b):
        for parameter in module.parameters():
            assert parameter.grad is None


def test_update_reports_entropy_floor_deficit_when_active() -> None:
    """地板远高于合成熵:update 上报正亏额,loss 恒等式含屏障项。"""
    torch.manual_seed(2026)
    kwargs = dict(_learner_kwargs())
    kwargs.update({"entropy_floor": 2.0, "entropy_floor_coef": 0.02})
    learner = PPOLearner("v19", "cpu", **kwargs)
    metrics = learner.update(_buffer(2026), shuffle_seed=7)
    assert "entropy_floor_deficit" in metrics
    assert metrics["entropy_floor_deficit"] > 0.0
    # loss 恒等式(逐样本均值):policy + value_coef·value - entropy_coef·
    # entropy_norm + belief_sft_coef·belief + floor_coef·deficit。
    expected = (
        metrics["policy_loss"]
        + 0.5 * metrics["value_loss"]
        - metrics["system/entropy_coef"] * metrics["entropy_normalized"]
        + metrics["belief/total_loss"]
        + 0.02 * metrics["entropy_floor_deficit"]
    )
    assert abs(metrics["loss"] - expected) < 1e-5


def test_update_without_floor_reports_no_deficit_key() -> None:
    """未配置地板时不上报亏额指标(默认行为与历史一致)。"""
    torch.manual_seed(2027)
    learner = PPOLearner("v19", "cpu", **_learner_kwargs())
    metrics = learner.update(_buffer(2027), shuffle_seed=7)
    assert "entropy_floor_deficit" not in metrics


def test_update_belief_sft_coef_scales_loss_contribution() -> None:
    """belief_sft_coef=0.5 时信念项以一半权重进入 loss 恒等式。"""
    torch.manual_seed(2028)
    kwargs = dict(_learner_kwargs())
    kwargs.update({"belief_sft_coef": 0.5})
    learner = PPOLearner("v19", "cpu", **kwargs)
    metrics = learner.update(_buffer(2028), shuffle_seed=7)
    expected = (
        metrics["policy_loss"]
        + 0.5 * metrics["value_loss"]
        - metrics["system/entropy_coef"] * metrics["entropy_normalized"]
        + 0.5 * metrics["belief/total_loss"]
    )
    assert abs(metrics["loss"] - expected) < 1e-5


def test_update_emits_grad_term_diagnostics() -> None:
    """逐项归因诊断:update 指标含各项范数;无参考策略时 sft_kl 项跳过。"""
    torch.manual_seed(2029)
    kwargs = dict(_learner_kwargs())
    kwargs.update({"grad_term_diagnostics_interval_updates": 1})
    learner = PPOLearner("v19", "cpu", **kwargs)
    metrics = learner.update(_buffer(2029), shuffle_seed=3)
    for name in (
        "grad_term/policy/actor", "grad_term/policy/shared",
        "grad_term/value/critic", "grad_term/value/shared",
        "grad_term/entropy/actor", "grad_term/entropy/shared",
        "grad_term/belief/belief", "grad_term/belief/shared",
    ):
        assert name in metrics, name
        assert np.isfinite(metrics[name]), name
        assert metrics[name] >= 0.0, name
    # 测试配置无 SFT 参考策略(sft_kl_coef=0):该项与参数无图连接被跳过。
    assert "grad_term/sft_kl/actor" not in metrics
    assert "grad_term/diagnostics_failed" not in metrics
