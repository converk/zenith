"""V19 Actor-only BC + 信念网络的梯度与优化器隔离。"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.sft.actor_bc import actor_parameters, freeze_critic, is_actor_parameter
from riichi_ppo_v1.tests.v19_fixtures import actor_inputs


def _actor_loss(model: torch.nn.Module, inputs: dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    """BC 损失:与 sft/trainer.py 一致,直接对 policy_logits 求交叉熵。"""
    output = model(
        actor_factors=inputs["actor_factors"],
        actor_numeric=inputs["actor_numeric"],
        actor_lengths=inputs["actor_lengths"],
        query_action_ids=inputs["action_ids"],
        query_pair_counts=inputs["query_pair_counts"],
        legal_mask=inputs["legal_mask"],
        policy_only=True,
    )
    return F.cross_entropy(output["policy_logits"].float(), targets.long())


def test_actor_parameter_scope() -> None:
    model = KyokuTransformerActorCritic()
    freeze_critic(model)
    assert any(name.startswith("belief_network.") for name, _parameter in model.named_parameters())
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == is_actor_parameter(name)
    optimized = list(actor_parameters(model))
    assert len(optimized) > 0
    assert all(parameter.requires_grad for parameter in optimized)


def test_actor_gradient_isolation() -> None:
    model = KyokuTransformerActorCritic()
    freeze_critic(model)
    inputs = actor_inputs(batch=2, action_ids=(1, 7, 12))
    targets = torch.tensor([1, 7])
    loss = _actor_loss(model, inputs, targets)
    loss.backward()
    actor_grads = 0
    for name, parameter in model.named_parameters():
        if is_actor_parameter(name):
            if parameter.grad is not None:
                actor_grads += 1
        else:
            assert parameter.grad is None, name
    # 参与 Actor 前向的参数应获得梯度；Critic 专用表（13/14）在 actor-only 下无梯度。
    assert actor_grads >= 20


def test_optimizer_step_updates_actor_only() -> None:
    model = KyokuTransformerActorCritic()
    freeze_critic(model)
    inputs = actor_inputs(batch=2, action_ids=(1, 7))
    adam = torch.optim.AdamW(actor_parameters(model), lr=1e-3)
    loss = _actor_loss(model, inputs, torch.tensor([1, 7]))
    loss.backward()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    adam.step()
    changed = [
        name for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    ]
    assert changed
    assert all(is_actor_parameter(name) for name in changed)
