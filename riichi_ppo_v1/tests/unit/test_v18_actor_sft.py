"""V18 Actor-only BC 的梯度与优化器隔离。"""

from __future__ import annotations

import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.sft.actor_bc import actor_parameters, behavior_cloning_loss, freeze_critic, is_actor_parameter
from riichi_ppo_v1.tests.v18_fixtures import actor_inputs


def test_actor_parameter_scope() -> None:
    model = KyokuTransformerActorCritic()
    freeze_critic(model)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == is_actor_parameter(name)
    optimized = list(actor_parameters(model))
    assert len(optimized) > 0
    assert all(parameter.requires_grad for parameter in optimized)


def test_behavior_cloning_gradient_isolation() -> None:
    model = KyokuTransformerActorCritic()
    freeze_critic(model)
    inputs = actor_inputs(batch=2, action_ids=(1, 7, 12))
    targets = torch.tensor([1, 7])
    loss = behavior_cloning_loss(model, inputs, targets)
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
    loss = behavior_cloning_loss(model, inputs, torch.tensor([1, 7]))
    loss.backward()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    adam.step()
    changed = [
        name for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    ]
    assert changed
    assert all(is_actor_parameter(name) for name in changed)
