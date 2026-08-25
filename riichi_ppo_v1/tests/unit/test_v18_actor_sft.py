"""V18 Actor-only BC 的梯度与优化器隔离。"""

import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.sft.actor_bc import (
    actor_parameters,
    behavior_cloning_loss,
    freeze_critic,
    is_actor_parameter,
)
from riichi_ppo_v1.tests.v18_fixtures import actor_inputs


def test_actor_only_backward_excludes_all_critic_parameters() -> None:
    model = KyokuTransformerActorCritic()
    freeze_critic(model)
    optimizer = torch.optim.AdamW(actor_parameters(model), lr=1e-3)
    inputs = actor_inputs(batch=2)
    actor_before = next(parameter for name, parameter in model.named_parameters() if is_actor_parameter(name)).detach().clone()
    critic_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not is_actor_parameter(name)
    }
    loss = behavior_cloning_loss(model, inputs, torch.tensor([1, 7]))
    loss.backward()
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    for name, parameter in model.named_parameters():
        if is_actor_parameter(name):
            assert id(parameter) in optimized
        else:
            assert id(parameter) not in optimized
            assert parameter.grad is None
    optimizer.step()
    actor_after = next(parameter for name, parameter in model.named_parameters() if is_actor_parameter(name))
    assert not torch.equal(actor_before, actor_after)
    for name, parameter in model.named_parameters():
        if name in critic_before:
            assert torch.equal(critic_before[name], parameter)
