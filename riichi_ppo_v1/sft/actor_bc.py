"""V18 Actor-only 行为克隆的参数隔离边界。

checkpoint 持久化统一走 ``sft/checkpoint.py`` 的完整训练状态 payload;
bot 等推理端直接加载该 payload,本模块只负责 Actor/Critic 参数划分。
"""

from __future__ import annotations

from typing import Iterable

from torch import nn

_CRITIC_ROOTS = frozenset({
    "critic_backbone", "value_head", "value_query",
})


def is_actor_parameter(name: str) -> bool:
    return name.split(".", 1)[0] not in _CRITIC_ROOTS


def freeze_critic(model: nn.Module) -> None:
    """冻结所有 Critic 私有参数并清除可能残留的梯度。"""
    for name, parameter in model.named_parameters():
        actor = is_actor_parameter(name)
        parameter.requires_grad_(actor)
        if not actor:
            parameter.grad = None


def actor_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """只迭代可训练 Actor 参数,供优化器直接消费。"""
    for name, parameter in model.named_parameters():
        if is_actor_parameter(name):
            yield parameter
