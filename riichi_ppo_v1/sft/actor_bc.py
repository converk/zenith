"""V18 Actor-only 行为克隆的参数、损失与持久化边界。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..model import KyokuTransformerActorCritic, ModelConfig
from .contract import SFT_CONTRACT_VERSION

_CRITIC_ROOTS = frozenset({
    "critic_embedding", "critic_backbone", "value_head", "value_query",
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


def actor_logits(model: nn.Module, batch: Mapping[str, Tensor]) -> Tensor:
    output = model(
        history_factors=batch["history_factors"],
        history_numeric=batch["history_numeric"],
        history_lengths=batch["history_lengths"],
        snapshot_factors=batch["snapshot_factors"],
        snapshot_numeric=batch["snapshot_numeric"],
        snapshot_lengths=batch["snapshot_lengths"],
        query_rows=batch["query_rows"],
        query_action_ids=batch["query_action_ids"],
        query_pair_counts=batch["query_pair_counts"],
        legal_mask=batch["legal_mask"],
        policy_only=True,
    )
    return output["policy_logits"]


def behavior_cloning_loss(
    model: nn.Module, batch: Mapping[str, Tensor], targets: Tensor,
) -> Tensor:
    return F.cross_entropy(actor_logits(model, batch).float(), targets.long())


def actor_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if is_actor_parameter(name)
    }


def save_actor(path: str | Path, model: KyokuTransformerActorCritic) -> None:
    payload = {
        "sft_contract_version": SFT_CONTRACT_VERSION,
        "artifact_type": "v18_actor_only",
        "model_config": asdict(model.config),
        "actor": actor_state_dict(model),
    }
    torch.save(payload, Path(path))


def load_actor(path: str | Path, *, map_location: str | torch.device = "cpu") -> KyokuTransformerActorCritic:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("Actor artifact must be a mapping")
    if payload.get("sft_contract_version") != SFT_CONTRACT_VERSION:
        raise RuntimeError("Actor artifact is not a pure V18 SFT contract")
    if payload.get("artifact_type") != "v18_actor_only":
        raise RuntimeError("Actor artifact has an incompatible artifact_type")
    raw_config = payload.get("model_config")
    state = payload.get("actor")
    if not isinstance(raw_config, Mapping) or not isinstance(state, Mapping):
        raise RuntimeError("Actor artifact is incomplete")
    config = ModelConfig.from_mapping(dict(raw_config))
    model = KyokuTransformerActorCritic(config)
    expected = {name for name in model.state_dict() if is_actor_parameter(name)}
    if set(state) != expected:
        raise RuntimeError("Actor artifact keys do not exactly match V18 Actor state")
    incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != {
        name for name in model.state_dict() if not is_actor_parameter(name)
    }:
        raise RuntimeError("Actor artifact failed exact key validation")
    freeze_critic(model)
    return model
