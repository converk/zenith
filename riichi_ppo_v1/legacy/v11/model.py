"""Strict weights-only loader for historical v11 checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from ...model import KyokuTransformerActorCritic, ModelConfig
from .contract import V11_POLICY_HEAD, V11_TOKEN_SCHEMA


def load_v11_weights_only(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> KyokuTransformerActorCritic:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("token_schema_version") != V11_TOKEN_SCHEMA:
        raise RuntimeError("v11 evaluation loader requires an explicit schema-11 checkpoint")
    stored_config = payload.get("model_config")
    state = payload.get("model")
    if not isinstance(stored_config, Mapping) or not isinstance(state, Mapping):
        raise RuntimeError("v11 checkpoint is missing model_config or model tensors")
    explicit_head = stored_config.get("policy_head_type")
    if explicit_head is not None and explicit_head != V11_POLICY_HEAD:
        raise RuntimeError("v11 checkpoint declares an incompatible policy head")
    # This is a fixed fact of V11_CONTRACT_ID, not a fallback inferred from
    # another checkpoint field.  Old v11 checkpoints predate serialization of
    # the otherwise invariant head name.
    config = dict(stored_config)
    config["policy_head_type"] = V11_POLICY_HEAD
    try:
        model = KyokuTransformerActorCritic(ModelConfig(**config))
        model.load_state_dict(state, strict=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("v11 checkpoint tensors do not match the frozen model contract") from exc
    model.to(device)
    model.eval()
    return model
