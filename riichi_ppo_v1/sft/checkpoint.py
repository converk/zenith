"""Explicit v13 SFT exact-resume and weights-only checkpoint loaders."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from ..model import KyokuTransformerActorCritic, ModelConfig
from .contract import (
    DATA_CURSOR_VERSION,
    DATA_PLAN_VERSION,
    SFT_CONTRACT_VERSION,
    TRAINING_MODES,
    validate_v13_manifest,
)
from ..model.feature_schema import ENCODED_FORMAT


def _require_mapping(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"SFT checkpoint is missing {field}")
    return value


def _validate_v13_model_config(value: Mapping[str, Any]) -> ModelConfig:
    if value.get("policy_head_type") != "isolated_action_query":
        raise RuntimeError("v13 checkpoint must explicitly use isolated_action_query")
    try:
        return ModelConfig(**dict(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("SFT checkpoint has an invalid model_config") from exc


def load_v13_weights_only(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> KyokuTransformerActorCritic:
    """Load only v13 tensors; never restore optimizer, cursor, or RNG."""
    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"invalid checkpoint payload: {checkpoint}")
    contract = payload.get("sft_contract_version")
    if contract is None:
        # Explicit read-only compatibility for the immutable current v13
        # checkpoint format.  No missing model/head field is synthesized.
        if payload.get("token_schema_version") != 13:
            raise RuntimeError("weights-only v13 load requires a v13 checkpoint")
        validate_v13_manifest({
            "format": ENCODED_FORMAT,
            "token_schema_version": payload.get("token_schema_version"),
            "feature_schema_sha256": payload.get("feature_schema_sha256"),
            "rust_analysis_version": payload.get("rust_analysis_version"),
            "decision_analysis_version": payload.get("decision_analysis_version"),
        })
    elif contract != SFT_CONTRACT_VERSION:
        raise RuntimeError(f"unsupported SFT contract: {contract!r}")
    config = _validate_v13_model_config(_require_mapping(payload, "model_config"))
    state = _require_mapping(payload, "model")
    model = KyokuTransformerActorCritic(config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("v13 checkpoint tensor shapes do not match model_config") from exc
    model.to(device)
    model.eval()
    return model


def load_exact_resume(
    path: str | Path,
    *,
    model_config: ModelConfig,
    training_mode: str,
    dataset_manifest_hash: str,
    world_size: int,
) -> dict[str, Any]:
    """Load a complete current-format training state with no fallbacks."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("invalid SFT checkpoint payload")
    required = {
        "sft_contract_version", "data_plan_version", "model_config",
        "training_mode", "dataset_manifest_hash", "model", "optimizer",
        "scheduler", "data_cursor", "rank_rng_states", "epoch", "global_step",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError("exact resume checkpoint is missing: " + ", ".join(missing))
    if payload["sft_contract_version"] != SFT_CONTRACT_VERSION:
        raise RuntimeError("exact resume checkpoint has an incompatible SFT contract")
    if payload["data_plan_version"] != DATA_PLAN_VERSION:
        raise RuntimeError("exact resume checkpoint has an incompatible data plan")
    if payload["model_config"] != asdict(model_config):
        raise RuntimeError("exact resume checkpoint has an incompatible model_config")
    if training_mode not in TRAINING_MODES or payload["training_mode"] != training_mode:
        raise RuntimeError("exact resume checkpoint has an incompatible training_mode")
    if payload["dataset_manifest_hash"] != dataset_manifest_hash:
        raise RuntimeError("exact resume checkpoint belongs to a different dataset manifest")
    cursor = _require_mapping(payload, "data_cursor")
    if cursor.get("version") != DATA_CURSOR_VERSION:
        raise RuntimeError("exact resume checkpoint has an incompatible data cursor")
    if cursor.get("world_size") != int(world_size):
        raise RuntimeError("exact resume checkpoint uses a different world size")
    progress = cursor.get("rank_batches_consumed")
    if not isinstance(progress, list) or len(progress) != int(world_size):
        raise RuntimeError("exact resume checkpoint has malformed rank cursor state")
    if any(not isinstance(value, int) or value < 0 for value in progress):
        raise RuntimeError("exact resume checkpoint has invalid rank cursor progress")
    rank_rng_states = payload["rank_rng_states"]
    if not isinstance(rank_rng_states, list) or len(rank_rng_states) != int(world_size):
        raise RuntimeError("exact resume checkpoint has malformed per-rank RNG state")
    for state in rank_rng_states:
        if not isinstance(state, Mapping) or set(state) != {
            "torch", "cuda", "numpy", "python",
        }:
            raise RuntimeError("exact resume checkpoint has incomplete per-rank RNG state")
    _require_mapping(payload, "model")
    _require_mapping(payload, "optimizer")
    _require_mapping(payload, "scheduler")
    return payload


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    config: dict[str, Any],
    manifest_hash: str,
    mode: str,
    epoch: int,
    global_step: int,
    rank_batches_consumed: list[int],
    best_validation_loss: float,
    best_heuristic_point_delta: float,
    metrics: dict[str, float],
    rank_rng_states: list[dict[str, Any]],
) -> dict[str, Any]:
    module = getattr(model, "module", model)
    return {
        "sft_contract_version": SFT_CONTRACT_VERSION,
        "data_plan_version": DATA_PLAN_VERSION,
        "model_config": asdict(module.config),
        "training_mode": mode,
        "dataset_manifest_hash": manifest_hash,
        "model": {name: value.detach().cpu() for name, value in module.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "data_cursor": {
            "version": DATA_CURSOR_VERSION,
            "epoch": int(epoch),
            "rank_batches_consumed": [int(value) for value in rank_batches_consumed],
            "world_size": len(rank_batches_consumed),
        },
        "sft_config": dict(config),
        "training_stage": "sft",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_loss": float(best_validation_loss),
        "best_heuristic_point_delta": float(best_heuristic_point_delta),
        "metrics": dict(metrics),
        "rank_rng_states": rank_rng_states,
    }
