from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
from riichi_ppo_v1.sft.checkpoint import checkpoint_payload, load_exact_resume
from riichi_ppo_v1.sft.contract import (
    DATA_PLAN_VERSION,
    SFT_CONTRACT_VERSION,
    validate_v13_manifest,
)


def _config() -> ModelConfig:
    return ModelConfig(
        layers=2, shared_layers=1, critic_layers=1, d_model=32,
        query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64,
        context_tokens=128, policy_head_type="isolated_action_query",
    )


def _rng_state() -> dict[str, object]:
    return {
        "torch": torch.get_rng_state(), "cuda": None,
        "numpy": np.random.get_state(), "python": __import__("random").getstate(),
    }


def test_manifest_contract_accepts_current_formal_cache_and_new_compact_format() -> None:
    validate_v13_manifest({
        "format": "riichi-sft-encoded-v3",
        "token_schema_version": 13,
        "feature_schema_sha256": "ad8dc752f116d6d6430930e16c6a17322b3da980549d3350a5ddc461ee123036",
        "rust_analysis_version": 4,
        "decision_analysis_version": 16,
    })
    validate_v13_manifest({
        "format": "riichi-sft-encoded-v3",
        "sft_contract_version": SFT_CONTRACT_VERSION,
    })
    with pytest.raises(RuntimeError, match="supported v13 SFT contract"):
        validate_v13_manifest({
            "format": "riichi-sft-encoded-v3", "token_schema_version": 11,
        })


def test_exact_resume_requires_current_complete_contract_and_world_size() -> None:
    model = KyokuTransformerActorCritic(_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    payload = checkpoint_payload(
        model, optimizer, scheduler, config={}, manifest_hash="manifest",
        mode="actor_only", epoch=2, global_step=7, rank_batches_consumed=[3, 4],
        best_validation_loss=1.0, best_heuristic_point_delta=2.0, metrics={},
        rank_rng_states=[_rng_state(), _rng_state()],
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "current.pt"
        torch.save(payload, path)
        loaded = load_exact_resume(
            path, model_config=model.config, training_mode="actor_only",
            dataset_manifest_hash="manifest", world_size=2,
        )
        assert loaded["data_plan_version"] == DATA_PLAN_VERSION
        with pytest.raises(RuntimeError, match="different world size"):
            load_exact_resume(
                path, model_config=model.config, training_mode="actor_only",
                dataset_manifest_hash="manifest", world_size=1,
            )


def test_rank_steps_only_and_missing_training_mode_are_rejected() -> None:
    old = {
        "token_schema_version": 13,
        "model_config": asdict(_config()),
        "model": {}, "optimizer": {}, "scheduler": {}, "rank_steps": [4],
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "old.pt"
        torch.save(old, path)
        with pytest.raises(RuntimeError, match="exact resume checkpoint is missing"):
            load_exact_resume(
                path, model_config=_config(), training_mode="actor_only",
                dataset_manifest_hash="manifest", world_size=1,
            )


def test_v13_training_modules_do_not_import_legacy_v11() -> None:
    package = Path(__file__).parents[2]
    for relative in ("sft/data.py", "sft/precompute.py", "sft/train.py", "sft/checkpoint.py"):
        assert "legacy.v11" not in (package / relative).read_text(encoding="utf-8")
