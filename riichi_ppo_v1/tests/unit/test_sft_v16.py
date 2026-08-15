"""V16 SFT 编码/manifest/加载器/训练 collate 的契约测试。"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
from riichi_ppo_v1.model.encoding_protocol import (
    DEFENSE_SLOT_ORDER,
    ENCODED_FORMAT,
    ENCODING_PROTOCOL_VERSION,
    OFFENSE_SLOT_ORDER,
    QUERY_ROW_ANSWER_START,
    SLOT_CARDINALITIES,
)
from riichi_ppo_v1.sft.contract import (
    V16_ACTOR_INPUT_CONTRACT_SHA256,
    validate_v16_manifest,
)
from riichi_ppo_v1.sft.data import encode_kyoku_v16
from riichi_ppo_v1.sft.precompute import (
    _write_chunk_v16,
    iter_precomputed_v16_samples,
)
from riichi_ppo_v1.sft.prepare import _json_lines, split_game_events
from riichi_ppo_v1.sft.train_v16 import collate_v16, validate_config

FIXTURE = Path(__file__).parents[3] / "RiichiEnv/tests/data/126_204_0_mjai.jsonl"


def _first_kyoku() -> str:
    events = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line]
    return _json_lines(split_game_events(events)[0])


def _v16_manifest(**overrides) -> dict:
    manifest = {
        "format": ENCODED_FORMAT,
        "encoding_protocol_version": ENCODING_PROTOCOL_VERSION,
        "encoding_contract_sha256": V16_ACTOR_INPUT_CONTRACT_SHA256,
        "source_manifest_sha256": "source",
        "subset_denominator": 5,
        "subset_remainders": [0, 1],
        "counts": {
            "train_kyokus": 1, "validation_kyokus": 1,
            "train_decisions": 1, "validation_decisions": 1,
        },
    }
    manifest.update(overrides)
    return manifest


def test_validate_v16_manifest_accepts_and_rejects() -> None:
    validate_v16_manifest(_v16_manifest())
    with pytest.raises(RuntimeError, match="v16 encoded SFT format"):
        validate_v16_manifest(_v16_manifest(format="riichi-sft-encoded-v3"))
    with pytest.raises(RuntimeError, match="encoding_protocol_version"):
        validate_v16_manifest(_v16_manifest(encoding_protocol_version=13))
    with pytest.raises(RuntimeError, match="contract hash"):
        validate_v16_manifest(_v16_manifest(encoding_contract_sha256="unknown"))
    with pytest.raises(RuntimeError, match="positive"):
        validate_v16_manifest(_v16_manifest(counts={"train_decisions": 0}))


def test_encode_kyoku_v16_emits_one_pair_per_unique_action() -> None:
    samples = encode_kyoku_v16(
        _first_kyoku(),
        year=2024, game_id="fixture", kyoku_index=0,
    )
    assert {sample.seat for sample in samples} == {0, 1, 2, 3}
    for sample in samples:
        ids = np.flatnonzero(sample.legal_mask).tolist()
        assert list(sample.action_ids) == ids
        assert sample.query_rows.shape[0] == 2 * len(ids)
        assert bool(sample.legal_mask[sample.action])
        offense = sample.query_rows[0::2, QUERY_ROW_ANSWER_START:]
        defense = sample.query_rows[1::2, QUERY_ROW_ANSWER_START:]
        for index, slot in enumerate(OFFENSE_SLOT_ORDER):
            assert bool(np.all(offense[:, index] < SLOT_CARDINALITIES[slot]))
        for index, slot in enumerate(DEFENSE_SLOT_ORDER):
            assert bool(np.all(defense[:, index] < SLOT_CARDINALITIES[slot]))


def test_v16_chunk_round_trip_and_ddp_partition() -> None:
    samples = encode_kyoku_v16(
        _first_kyoku(),
        year=2024, game_id="fixture", kyoku_index=0,
    )[:4]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        train = root / "train"
        train.mkdir()
        for index in range(4):
            _write_chunk_v16(train / f"train-{index:05d}.npz", samples[index:index + 1])
        (root / "manifest.json").write_text(
            json.dumps(_v16_manifest(counts={
                "train_kyokus": 1, "validation_kyokus": 1,
                "train_decisions": 4, "validation_decisions": 4,
            }))
        )
        rank0 = list(iter_precomputed_v16_samples(
            root, "train", seed=7, shuffle=False, rank=0, world_size=2,
        ))
        rank1 = list(iter_precomputed_v16_samples(
            root, "train", seed=7, shuffle=False, rank=1, world_size=2,
        ))
    assert len(rank0) == 2 and len(rank1) == 2
    combined = rank0 + rank1
    for loaded, source in zip(combined, samples, strict=True):
        assert loaded.action == source.action
        assert loaded.history_length == source.history_length
        assert loaded.snapshot_length == source.snapshot_length
        assert loaded.query_pair_count == source.query_pair_count
        np.testing.assert_array_equal(loaded.query_rows, source.query_rows)


def test_v16_collate_and_actor_only_backward() -> None:
    samples = encode_kyoku_v16(
        _first_kyoku(),
        year=2024, game_id="fixture", kyoku_index=0,
    )[:2]
    batch = collate_v16(samples, torch.device("cpu"))
    config = ModelConfig(
        layers=2, shared_layers=1, critic_layers=1, d_model=32,
        query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64,
        context_tokens=4096, policy_head_type="symmetric_action_query",
    )
    model = KyokuTransformerActorCritic(config)
    critic_roots = {"critic_embedding", "critic_backbone", "value_head", "value_query"}
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.split(".", 1)[0] in critic_roots
    }
    optimizer = torch.optim.AdamW([
        parameter for name, parameter in model.named_parameters()
        if name.split(".", 1)[0] not in critic_roots
    ], lr=1e-3)
    output = model.forward_v16(
        batch["history_factors"], batch["history_numeric"], batch["history_lengths"],
        batch["snapshot_kinds"], batch["snapshot_cat"], batch["snapshot_num"],
        batch["snapshot_lengths"],
        batch["query_rows"], batch["action_ids"], batch["pair_counts"],
        batch["legal_mask"], policy_only=True,
    )
    F.cross_entropy(output["policy_logits"], batch["actions"]).backward()
    optimizer.step()
    assert any(parameter.grad is not None for parameter in model.action_fusion.parameters())
    for name, parameter in model.named_parameters():
        if name in before:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_v16_train_config_rejects_duplicated_cadence_keys(tmp_path: Path) -> None:
    base = {
        "policy_head_type": "symmetric_action_query",
        "model_size": "v16",
        "learner_gpus": 1,
        "device": "cpu",
        "batch_size": 16,
        "epochs": 1,
        "context_tokens": 4096,
        "dataset": str(tmp_path),
    }
    with pytest.raises(ValueError, match="cadence"):
        validate_config({**base, "validation_interval_steps": 3000})
