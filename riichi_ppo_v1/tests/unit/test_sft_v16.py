"""V16 SFT 编码/manifest/加载器/训练 collate 的契约测试。"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
import tarfile
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
from riichi_ppo_v1.sft.data import _member_metadata, encode_kyoku_v16
from riichi_ppo_v1.sft.precompute import (
    _selection_bucket,
    _write_chunk_v16,
    iter_precomputed_v16_samples,
    precompute_v16,
)
from riichi_ppo_v1.sft.prepare import _json_lines, split_game_events
from riichi_ppo_v1.sft.train_v16 import collate_v16, validate_config

FIXTURE = Path(__file__).parents[3] / "RiichiEnv/tests/data/126_204_0_mjai.jsonl"


def _first_kyoku() -> str:
    events = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line]
    return _json_lines(split_game_events(events)[0])


def _make_v16_source(root: Path) -> Path:
    """构造覆盖 0/1/2 三个 remainder 的小型 tar 源(直接复用首个 kyoku)。"""
    source = root / "source"
    source_manifest = {"format": "riichi-sft-kyoku-v1"}
    for split in ("train", "validation"):
        directory = source / split
        directory.mkdir(parents=True, exist_ok=True)
        names: list[str] = []
        remainders: set[int] = set()
        index = 0
        while not {0, 1, 2}.issubset(remainders):
            name = f"2024-{split}-g{index:03d}-00.mjson"
            game_id = _member_metadata(name)[1]
            remainders.add(_selection_bucket(game_id, "subset", 5))
            names.append(name)
            index += 1
        payload = gzip.compress(_first_kyoku().encode("utf-8"), mtime=0)
        with tarfile.open(directory / f"{split}-00000.tar", "w") as archive:
            for name in names:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mtime = 0
                archive.addfile(info, io.BytesIO(payload))
    (source / "manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False), encoding="utf-8",
    )
    return source


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


def test_precompute_v16_reuses_base_encoded(tmp_path: Path) -> None:
    """复用 40% 缓存追加 20%:只编码新 remainder,chunk/manifest 合并正确。"""
    source = _make_v16_source(tmp_path / "src")
    base = tmp_path / "base"
    final = tmp_path / "final"
    precompute_v16(
        source, base, denominator=5, remainders=(0, 1),
        workers=1, kyokus_per_shard=4,
    )
    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    base_counts = dict(base_manifest["counts"])
    base_chunks = {
        split: sorted((base / split).glob(f"{split}-*.npz"))
        for split in ("train", "validation")
    }

    precompute_v16(
        source, final, denominator=5, remainders=(0, 1, 2),
        workers=1, kyokus_per_shard=4, base_encoded=base,
    )
    final_manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    validate_v16_manifest(final_manifest)
    assert final_manifest["subset_remainders"] == [0, 1, 2]
    assert Path(str(final_manifest["reused_encoded_cache"])) == base
    assert final_manifest["reused_counts"] == base_counts
    for split in ("train", "validation"):
        chunks = sorted((final / split).glob(f"{split}-*.npz"))
        new_chunks = [path for path in chunks if "-r2-" in path.name]
        assert len(chunks) == len(base_chunks[split]) + len(new_chunks)
        assert new_chunks, f"{split} 应包含追加的 r2 chunk"
        for path in base_chunks[split]:
            assert (final / split / path.name).is_file()
        assert final_manifest["counts"][f"{split}_kyokus"] > base_counts[f"{split}_kyokus"]
        assert final_manifest["counts"][f"{split}_decisions"] > base_counts[f"{split}_decisions"]
    total = sum(
        len(list(iter_precomputed_v16_samples(
            final, split, seed=1, shuffle=False,
        )))
        for split in ("train", "validation")
    )
    expected = (
        int(final_manifest["counts"]["train_decisions"])
        + int(final_manifest["counts"]["validation_decisions"])
    )
    assert total == expected
