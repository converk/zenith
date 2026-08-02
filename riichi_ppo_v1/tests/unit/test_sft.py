import gzip
import io
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import tarfile
import zipfile

import numpy as np
import torch
import pytest
from torch.nn import functional as F

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
from riichi_ppo_v1.model.critic_features import FIELD_OPPONENT_HAND, SEGMENT_PUBLIC_SUMMARY
from riichi_ppo_v1.model.feature_schema import (
    DECISION_ANALYSIS_VERSION,
    RUST_ANALYSIS_VERSION,
    feature_schema_sha256,
)
from riichi_ppo_v1.model.schema import TOKEN_SCHEMA_VERSION
from riichi_ppo_v1.sft.data import encode_kyoku, iter_split_samples
from riichi_ppo_v1.sft import precompute as sft_precompute
from riichi_ppo_v1.sft.audit import audit_kyoku, select_coverage_kyokus, validate_encoded_chunk
from riichi_ppo_v1.sft.precompute import (
    _empty_field_statistics, _require_complete_action_coverage,
    _selection_bucket, _write_chunk, encoded_identity_digests, selected_any,
)
from riichi_ppo_v1.sft.prepare import (
    _json_lines,
    _zip_directory_is_readable,
    prepare_archives,
    split_game_events,
    stable_split,
)
from riichi_ppo_v1.sft.train import (
    collate_samples, length_bucketed_batches, load_config,
)
from riichi_ppo_v1.training.learner import PPOLearner


FIXTURE = Path(__file__).parents[3] / "RiichiEnv/tests/data/126_204_0_mjai.jsonl"


def _first_kyoku() -> str:
    events = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line]
    return _json_lines(split_game_events(events)[0])


def test_stable_split_is_game_level_and_deterministic() -> None:
    assert stable_split("game-a") == stable_split("game-a")
    assert stable_split("game-a", validation_percent=100) == "validation"


def test_zip_validation_rejects_a_truncated_archive() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "data.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("row.mjson", b"{}\n")
        assert _zip_directory_is_readable(path)
        path.write_bytes(path.read_bytes()[:-8])
        assert not _zip_directory_is_readable(path)


def test_replay_encoder_emits_all_seats_public_suffix_and_private_hands() -> None:
    samples = encode_kyoku(_first_kyoku(), year=2024, game_id="fixture", kyoku_index=0)
    assert {sample.seat for sample in samples} == {0, 1, 2, 3}
    assert all(sample.legal_mask[sample.action] for sample in samples)
    assert all(np.all(sample.critic_factors[:, 0] == 4) for sample in samples)
    assert all(np.all(sample.critic_factors[:, 2] == FIELD_OPPONENT_HAND) for sample in samples)
    with_public = next(sample for sample in reversed(samples) if SEGMENT_PUBLIC_SUMMARY in sample.token_factors[:, 0])
    first_public = int(np.flatnonzero(with_public.token_factors[:, 0] == SEGMENT_PUBLIC_SUMMARY)[0])
    first_query = int(np.flatnonzero(with_public.token_factors[:, 0] == 7)[0])
    assert first_public < first_query
    assert np.all(with_public.token_factors[first_query:, 0] == 7)
    for seat in range(4):
        seat_rows = [sample for sample in samples if sample.seat == seat]
        assert seat_rows[-1].value_target == [-4.0, -3.0, -2.0, 9.0][seat]


def test_sft_audit_selects_deterministic_event_coverage() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        split = root / "train"
        split.mkdir()
        with tarfile.open(split / "train-00000.tar", "w") as archive:
            for index, kind in enumerate(("dahai", "chi", "pon", "dora")):
                payload = json.dumps({"type": kind}).encode() + b"\n"
                member = tarfile.TarInfo(f"2024-a-{index}.mjson")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        records, universe, scanned = select_coverage_kyokus(root, denominator=1, remainder=0, sample_size=4)
        assert scanned == 4
        assert universe == {"dahai", "chi", "pon", "dora"}
        assert [identity.rsplit("/", 1)[-1] for identity, _content in records] == [
            "train-00000.tar:2024-a-0.mjson", "train-00000.tar:2024-a-1.mjson",
            "train-00000.tar:2024-a-2.mjson", "train-00000.tar:2024-a-3.mjson",
        ]


def test_sft_audit_checks_real_replay_event_history_and_public_state() -> None:
    result = audit_kyoku(_first_kyoku(), identity="fixture")
    assert result["decisions"] > 0
    assert set(result["events"]) == {"dahai", "chi", "pon", "daiminkan", "ankan", "kakan", "reach", "reach_accepted", "dora"}


def test_encoded_sft_chunk_parses_model_inputs_masks_and_expert_actions() -> None:
    samples = encode_kyoku(_first_kyoku(), include_critic=False)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "one-kyoku.npz"
        _write_chunk(path, samples)
        structure = validate_encoded_chunk(path)
    assert structure["rows"] == len(samples)
    assert structure["history_tokens"] > 0
    assert structure["state_tokens"] > 0
    assert structure["candidate_tokens"] == 2 * sum(int(sample.legal_mask.sum()) for sample in samples)
    assert structure["public_tokens"] > 0


def test_early_abortive_draw_is_preserved_as_expert_action_240() -> None:
    content = "\n".join((
        '{"type":"start_game","names":["a","b","c","d"],"kyoku_first":0,"aka_flag":true}',
        '{"type":"start_kyoku","bakaze":"E","dora_marker":"7s","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","9m","1p","9p","1s","9s","E","S","W","N","P","F","C"],["2m","3m","4m","2p","3p","4p","2s","3s","4s","5s","6s","7s","8s"],["2m","3m","4m","2p","3p","4p","2s","3s","4s","5s","6s","7s","8s"],["2m","3m","4m","2p","3p","4p","2s","3s","4s","5s","6s","7s","8s"]]}',
        '{"type":"tsumo","actor":0,"pai":"1m"}',
        '{"type":"ryukyoku","deltas":[0,0,0,0]}',
        '{"type":"end_kyoku"}',
        '{"type":"end_game"}',
    )) + "\n"
    samples = encode_kyoku(content, include_critic=False)
    assert len(samples) == 1
    assert samples[0].action == 240
    assert bool(samples[0].legal_mask[240])


def test_encoded_sft_shards_are_disjoint_across_ddp_ranks() -> None:
    samples = encode_kyoku(_first_kyoku(), include_critic=False)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        train = root / "train"
        train.mkdir()
        for index in range(4):
            _write_chunk(train / f"train-{index:05d}.npz", samples[index:index + 1])
        (root / "manifest.json").write_text(json.dumps({
            "format": "riichi-sft-encoded-v3",
            "token_schema_version": TOKEN_SCHEMA_VERSION,
            "feature_schema_sha256": feature_schema_sha256(),
            "rust_analysis_version": RUST_ANALYSIS_VERSION,
            "decision_analysis_version": DECISION_ANALYSIS_VERSION,
        }))
        rank0 = list(iter_split_samples(
            root, "train", seed=7, shuffle=False, rank=0, world_size=2,
            include_critic=False,
        ))
        rank1 = list(iter_split_samples(
            root, "train", seed=7, shuffle=False, rank=1, world_size=2,
            include_critic=False,
        ))
    assert [(sample.seat, sample.decision_index) for sample in rank0] == [
        (samples[0].seat, samples[0].decision_index),
        (samples[1].seat, samples[1].decision_index),
    ]
    assert [(sample.seat, sample.decision_index) for sample in rank1] == [
        (samples[2].seat, samples[2].decision_index),
        (samples[3].seat, samples[3].decision_index),
    ]


def test_encoded_sft_global_plan_balances_odd_rows_without_duplicates() -> None:
    samples = encode_kyoku(_first_kyoku(), include_critic=False)[:7]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        train = root / "train"
        train.mkdir()
        _write_chunk(train / "train-00000.npz", samples[:2])
        _write_chunk(train / "train-00001.npz", samples[2:])
        (root / "manifest.json").write_text(json.dumps({
            "format": "riichi-sft-encoded-v3",
            "token_schema_version": TOKEN_SCHEMA_VERSION,
            "feature_schema_sha256": feature_schema_sha256(),
            "rust_analysis_version": RUST_ANALYSIS_VERSION,
            "decision_analysis_version": DECISION_ANALYSIS_VERSION,
        }))
        ranks = [
            list(iter_split_samples(
                root, "train", seed=19, shuffle=True, rank=rank, world_size=2,
                include_critic=False,
            ))
            for rank in range(2)
        ]
    assert [len(rows) for rows in ranks] == [4, 3]
    identities = [
        (sample.seat, sample.decision_index)
        for rows in ranks for sample in rows
    ]
    expected = [(sample.seat, sample.decision_index) for sample in samples]
    assert len(identities) == len(set(identities))
    assert set(identities) == set(expected)
    assert [len(list(length_bucketed_batches(rows, 2, window_batches=1))) for rows in ranks] == [2, 2]


def test_encoded_sft_shard_columns_are_decompressed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = encode_kyoku(_first_kyoku(), include_critic=False)[:3]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        train = root / "train"
        train.mkdir()
        _write_chunk(train / "train-00000.npz", samples)
        (root / "manifest.json").write_text(json.dumps({
            "format": "riichi-sft-encoded-v3",
            "token_schema_version": TOKEN_SCHEMA_VERSION,
            "feature_schema_sha256": feature_schema_sha256(),
            "rust_analysis_version": RUST_ANALYSIS_VERSION,
            "decision_analysis_version": DECISION_ANALYSIS_VERSION,
        }))
        original_load = np.load
        accesses: Counter[str] = Counter()

        class CountingArchive:
            def __init__(self, *args, **kwargs) -> None:
                self.archive = original_load(*args, **kwargs)

            def __enter__(self):
                self.archive.__enter__()
                return self

            def __exit__(self, *args) -> None:
                self.archive.__exit__(*args)

            def __contains__(self, name: str) -> bool:
                return name in self.archive

            def __getitem__(self, name: str):
                accesses[name] += 1
                return self.archive[name]

        monkeypatch.setattr(sft_precompute.np, "load", CountingArchive)
        loaded = list(iter_split_samples(
            root, "train", seed=7, shuffle=False, include_critic=False,
        ))

    assert len(loaded) == len(samples)
    expected = {
        "offsets", "factors", "numeric", "legal", "actions", "value_targets",
        "teacher_masks", "years", "game_ids", "kyoku_indices", "seats",
        "decision_indices",
    }
    assert set(accesses) == expected
    assert all(accesses[name] == 1 for name in expected)


def test_unknown_encoded_contract_is_rejected_fail_closed() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "manifest.json").write_text(json.dumps({
            "format": "riichi-sft-encoded-v2", "token_schema_version": 12,
        }))
        with pytest.raises(RuntimeError, match="only the v13 encoded SFT format"):
            list(iter_split_samples(root, "train", shuffle=False, include_critic=False))


def test_subset_selection_is_by_game_id_not_individual_kyoku() -> None:
    names = ("2025-game-identity-00.jsonl", "2025-game-identity-07.jsonl")
    outcomes = [selected_any(name, 5, (0, 1)) for name in names]
    assert outcomes[0] == outcomes[1]


def test_canary_and_production_subset_use_independent_hash_namespaces() -> None:
    buckets = [
        (
            _selection_bucket(f"game-{index}", "subset", 1000),
            _selection_bucket(f"game-{index}", "canary", 1000),
        )
        for index in range(100)
    ]
    assert any(subset != canary for subset, canary in buckets)


def test_semantic_canary_rejects_missing_action_groups() -> None:
    statistics = _empty_field_statistics()
    with pytest.raises(RuntimeError, match="legal:pass"):
        _require_complete_action_coverage(statistics)
    statistics["legal_actions"][:] = 1
    statistics["expert_actions"][:] = 1
    _require_complete_action_coverage(statistics)


def test_v13_cache_requires_complete_analysis_metadata() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "manifest.json").write_text(json.dumps({
            "format": "riichi-sft-encoded-v3",
            "token_schema_version": TOKEN_SCHEMA_VERSION,
            "feature_schema_sha256": feature_schema_sha256(),
            "rust_analysis_version": RUST_ANALYSIS_VERSION,
        }))
        with pytest.raises(RuntimeError, match="supported v13 SFT contract"):
            list(iter_split_samples(root, "train", shuffle=False, include_critic=False))


def test_actor_only_replay_and_batch_omit_private_critic_data() -> None:
    samples = encode_kyoku(
        _first_kyoku(), year=2024, game_id="fixture", kyoku_index=0,
        include_critic=False,
    )
    assert samples
    assert all(sample.critic_length == 0 for sample in samples)
    assert any(sample.value_target != 0.0 for sample in samples)
    batch = collate_samples(samples[:2], torch.device("cpu"), include_critic=False)
    assert "critic_factors" not in batch
    assert "critic_lengths" not in batch
    assert "value_targets" in batch
    assert "teacher_masks" in batch


def test_actor_only_backward_does_not_change_critic_parameters() -> None:
    samples = encode_kyoku(_first_kyoku(), include_critic=False)[:2]
    batch = collate_samples(samples, torch.device("cpu"), include_critic=False)
    config = ModelConfig(
        layers=2, shared_layers=1, critic_layers=1, d_model=32,
        query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64,
        context_tokens=4096,
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
    output = model(
        batch["token_factors"], batch["token_numeric"], batch["legal_mask"],
        batch["token_lengths"], policy_only=True,
    )
    F.cross_entropy(output["policy_logits"], batch["actions"]).backward()
    optimizer.step()
    assert any(parameter.grad is not None for parameter in model.policy_head.parameters())
    assert model.token_embedding.table.weight.grad is not None
    for name, parameter in model.named_parameters():
        if name in before:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_sft_default_is_actor_only() -> None:
    config = load_config(None)
    assert config["train_critic"] is False
    assert config["learning_rate"] == 1.5e-4
    assert config["critic_layers"] == 2


def test_prepare_zip_to_shard_and_joint_loss_backward() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        archive_path = root / "2024.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("2024/fixture.mjson", gzip.compress(FIXTURE.read_bytes(), mtime=0))
        dataset = root / "prepared"
        manifest = prepare_archives(
            {2024: archive_path}, dataset, shard_size=4,
            validation_percent=100, workers=2,
        )
        assert manifest["counts"]["games"] == 1
        assert manifest["counts"]["kyokus"] > 0
        assert manifest["counts"]["validation_decisions"] > 0
        rows = []
        for sample in iter_split_samples(dataset, "validation", shuffle=False):
            rows.append(sample)
            if len(rows) == 4:
                break
        batch = collate_samples(rows, torch.device("cpu"))
        config = ModelConfig(
            layers=2, shared_layers=1, critic_layers=1, d_model=32,
            query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64,
            context_tokens=4096,
        )
        model = KyokuTransformerActorCritic(config)
        output = model(
            batch["token_factors"], batch["token_numeric"], batch["legal_mask"],
            batch["token_lengths"], critic_factors=batch["critic_factors"],
            critic_lengths=batch["critic_lengths"],
        )
        loss = F.cross_entropy(output["policy_logits"], batch["actions"]) + 0.5 * F.huber_loss(
            output["value"], batch["value_targets"]
        )
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.policy_head.parameters())
        assert model.value_head.weight.grad is not None


def test_ppo_model_only_initialization_resets_iteration_and_optimizer() -> None:
    kwargs = {
        "learning_rate": 2e-5,
        "ppo_clip": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "update_epochs": 1,
        "minibatch_size": 2,
        "max_grad_norm": 0.5,
    }
    source = PPOLearner("mid", "cpu", **kwargs)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "sft.pt"
        torch.save({
            "model": source.weights(),
            "token_schema_version": TOKEN_SCHEMA_VERSION,
            "feature_schema_sha256": feature_schema_sha256(),
            "rust_analysis_version": RUST_ANALYSIS_VERSION,
            "decision_analysis_version": DECISION_ANALYSIS_VERSION,
            "training_stage": "sft",
            "training_mode": "actor_only",
            "model_config": asdict(source.config),
        }, path)
        target = PPOLearner("mid", "cpu", **kwargs)
        target.iteration = 99
        target.load_model_weights(path)
        assert target.iteration == 0
        assert not hasattr(target, "critic_warmup_enabled")
        assert not target.optimizer.state
        for name, value in source.model.state_dict().items():
            if name.startswith("value_head."):
                torch.testing.assert_close(
                    target.model.state_dict()[name],
                    torch.zeros_like(target.model.state_dict()[name]),
                )
            else:
                torch.testing.assert_close(target.model.state_dict()[name], value)
