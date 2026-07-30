import gzip
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tarfile
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
from riichi_ppo_v1.model.critic_features import FIELD_OPPONENT_HAND, SEGMENT_PUBLIC_SUMMARY
from riichi_ppo_v1.model.schema import TOKEN_SCHEMA_VERSION
from riichi_ppo_v1.sft.data import encode_kyoku, iter_split_samples
from riichi_ppo_v1.sft.audit import audit_kyoku, select_coverage_kyokus, validate_encoded_chunk
from riichi_ppo_v1.sft.precompute import _write_chunk
from riichi_ppo_v1.sft.prepare import (
    _json_lines,
    _zip_directory_is_readable,
    prepare_archives,
    split_game_events,
    stable_split,
)
from riichi_ppo_v1.sft.train import collate_samples, load_config
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
    assert np.all(with_public.token_factors[first_public:, 0] == SEGMENT_PUBLIC_SUMMARY)
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
    assert structure["candidate_tokens"] == sum(int(sample.legal_mask.sum()) for sample in samples)
    assert structure["public_tokens"] > 0


def test_encoded_sft_shards_are_disjoint_across_ddp_ranks() -> None:
    samples = encode_kyoku(_first_kyoku(), include_critic=False)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        train = root / "train"
        train.mkdir()
        for index in range(4):
            _write_chunk(train / f"train-{index:05d}.npz", samples[index:index + 1])
        (root / "manifest.json").write_text(json.dumps({
            "format": "riichi-sft-encoded-v1",
            "token_schema_version": TOKEN_SCHEMA_VERSION,
        }))
        rank0 = list(iter_split_samples(
            root, "train", seed=7, shuffle=False, rank=0, world_size=2,
            include_critic=False,
        ))
        rank1 = list(iter_split_samples(
            root, "train", seed=7, shuffle=False, rank=1, world_size=2,
            include_critic=False,
        ))
    assert [sample.action for sample in rank0] == [samples[0].action, samples[2].action]
    assert [sample.action for sample in rank1] == [samples[1].action, samples[3].action]


def test_actor_only_replay_and_batch_omit_private_critic_data() -> None:
    samples = encode_kyoku(
        _first_kyoku(), year=2024, game_id="fixture", kyoku_index=0,
        include_critic=False,
    )
    assert samples
    assert all(sample.critic_length == 0 for sample in samples)
    assert all(sample.value_target == 0.0 for sample in samples)
    batch = collate_samples(samples[:2], torch.device("cpu"), include_critic=False)
    assert "critic_factors" not in batch
    assert "critic_lengths" not in batch
    assert "value_targets" not in batch


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
    assert model.policy_head.weight.grad is not None
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
        assert model.policy_head.weight.grad is not None
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
            "training_stage": "sft",
            "training_mode": "actor_only",
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
