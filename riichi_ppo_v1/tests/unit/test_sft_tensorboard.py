from __future__ import annotations

import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from riichi_ppo_v1.model import ModelConfig
from riichi_ppo_v1.sft import train as sft_train
from riichi_ppo_v1.sft.prepare import _json_lines, prepare_archives, split_game_events, stable_split
from riichi_ppo_v1.sft.tensorboard import SftMetricWindow, write_sft_scalars


FIXTURE = Path(__file__).parents[3] / "RiichiEnv/tests/data/126_204_0_mjai.jsonl"


def _small_model_config(_config: dict[str, object]) -> ModelConfig:
    return ModelConfig(
        layers=2, shared_layers=1, critic_layers=1, d_model=32,
        query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64,
        context_tokens=4096,
    )


def _first_kyoku() -> bytes:
    events = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line]
    return gzip.compress(_json_lines(split_game_events(events)[0]).encode(), mtime=0)


def _game_ids_on_opposite_sides() -> tuple[str, str]:
    train = next(f"game-{index}" for index in range(1000) if stable_split(f"game-{index}", 50) == "train")
    validation = next(
        f"validation-{index}"
        for index in range(1000)
        if stable_split(f"validation-{index}", 50) == "validation"
    )
    return train, validation


def test_metric_window_and_event_projection_are_actor_only_aware(tmp_path: Path) -> None:
    actions = torch.tensor([0, 1, 75, 76, 133, 170, 239, 240])
    logits = torch.zeros((len(actions), 241))
    logits[torch.arange(len(actions)), actions] = 2.0
    legal = torch.ones_like(logits, dtype=torch.bool)
    lengths = torch.arange(1, len(actions) + 1)
    window = SftMetricWindow()
    window.update(
        logits=logits,
        actions=actions,
        legal_mask=legal,
        token_lengths=lengths,
        loss=torch.tensor(0.8),
        policy_ce=torch.tensor(0.8),
        value_huber=None,
        effective_tokens=int((lengths + 1).sum()),
        padded_tokens=len(actions) * int(lengths.max() + 1),
        step_seconds=0.25,
    )
    metrics = window.scalars()
    assert metrics["train/top1"] == 1.0
    assert metrics["train/top3"] == 1.0
    assert "train/value_huber" not in metrics
    assert all(metrics[f"train/action/{group}/count"] == 1.0 for group in (
        "pass", "discard", "reach", "chi", "pon", "kan", "hora", "ryukyoku",
    ))

    writer = SummaryWriter(str(tmp_path))
    write_sft_scalars(writer, metrics, 3)
    write_sft_scalars(writer, {"validation/policy_ce": 0.7, "validation/top1": 0.5}, 3)
    write_sft_scalars(
        writer, {"heuristic_eval/kyoku/win_rate": 0.2}, 3,
    )
    writer.close()
    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert "SFT/训练/策略交叉熵 (policy_ce)" in tags
    assert "SFT/训练动作/杠/Top-1 准确率 (kan_top1)" in tags
    assert "SFT/验证/policy_ce" in tags
    assert "SFT/启发式对局评测/kyoku/win_rate" in tags
    assert all("价值" not in tag for tag in tags)


def test_output_defaults_to_config_and_cli_override_wins() -> None:
    config = sft_train.load_config(None)
    assert sft_train.resolve_output(config, None) == Path(
        "checkpoints/checkpoints/train_riichi_v10_sft"
    )
    assert sft_train.resolve_output(config, Path("/tmp/custom-sft")) == Path("/tmp/custom-sft")


def test_train_worker_closes_writer_when_training_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeWriter:
        flushed = False
        closed = False

        def flush(self) -> None:
            self.flushed = True

        def close(self) -> None:
            self.closed = True

    writer = FakeWriter()

    def fail(*args: object) -> None:
        args[-1].append(writer)
        raise RuntimeError("expected")

    monkeypatch.setattr(sft_train, "_train_worker_impl", fail)
    with pytest.raises(RuntimeError, match="expected"):
        sft_train.train_worker(0, 1, {}, Path("."), Path("."))
    assert writer.flushed
    assert writer.closed


def test_three_step_training_writes_validation_and_best_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        train_id, validation_id = _game_ids_on_opposite_sides()
        archive_path = root / "2024.zip"
        payload = _first_kyoku()
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(f"2024/{train_id}.mjson", payload)
            archive.writestr(f"2024/{validation_id}.mjson", payload)
        dataset = root / "dataset"
        prepare_archives(
            {2024: archive_path}, dataset,
            shard_size=4, validation_percent=50,
        )
        output = root / "output"
        config = sft_train.load_config(None)
        config.update({
            "device": "cpu",
            "learner_gpus": 1,
            "epochs": 1,
            "batch_size": 32,
            "log_interval_steps": 1,
            "checkpoint_interval_steps": 1,
            "validation_interval_steps": 1,
            "validation_samples_per_run": 8,
            "validation_max_samples": 8,
            "heuristic_evaluation_enabled": False,
            "shuffle_buffer_kyokus": 1,
            "length_bucket_window_batches": 1,
            "checkpoint_dir": str(output),
        })
        monkeypatch.setattr(sft_train, "_model_config", _small_model_config)
        sft_train.train_worker(0, 1, config, dataset, output)

        latest = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        assert latest["global_step"] == 3
        assert best["best_validation_loss"] < float("inf")
        accumulator = EventAccumulator(str(output / "tensorboard"))
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags()["scalars"])
        assert "SFT/训练/Top-1 准确率 (top1)" in scalar_tags
        assert "SFT/验证/loss" in scalar_tags
        assert not any("value_huber" in tag for tag in scalar_tags)

        previous_best = float(latest["best_validation_loss"])
        config["resume"] = str(output / "latest.pt")
        sft_train.train_worker(0, 1, config, dataset, output)
        resumed = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        assert resumed["global_step"] == 3
        assert float(resumed["best_validation_loss"]) <= previous_best
