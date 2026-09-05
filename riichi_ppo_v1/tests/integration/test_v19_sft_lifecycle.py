"""V19 小规模 SFT 生命周期：replay→precompute→collate→信念联合训练→checkpoint。

必须覆盖：encode_kyoku 产出信念五头标签、precompute 持久化/读取、2 步 CPU 单卡
训练（batch_size=8）产生有限且非 NaN 的 belief 曲线。
"""

from __future__ import annotations

import io
import json
import math
import tarfile
from pathlib import Path

import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.sft.contract import dataset_manifest_hash
from riichi_ppo_v1.sft.data import iter_split_samples
from riichi_ppo_v1.sft.precompute import precompute
from riichi_ppo_v1.sft.trainer import collate_samples, main as train_main
from riichi_ppo_v1.tests.v18_fixtures import first_kyoku_record


def _build_source(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    (source / "train").mkdir(parents=True, exist_ok=True)
    record, _game_id = first_kyoku_record()
    def add(split: str, member: str) -> None:
        (source / split).mkdir(parents=True, exist_ok=True)
        with tarfile.open(source / split / f"{split}-00000.tar", "w") as archive:
            payload = record.encode("utf-8")
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    add("train", "2024-test-1-00.mjson")
    add("validation", "2025-test-2-00.mjson")
    (source / "manifest.json").write_text(json.dumps({
        "format": "riichi-sft-kyoku-v1",
        "years": [2024, 2025],
        "counts": {"train": 1, "validation": 1,
                   "train_decisions": 65, "validation_decisions": 65},
    }), encoding="utf-8")
    return source


def _train_config(encoded: Path, output: Path) -> dict[str, object]:
    return {
        "seed": 0,
        "device": "cpu",
        "learner_gpus": 1,
        "model_size": "v19",
        "context_tokens": 320,
        "policy_head_type": "current_state_snapshot",
        "dense_slot_dim": 32,
        "dense_fusion_dim": 512,
        "epochs": 1,
        "train_critic": False,
        "train_public_value": False,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "min_learning_rate": 1e-5,
        "warmup_fraction": 0.02,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "inference_dtype": "bf16",
        "length_bucket_window_batches": 32,
        "log_interval_steps": 1,
        "validate_semantics": False,
        "validation_max_samples": 0,
        "validation_samples_per_run": 16,
        "max_train_steps": 2,
        "stop_after_steps": 2,
        "tensorboard_enabled": False,
        "tensorboard_dirname": "tensorboard",
        "resume": None,
        "init_model": None,
        "belief_sft_coef": 1.0,
        "belief_head_weight_hand": 1.0,
        "belief_head_weight_shanten": 1.0,
        "belief_head_weight_wait": 1.0,
        "belief_head_weight_danger": 1.0,
        "belief_head_weight_loss": 1.0,
        "belief_wait_danger_weight": 0.05,
        "dataset": str(encoded),
        "checkpoint_dir": str(output),
    }


def test_small_sft_lifecycle(tmp_path: Path) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "encoded"
    precompute(source, output, denominator=1, remainders=(0,), kyokus_per_shard=1, workers=1)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "riichi-sft-encoded-v19"
    assert manifest["belief_labels"] is True
    assert manifest["belief_shape"] == {
        "hand": [102], "shanten": [3], "wait": [105], "danger": [102], "loss": [102],
    }
    assert manifest["counts"]["train_kyokus"] == 1
    assert manifest["counts"]["train_decisions"] > 0

    samples = list(iter_split_samples(output, "train", seed=0, shuffle=False, include_critic=False))
    assert samples
    for sample in samples[:8]:
        assert sample.belief_hand.shape == (102,)
        assert sample.belief_shanten.shape == (3,)
        assert sample.belief_wait.shape == (105,)
        assert sample.belief_danger.shape == (102,)
        assert sample.belief_loss.shape == (102,)
        assert torch.isfinite(torch.as_tensor(sample.belief_loss)).all()

    batch = collate_samples(samples[:8], torch.device("cpu"))
    model = KyokuTransformerActorCritic()
    with torch.no_grad():
        output_forward = model(
            actor_factors=batch["actor_factors"], actor_numeric=batch["actor_numeric"],
            actor_lengths=batch["actor_lengths"], query_action_ids=batch["action_ids"],
            query_pair_counts=batch["pair_counts"], legal_mask=batch["legal_mask"],
            policy_only=True,
        )
    for key in (
        "belief_hand_logits", "belief_shanten_logits", "belief_wait_logits",
        "belief_danger_logits", "belief_loss_pred",
    ):
        assert torch.isfinite(output_forward[key].float()).all(), key

    train_dir = tmp_path / "train"
    train_main(_train_config(output, train_dir), dataset=output, output=train_dir)

    metrics = json.loads((train_dir / "metrics.json").read_text(encoding="utf-8"))
    assert math.isfinite(metrics["validation/policy_ce"])
    for key in (
        "train/belief_hand_acc", "train/belief_shanten_top1", "train/belief_wait_topk",
        "train/belief_danger_auc", "train/belief_loss_mae",
        "validation/belief_hand_acc", "validation/belief_shanten_top1",
        "validation/belief_wait_topk", "validation/belief_danger_auc",
        "validation/belief_loss_mae",
    ):
        assert key in metrics, key
        assert math.isfinite(float(metrics[key])), key

    checkpoint = train_dir / "latest.pt"
    assert checkpoint.is_file()
    assert dataset_manifest_hash(output)  # manifest 哈希在 checkpoint 路径上可读
