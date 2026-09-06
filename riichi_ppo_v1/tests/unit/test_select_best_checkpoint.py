"""Best-checkpoint 选择逻辑测试:最高 point_diff 获胜,回退到 mean_rank。"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from riichi_ppo_v1.evaluation.select_best_checkpoint import select_best_checkpoint


def _write_summary(directory: Path, update: int, point_diff: float, mean_rank: float) -> None:
    (directory / f"vs_sft_u{update:03d}.json").write_text(json.dumps({
        "hanchan_count": 6000,
        "model_a": {
            "checkpoint": f"checkpoints/train_riichi_v19/ppo/checkpoint_{update:05d}.pt",
            "point_diff_vs_mean_opponent_mean": point_diff,
            "point_diff_vs_mean_opponent_bootstrap_ci95": [point_diff - 1, point_diff + 1],
            "mean_rank": mean_rank,
            "first_place_rate": 0.3,
            "top2_rate": 0.55,
            "fourth_place_rate": 0.15,
        },
    }, ensure_ascii=False))


def test_best_checkpoint_uses_highest_point_diff() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_summary(root, 5, point_diff=-12.0, mean_rank=2.4)
        _write_summary(root, 15, point_diff=+8.0, mean_rank=2.1)
        _write_summary(root, 35, point_diff=-3.0, mean_rank=1.9)
        result = select_best_checkpoint(root)
        assert result["best"]["update"] == 15
        assert result["best"]["metric_value"] == 8.0
        assert result["best"]["checkpoint"].endswith("checkpoint_00015.pt")


def test_best_checkpoint_falls_back_to_mean_rank() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_summary(root, 5, point_diff=0.0, mean_rank=2.5)
        _write_summary(root, 25, point_diff=0.0, mean_rank=1.8)
        result = select_best_checkpoint(root)
        assert result["best"]["update"] == 25  # 打平 → mean_rank 更小者


def test_incomplete_summaries_are_skipped() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "vs_sft_u010.json").write_text(json.dumps({
            "hanchan_count": 2000,  # 只完成一半
            "model_a": {
                "checkpoint": "checkpoints/.../checkpoint_00010.pt",
                "point_diff_vs_mean_opponent_mean": 100.0,
                "mean_rank": 1.0,
            },
        }))
        _write_summary(root, 20, point_diff=4.0, mean_rank=2.2)
        result = select_best_checkpoint(root)
        assert result["best"]["update"] == 20
        assert len(result["summaries"]) == 2  # 两个都记录,但只选完整的


def test_no_complete_summary_raises() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "vs_sft_u010.json").write_text(json.dumps({
            "hanchan_count": 100,
            "model_a": {
                "checkpoint": "checkpoints/.../checkpoint_00010.pt",
                "point_diff_vs_mean_opponent_mean": 1.0,
                "mean_rank": 2.0,
            },
        }))
        with pytest.raises(RuntimeError, match="no complete 1v3 summaries"):
            select_best_checkpoint(root)
