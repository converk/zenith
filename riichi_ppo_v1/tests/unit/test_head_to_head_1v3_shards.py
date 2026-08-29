"""Shard merge consistency: 10x400 synthetic shards == one 4000 aggregate."""

from __future__ import annotations

import hashlib
import numpy as np
import pytest

from riichi_ppo_v1.evaluation.head_to_head_1v3_shards import (
    checkpoint_sha256,
    merge_1v3_shards,
    pooled_bootstrap_ci,
    summary_matches_checkpoint,
    validate_1v3_shard_plan,
    validate_non_overlapping_seed_ranges,
)


def synthetic_shard(index: int, *, seed_base: int = 20260812) -> dict:
    rng = np.random.default_rng(seed_base + index)
    hanchans = 400
    point_diffs = rng.normal(loc=5.0 - 0.2 * index, scale=40.0, size=hanchans)
    ranks = np.asarray([
        1 if value > 0 else (2 if value > -5 else (4 if value < -35 else 3))
        for value in point_diffs
    ])
    first_places = int(np.count_nonzero(ranks == 1))
    second_places = int(np.count_nonzero(ranks == 2))
    third_places = int(np.count_nonzero(ranks == 3))
    top2 = int(np.count_nonzero(ranks <= 2))
    fourths = int(np.count_nonzero(ranks == 4))
    kyoku_count = 20 + (index % 5) * 5
    win_rate = 0.12 + 0.005 * index
    deal_in_rate = 0.10 - 0.004 * index
    tsumo_loss_rate = 0.08 + 0.002 * index
    riichi_rate = 0.20 - 0.001 * index
    draw_tenpai_rate = 0.5 + 0.01 * (index % 3)
    kyoku_metrics = {
        "kyoku_count": float(kyoku_count),
        "win_rate": win_rate,
        "deal_in_rate": deal_in_rate,
        "tsumo_loss_rate": tsumo_loss_rate,
        "riichi_rate": riichi_rate,
        "draw_tenpai_rate": draw_tenpai_rate,
        "win_points_mean": 5500.0 + index,
        "riichi_opportunity_count": float(kyoku_count * 3),
        "riichi_opportunity_accept_rate": 0.6 - 0.01 * index,
    }
    return {
        "protocol_version": 1,
        "format": "1v3",
        "hanchan_count": hanchans,
        "seed_base": seed_base + index * hanchans,
        "elapsed_s": 20.0 + 0.5 * index,
        "model_a": {
            "checkpoint": f"/tmp/checkpoint_{index:05d}.pt",
            "first_place_count": first_places,
            "second_place_count": second_places,
            "third_place_count": third_places,
            "top2_count": top2,
            "fourth_place_count": fourths,
            "mean_rank": float(np.mean(ranks)),
            "final_score_mean": 25000.0 + index,
            "flying_rate": 0.01 * (index % 2),
            "point_diff_samples": [float(value) for value in point_diffs],
            "kyoku_metrics": kyoku_metrics,
            "semantic_metrics": {
                "model_a/match/count": float(hanchans),
                "model_a/match/final_score_mean": 25000.0 + index,
                "model_a/match/flying_rate": 0.01 * (index % 2),
            },
        },
        "model_b": {"checkpoint": "/tmp/best_heuristic.pt"},
    }


def test_merged_4000_hanchans_match_direct_aggregation() -> None:
    seed_base = 20260812
    shards = [synthetic_shard(index, seed_base=seed_base) for index in range(10)]
    samples = np.concatenate([
        np.asarray(shard["model_a"]["point_diff_samples"], dtype=np.float64)
        for shard in shards
    ])

    merged = merge_1v3_shards(shards, seed_base=seed_base, update=5)
    model_a = merged["model_a"]

    assert merged["hanchan_count"] == 4000
    assert merged["update"] == 5
    assert model_a["first_place_count"] == sum(
        shard["model_a"]["first_place_count"] for shard in shards
    )
    assert model_a["first_place_rate"] == pytest.approx(
        model_a["first_place_count"] / 4000
    )
    assert model_a["top2_count"] == sum(
        shard["model_a"]["top2_count"] for shard in shards
    )
    assert model_a["second_place_count"] == sum(
        shard["model_a"]["second_place_count"] for shard in shards
    )
    assert model_a["third_place_count"] == sum(
        shard["model_a"]["third_place_count"] for shard in shards
    )
    assert model_a["fourth_place_count"] == sum(
        shard["model_a"]["fourth_place_count"] for shard in shards
    )
    assert model_a["mean_rank"] == pytest.approx(
        sum(
            shard["model_a"]["mean_rank"] * shard["hanchan_count"]
            for shard in shards
        ) / 4000
    )
    assert model_a["point_diff_mean"] == pytest.approx(float(samples.mean()))
    assert model_a["second_place_rate"] == pytest.approx(model_a["second_place_count"] / 4000)
    assert model_a["third_place_rate"] == pytest.approx(model_a["third_place_count"] / 4000)
    assert model_a["final_score_mean"] == pytest.approx(
        np.mean([25000.0 + index for index in range(10)])
    )
    assert model_a["flying_rate"] == pytest.approx(
        np.mean([0.01 * (index % 2) for index in range(10)])
    )
    assert model_a["semantic_metrics"]["model_a/match/count"] == 4000.0
    assert model_a["point_diff_bootstrap_ci95"] == pytest.approx(
        pooled_bootstrap_ci(samples, seed_base)
    )
    assert len(model_a["point_diff_samples"]) == 4000
    assert model_a["point_diff_samples"] == pytest.approx(samples.tolist())


def test_kyoku_metrics_are_weighted_by_kyoku_count() -> None:
    shards = [synthetic_shard(index) for index in range(10)]
    merged = merge_1v3_shards(shards, seed_base=20260812, update=90)
    metrics = merged["model_a"]["kyoku_metrics"]

    total_kyokus = sum(
        shard["model_a"]["kyoku_metrics"]["kyoku_count"] for shard in shards
    )
    assert metrics["kyoku_count"] == total_kyokus
    for name in ("win_rate", "deal_in_rate", "tsumo_loss_rate", "riichi_rate"):
        expected = sum(
            shard["model_a"]["kyoku_metrics"][name]
            * shard["model_a"]["kyoku_metrics"]["kyoku_count"]
            for shard in shards
        ) / total_kyokus
        assert metrics[name] == pytest.approx(expected)


def test_merge_rejects_mismatched_point_diff_samples() -> None:
    shards = [synthetic_shard(0), synthetic_shard(1)]
    shards[1]["model_a"]["point_diff_samples"] = shards[1]["model_a"][
        "point_diff_samples"
    ][:-1]
    with pytest.raises(RuntimeError, match="point-diff samples total"):
        merge_1v3_shards(shards, seed_base=20260812, update=5)


def test_shard_seed_ranges_are_disjoint_and_overlap_is_rejected() -> None:
    shards = [synthetic_shard(index) for index in range(10)]
    validate_non_overlapping_seed_ranges(shards)
    assert [shard["seed_base"] for shard in shards] == [
        20260812 + index * 400 for index in range(10)
    ]
    shards[1]["seed_base"] = shards[0]["seed_base"] + 1
    with pytest.raises(RuntimeError, match="seed ranges overlap"):
        validate_non_overlapping_seed_ranges(shards)


def test_project_1v3_protocol_requires_ten_exact_disjoint_shards() -> None:
    shards = [synthetic_shard(index) for index in range(10)]
    validate_1v3_shard_plan(
        shards, seed_base=20260812, hanchans_per_process=400,
    )
    with pytest.raises(RuntimeError, match="exactly 10 shards"):
        validate_1v3_shard_plan(
            shards[:-1], seed_base=20260812, hanchans_per_process=400,
        )
    shards[1]["seed_base"] = shards[0]["seed_base"] + 1
    with pytest.raises(RuntimeError, match="seed plan differs"):
        validate_1v3_shard_plan(
            shards, seed_base=20260812, hanchans_per_process=400,
        )


def test_pooled_bootstrap_ci_is_deterministic() -> None:
    samples = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    assert pooled_bootstrap_ci(samples, 42) == pooled_bootstrap_ci(samples, 42)


def test_checkpoint_sha256_matches_hashlib(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint_00001.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    assert checkpoint_sha256(checkpoint) == hashlib.sha256(
        b"checkpoint-bytes"
    ).hexdigest()


def test_summary_cache_validation(tmp_path) -> None:
    """缓存命中校验:旧格式(无指纹)/同哈希/异哈希三例,仅同哈希可复用。"""
    checkpoint = tmp_path / "checkpoint_00001.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    digest = checkpoint_sha256(checkpoint)

    # 旧格式:summary 无 checkpoint_sha256 记录 → 一律视为未命中(需重跑)。
    assert not summary_matches_checkpoint({"shards": []}, checkpoint)
    assert not summary_matches_checkpoint({"checkpoint_sha256": ""}, checkpoint)
    assert not summary_matches_checkpoint(None, checkpoint)
    # 指纹与 checkpoint 内容一致 → 命中。
    assert summary_matches_checkpoint({"checkpoint_sha256": digest}, checkpoint)
    # 指纹不一致(checkpoint 已被覆盖为不同内容)→ 未命中(需重跑覆盖)。
    assert not summary_matches_checkpoint(
        {"checkpoint_sha256": hashlib.sha256(b"other-content").hexdigest()},
        checkpoint,
    )
    # 评测参数指纹:参数一致才命中;缺失记录/参数变更一律未命中。
    params = {"seed_base": 20260829, "hanchans_per_process": 600,
              "parallel_hanchans": 300}
    with_params = {"checkpoint_sha256": digest, "eval_params": params}
    assert summary_matches_checkpoint(with_params, checkpoint, eval_params=params)
    # 旧调用(不传参数)保持仅指纹校验的历史语义。
    assert summary_matches_checkpoint(with_params, checkpoint)
    assert not summary_matches_checkpoint(
        with_params, checkpoint,
        eval_params={**params, "seed_base": 20260827},
    )
    assert not summary_matches_checkpoint(
        {"checkpoint_sha256": digest}, checkpoint, eval_params=params,
    )
