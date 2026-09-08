"""跨代 2v2 评测的单元测试:座位轮换 / 分片计划校验 / 合并聚合 / 消息帧。

纯 Python 逻辑测试,不依赖 GPU、Rust 扩展或真实 checkpoint。
"""

from __future__ import annotations

import json
import socket
import sys
from typing import Any

import pytest

from riichi_ppo_v1.evaluation.head_to_head_2v2 import (
    TEAM_A_PAIR_EVEN,
    TEAM_A_PAIR_ODD,
    recv_msg,
    send_msg,
    team_a_seats_for,
)
from riichi_ppo_v1.evaluation.head_to_head_2v2_shards import (
    DEFAULT_2V2_HANCHANS_PER_PROCESS,
    REQUIRED_2V2_PROCESSES,
    TOTAL_2V2_HANCHANS,
    _weighted_metrics,
    merge_2v2_shards,
    validate_2v2_shard_plan,
)
from riichi_ppo_v1.evaluation.v19_eval_runtime import ensure_v19_runtime

HANCHANS = DEFAULT_2V2_HANCHANS_PER_PROCESS


# ---------- 座位轮换 ----------

def test_team_a_seats_are_opposite_pairs() -> None:
    """model_a 的两席必须固定为一组对家,且只有 {0,2}/{1,3} 两种组合。"""
    allowed = {TEAM_A_PAIR_EVEN, TEAM_A_PAIR_ODD}
    for index in range(64):
        assert team_a_seats_for(index) in allowed


def test_team_a_seat_rotation_is_balanced() -> None:
    """奇偶轮换保证两种对家组合恰好各占一半。"""
    even = sum(team_a_seats_for(i) == TEAM_A_PAIR_EVEN for i in range(6000))
    odd = sum(team_a_seats_for(i) == TEAM_A_PAIR_ODD for i in range(6000))
    assert even == odd == 3000


# ---------- 分片计划校验 ----------

def _shard(shard: int, seed_base: int, hanchans: int = HANCHANS) -> dict[str, Any]:
    """构造最小合法分片摘要。"""
    return {
        "hanchan_count": hanchans,
        "seed_base": seed_base,
        "elapsed_s": 1.0,
        "team_a_seat_rotation": "global_hanchan_index % 2: even={0,2}, odd={1,3}",
        "model_a_seat_counts": {"0": hanchans},
        "model_b_seat_counts": {"1": hanchans},
        "model_a": {
            "checkpoint": "a.pt",
            "metadata": {"contract_id": "riichi-runtime-v19"},
            "sample_count": 2 * hanchans,
            "first_place_count": 2 * hanchans,
            "second_place_count": 0,
            "third_place_count": 0,
            "fourth_place_count": 0,
            "mean_rank": 1.0,
            "point_diff_samples": [100.0] * (2 * hanchans),
            "flying_rate": 0.0,
            "team_point_diff_samples": [200.0] * hanchans,
            "action_type_rates": {"discard": 1.0},
            "kyoku_metrics": {"kyoku_count": 10 * hanchans, "win_rate": 0.2},
            "semantic_metrics": {
                "model_a/match/count": 2 * hanchans,
                "model_a/match/final_score_mean": 25000.0,
            },
            "belief_metrics": {"hand_accuracy": 0.5, "decision_count": 10 * hanchans},
        },
        "model_b": {
            "checkpoint": "b.pt",
            "metadata": {"contract_id": "riichi-runtime-v18"},
            "sample_count": 2 * hanchans,
            "first_place_count": 0,
            "second_place_count": 0,
            "third_place_count": 0,
            "fourth_place_count": 2 * hanchans,
            "mean_rank": 3.0,
            "point_diff_samples": [-100.0] * (2 * hanchans),
            "flying_rate": 0.0,
            "action_type_rates": {"discard": 1.0},
            "kyoku_metrics": {"kyoku_count": 10 * hanchans, "win_rate": 0.1},
            "semantic_metrics": {
                "model_b/match/count": 2 * hanchans,
                "model_b/match/final_score_mean": 25000.0,
            },
        },
    }


def _canonical_shards(seed_base: int) -> list[dict[str, Any]]:
    return [
        _shard(shard, seed_base + shard * HANCHANS)
        for shard in range(REQUIRED_2V2_PROCESSES)
    ]


def test_validate_plan_accepts_canonical_allocation() -> None:
    validate_2v2_shard_plan(
        _canonical_shards(700000000),
        seed_base=700000000,
        hanchans_per_process=HANCHANS,
    )


def test_validate_plan_rejects_wrong_shard_count() -> None:
    shards = _canonical_shards(700000000)[:-1]
    with pytest.raises(RuntimeError, match="exactly 10"):
        validate_2v2_shard_plan(
            shards, seed_base=700000000, hanchans_per_process=HANCHANS,
        )


def test_validate_plan_rejects_dislocated_seed_base() -> None:
    shards = _canonical_shards(700000000)
    shards[5]["seed_base"] += 1
    with pytest.raises(RuntimeError, match="disjoint allocation"):
        validate_2v2_shard_plan(
            shards, seed_base=700000000, hanchans_per_process=HANCHANS,
        )


def test_total_hanchans_matches_mechanism() -> None:
    assert TOTAL_2V2_HANCHANS == REQUIRED_2V2_PROCESSES * HANCHANS == 6000


# ---------- 合并聚合 ----------

def test_merge_aggregates_counts_rates_and_team_diff() -> None:
    seed_base = 700000000
    summary = merge_2v2_shards(
        _canonical_shards(seed_base), seed_base=seed_base,
        hanchans_per_process=HANCHANS,
    )
    total = REQUIRED_2V2_PROCESSES * HANCHANS
    assert summary["hanchan_count"] == total == 6000
    a, b = summary["model_a"], summary["model_b"]
    # 每队每半庄 2 个座位样本
    assert a["sample_count"] == b["sample_count"] == 2 * total
    # 合成数据:模型_a 全部一位,模型_b 全部四位
    assert a["first_place_rate"] == 1.0
    assert b["fourth_place_rate"] == 1.0
    assert a["mean_rank"] == 1.0 and b["mean_rank"] == 3.0
    # 点差样本逐座拼接,队伍点差逐半庄拼接
    assert len(a["point_diff_samples"]) == 2 * total
    assert len(a["team_point_diff_samples"]) == total
    assert a["point_diff_vs_mean_others_mean"] == pytest.approx(100.0)
    assert a["team_point_diff_mean"] == pytest.approx(200.0)
    # 语义/小局面按 match/kyoku 计数加权
    assert a["final_score_mean"] == pytest.approx(25000.0)
    assert a["kyoku_metrics"]["kyoku_count"] == 10 * total
    assert a["kyoku_metrics"]["win_rate"] == pytest.approx(0.2)
    # 信念面仅 model_a 输出,且按决策数加权
    assert a["belief_metrics"]["hand_accuracy"] == pytest.approx(0.5)
    assert a["belief_metrics"]["decision_count"] == 10 * total
    assert "belief_metrics" not in b
    # 动作分组率按半庄数加权合并,且不混入权重键
    assert a["action_type_rates"] == {"discard": pytest.approx(1.0)}
    assert "hanchans" not in a["action_type_rates"]
    # 双方 checkpoint 与种子记录完整
    assert summary["model_a"]["checkpoint"] == "a.pt"
    assert summary["model_b"]["checkpoint"] == "b.pt"
    assert summary["seed_base"] == seed_base
    assert summary["format"] == "2v2_sharded"


def test_merge_rejects_inconsistent_point_diff_samples() -> None:
    shards = _canonical_shards(700000000)
    shards[0]["model_a"]["point_diff_samples"] = shards[0][
        "model_a"
    ]["point_diff_samples"][:-1]
    with pytest.raises(RuntimeError, match="seat samples"):
        merge_2v2_shards(
            shards, seed_base=700000000, hanchans_per_process=HANCHANS,
        )


def test_weighted_metrics_sums_counts_and_weights_rates() -> None:
    rows = [
        {"win_rate": 0.2, "kyoku_count": 100.0},
        {"win_rate": 0.6, "kyoku_count": 300.0},
    ]
    merged = _weighted_metrics(rows, "kyoku_count")
    assert merged["kyoku_count"] == 400.0
    assert merged["win_rate"] == pytest.approx(0.5)  # (0.2*100 + 0.6*300) / 400


# ---------- 锁步消息帧 ----------

def test_message_framing_roundtrip() -> None:
    host, partner = socket.socketpair()
    try:
        payload = {"t": "act", "acts": [[1, 3, '{"type":"dahai"}'], [0, 2, None]]}
        send_msg(host, payload)
        assert recv_msg(partner) == payload
        send_msg(partner, {"t": "act", "acts": []})
        assert recv_msg(host) == {"t": "act", "acts": []}
    finally:
        host.close()
        partner.close()


def test_recv_msg_raises_on_partner_disconnect() -> None:
    host, partner = socket.socketpair()
    try:
        partner.close()
        with pytest.raises(RuntimeError, match="disconnected"):
            recv_msg(host)
    finally:
        host.close()


def test_json_payload_survives_unicode() -> None:
    host, partner = socket.socketpair()
    try:
        payload = {"t": "final", "note": "跨代评测汇总"}
        send_msg(host, payload)
        assert recv_msg(partner) == payload
    finally:
        host.close()
        partner.close()


# ---------- V19 评测运行时引导 ----------

def test_runtime_bootstrap_rejects_missing_build(tmp_path, monkeypatch) -> None:
    # 测试进程可能已间接导入 riichienv,先摘除以模拟入口引导前的干净状态。
    for name in ("riichi", "riichienv"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(RuntimeError, match="扩展缺失"):
        ensure_v19_runtime(tmp_path)


def test_runtime_bootstrap_rejects_late_invocation(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "riichi", object())
    with pytest.raises(RuntimeError, match="引导"):
        ensure_v19_runtime("/nonexistent")


# 汇总文件契约:分片 JSON 必须可 json 序列化回读(评测落盘协议)。
def test_shard_payload_is_json_serializable() -> None:
    shard = _shard(0, 700000000)
    assert json.loads(json.dumps(shard)) == shard
