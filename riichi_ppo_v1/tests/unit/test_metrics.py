from __future__ import annotations

import json

import numpy as np

from riichi_ppo_v1.training.evaluation import evaluation_cases, merge_evaluation_summaries
from riichi_ppo_v1.training.metrics import SemanticMetrics, action_kind, append_metric_jsonl, metric_counters, ppo_buffer_metrics
from riichi_ppo_v1.training.trajectory import Transition


def _hora(actor: int, target: int) -> str:
    return json.dumps({"type": "hora", "actor": actor, "target": target, "deltas": [0, 0, 0, 0]})


def test_action_protocol_categories_cover_fixed_boundaries() -> None:
    assert [action_kind(value) for value in (0, 1, 2, 75, 76, 133, 170, 171, 205, 239, 240)] == [
        "pass", "discard", "tsumogiri", "riichi", "chi", "pon", "daiminkan", "ankan", "kakan", "hora", "kyushu",
    ]


def test_hora_multi_ron_and_draw_metrics_are_deduplicated() -> None:
    metrics = SemanticMetrics()
    # The same public events are received by each observation cursor.
    events = [[_hora(0, 2), _hora(1, 2), json.dumps({"type": "end_kyoku"})], [_hora(0, 2), _hora(1, 2)], [], []]
    metrics.record_kyoku([0, 2], [8000, 3000, -11000, 0], events, discard_count=12, open_meld_count=3)
    summary = metrics.summary()
    assert summary["train/kyoku/count"] == 2
    assert summary["train/kyoku/win_rate"] == 0.5
    assert summary["train/kyoku/deal_in_rate"] == 0.5
    assert summary["train/kyoku/win_points_mean"] == 8.0
    assert summary["train/kyoku/deal_in_points_mean"] == -11.0
    assert summary["train/kyoku/discard_count_mean"] == 12.0
    assert summary["train/kyoku/open_melds_mean"] == 3.0

    draw = SemanticMetrics()
    draw.record_kyoku([0], [1500, -500, -500, -500], [[json.dumps({"type": "ryukyoku", "deltas": [1500, -500, -500, -500]})], [], [], []], discard_count=18)
    assert draw.summary()["train/kyoku/draw_rate"] == 1.0
    assert draw.summary()["train/kyoku/discard_count_mean"] == 18.0


def test_metrics_jsonl_has_schema_and_resume_counters(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    append_metric_jsonl(path, update=3, global_decisions=10, global_kyokus=4, source="train", metrics={"train/x": 1.0})
    row = json.loads(path.read_text())
    assert row["schema_version"] == 1 and row["metrics"] == {"train/x": 1.0}
    assert metric_counters(path) == (10, 4)


def test_match_placement_metrics_are_candidate_centric_and_tie_stable() -> None:
    metrics = SemanticMetrics()
    metrics.record_match_result(0, [32000, 25000, 22000, 21000], kyoku_count=8, discard_count=120)
    metrics.record_match_result(1, [25000, 25000, 25000, 25000], kyoku_count=10, discard_count=160)
    metrics.record_match_result(0, [21000, 22000, 23000, 24000], kyoku_count=12, discard_count=180)

    summary = metrics.summary("eval")
    assert summary["eval/match/count"] == 3
    assert summary["eval/match/first_place_rate"] == 1 / 3
    assert summary["eval/match/mean_rank"] == 7 / 3
    assert summary["eval/match/top2_rate"] == 2 / 3
    assert summary["eval/match/last_place_rate"] == 1 / 3
    assert summary["eval/match/length_kyokus_mean"] == 10.0
    assert summary["eval/match/discard_count_mean"] == 460 / 3


def test_match_length_metrics_support_physical_self_play_hanchans() -> None:
    metrics = SemanticMetrics()
    metrics.record_match_length(8)
    metrics.record_match_length(12)

    summary = metrics.summary("train")
    assert summary["train/match/count"] == 2
    assert summary["train/match/length_kyokus_mean"] == 10.0
    assert summary["train/match/mean_rank"] == 0.0


def test_ppo_buffer_and_evaluation_matrix_metrics() -> None:
    transition = Transition(np.zeros((1, 10), np.uint8), np.zeros((1, 8), np.float32), 1,
                            np.ones(241, np.bool_), 0, 0.0, 1.0)
    transition.return_, transition.advantage = 2.0, 1.0
    result = ppo_buffer_metrics([transition])
    assert result["buffer/return_mean"] == 2.0
    assert "explained_variance" in result
    first = evaluation_cases(10, 100, cycle=0)
    second = evaluation_cases(10, 100, cycle=1)
    assert len(first) == len(second) == 100
    assert [seat for _seed, seat, _recipe in first].count(0) == 25
    assert [seat for _seed, seat, _recipe in second].count(0) == 25
    assert {seat for _seed, seat, _recipe in first + second} == {0, 1, 2, 3}
    assert all([seat for _seed, seat, _recipe in first + second].count(seat) == 50 for seat in range(4))
    merged = merge_evaluation_summaries([
        {"eval/kyoku/count": 2.0, "eval/kyoku/point_delta_mean": 1.0, "eval/match/count": 1.0, "eval/match/mean_rank": 1.0},
        {"eval/kyoku/count": 2.0, "eval/kyoku/point_delta_mean": 3.0, "eval/match/count": 3.0, "eval/match/mean_rank": 3.0},
    ])
    assert merged["eval/kyoku/point_delta_mean"] == 2.0
    assert merged["eval/kyoku/point_delta_mean_stderr"] == 1.0
    assert merged["eval/match/count"] == 4.0
    assert merged["eval/match/mean_rank"] == 2.5
