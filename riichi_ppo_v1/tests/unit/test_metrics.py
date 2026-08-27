from __future__ import annotations

import json

import numpy as np

from riichi_ppo_v1.training.metrics import SemanticMetrics, action_kind, append_metric_jsonl, metric_counters, ppo_buffer_metrics
from riichi_ppo_v1.training.trajectory import Transition


def _hora(actor: int, target: int) -> str:
    return json.dumps({"type": "hora", "actor": actor, "target": target, "deltas": [0, 0, 0, 0]})


def test_action_protocol_categories_cover_fixed_boundaries() -> None:
    assert [action_kind(value) for value in (0, 1, 2, 75, 76, 133, 170, 171, 205, 239, 240)] == [
        "pass", "discard", "tsumogiri", "riichi", "chi", "pon", "daiminkan", "ankan", "kakan", "hora", "kyushu",
    ]


def test_riichi_metric_reports_opportunity_acceptance() -> None:
    metrics = SemanticMetrics()
    riichi_legal = np.zeros(241, dtype=np.bool_)
    riichi_legal[[1, 75]] = True
    no_riichi_legal = np.zeros(241, dtype=np.bool_)
    no_riichi_legal[1] = True

    metrics.record_decision(75, riichi_legal)
    metrics.record_decision(1, riichi_legal)
    metrics.record_decision(1, no_riichi_legal)

    summary = metrics.summary()
    assert summary["train/action/riichi_opportunity_count"] == 2
    assert summary["train/action/riichi_opportunity_accept_rate"] == 0.5
    assert summary["train/action/riichi_rate"] == 1 / 3


def test_hora_multi_ron_and_draw_metrics_are_deduplicated() -> None:
    metrics = SemanticMetrics()
    # The same public events are received by each observation cursor.
    events = [[json.dumps({"type": "reach", "actor": 0}), _hora(0, 2), _hora(1, 2), json.dumps({"type": "end_kyoku"})], [_hora(0, 2), _hora(1, 2)], [], []]
    metrics.record_kyoku([0, 2], [8000, 3000, -11000, 0], events, discard_count=12, open_meld_count=3)
    summary = metrics.summary()
    assert summary["train/kyoku/count"] == 2
    assert summary["train/kyoku/win_rate"] == 0.5
    assert summary["train/kyoku/deal_in_rate"] == 0.5
    assert summary["train/kyoku/win_points_mean"] == 8.0
    assert summary["train/kyoku/deal_in_points_mean"] == -11.0
    assert summary["train/kyoku/discard_count_mean"] == 12.0
    assert summary["train/kyoku/open_melds_mean"] == 3.0
    assert summary["train/kyoku/riichi_rate"] == 0.5
    assert summary["train/kyoku/post_riichi_win_rate"] == 1.0

    draw = SemanticMetrics()
    draw.record_kyoku([0], [1500, -500, -500, -500], [[json.dumps({"type": "ryukyoku", "deltas": [1500, -500, -500, -500]})], [], [], []], discard_count=18)
    assert draw.summary()["train/kyoku/draw_rate"] == 1.0
    assert draw.summary()["train/kyoku/discard_count_mean"] == 18.0

    tsumo = SemanticMetrics()
    tsumo.record_kyoku([0], [-2000, 6000, -2000, -2000], [[_hora(1, 1)], [], [], []])
    assert tsumo.summary()["train/kyoku/tsumo_loss_rate"] == 1.0


def test_metrics_jsonl_has_schema_and_resume_counters(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    append_metric_jsonl(path, update=3, global_decisions=10, global_kyokus=4, source="train", metrics={"train/x": 1.0})
    row = json.loads(path.read_text())
    assert row["schema_version"] == 1 and row["metrics"] == {"train/x": 1.0}
    assert metric_counters(path) == (10, 4)


def test_match_placement_metrics_are_candidate_centric_and_tie_stable() -> None:
    metrics = SemanticMetrics()
    metrics.record_match_result(0, [32000, 25000, 22000, 21000], kyoku_count=8, discard_count=120, point_delta=7000)
    metrics.record_match_result(1, [25000, 25000, 25000, 25000], kyoku_count=10, discard_count=160, point_delta=0)
    metrics.record_match_result(0, [21000, 22000, 23000, 24000], kyoku_count=12, discard_count=180, point_delta=-4000)

    summary = metrics.summary("eval")
    assert summary["eval/match/count"] == 3
    assert summary["eval/match/first_place_rate"] == 1 / 3
    assert summary["eval/match/second_place_rate"] == 1 / 3
    assert summary["eval/match/third_place_rate"] == 0.0
    assert summary["eval/match/mean_rank"] == 7 / 3
    assert summary["eval/match/top2_rate"] == 2 / 3
    assert summary["eval/match/last_place_rate"] == 1 / 3
    assert summary["eval/match/point_delta_mean"] == 1000.0
    assert summary["eval/match/positive_point_delta_rate"] == 1 / 3
    assert summary["eval/match/final_score_mean"] == 26000.0
    assert summary["eval/match/flying_rate"] == 0.0
    assert summary["eval/match/length_kyokus_mean"] == 10.0
    assert summary["eval/match/discard_count_mean"] == 460 / 3


def test_match_length_metrics_support_physical_self_play_hanchans() -> None:
    metrics = SemanticMetrics()
    metrics.record_match_length(8)
    metrics.record_match_length(12)

    summary = metrics.summary("train")
    assert summary["train/match/count"] == 2
    assert summary["train/match/length_kyokus_mean"] == 10.0
    for name in (
        "first_place_rate", "top2_rate", "last_place_rate", "mean_rank",
        "final_score_mean", "flying_rate",
    ):
        assert f"train/match/{name}" not in summary


def test_ppo_buffer_and_evaluation_matrix_metrics() -> None:
    transition = Transition(
        actor_factors=np.zeros((1, 32), np.int32),
        actor_numeric=np.zeros((1, 8), np.float32),
        actor_length=1,
        query_rows=np.zeros((2, 15), np.int32),
        query_action_ids=np.zeros(1, np.int32),
        query_pair_counts=1,
        legal_mask=np.ones(241, np.bool_),
        action=0,
        logprob=0.0,
        value=0.0,
    )
    transition.value, transition.advantage = 0.25, 1.0
    result = ppo_buffer_metrics([transition])
    assert result["buffer/advantage_mean"] == 1.0
    assert result["buffer/advantage_std"] == 0.0
    assert result["buffer/value_mean"] == 0.25
    assert result["buffer/value_std"] == 0.0
