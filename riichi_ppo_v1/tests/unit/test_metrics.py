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
    metrics.record_kyoku([0, 2], [8000, 3000, -11000, 0], events)
    summary = metrics.summary()
    assert summary["train/kyoku/count"] == 2
    assert summary["train/kyoku/win_rate"] == 0.5
    assert summary["train/kyoku/deal_in_rate"] == 0.5
    assert summary["train/kyoku/win_points_mean"] == 8.0
    assert summary["train/kyoku/deal_in_points_mean"] == -11.0

    draw = SemanticMetrics()
    draw.record_kyoku([0], [1500, -500, -500, -500], [[json.dumps({"type": "ryukyoku", "deltas": [1500, -500, -500, -500]})], [], [], []])
    assert draw.summary()["train/kyoku/draw_rate"] == 1.0


def test_metrics_jsonl_has_schema_and_resume_counters(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    append_metric_jsonl(path, update=3, global_decisions=10, global_kyokus=4, source="train", metrics={"train/x": 1.0})
    row = json.loads(path.read_text())
    assert row["schema_version"] == 1 and row["metrics"] == {"train/x": 1.0}
    assert metric_counters(path) == (10, 4)


def test_ppo_buffer_and_evaluation_matrix_metrics() -> None:
    transition = Transition(np.zeros((1, 10), np.uint8), np.zeros((1, 8), np.float32), 1,
                            np.ones(241, np.bool_), 0, 0.0, 1.0)
    transition.return_, transition.advantage = 2.0, 1.0
    result = ppo_buffer_metrics([transition])
    assert result["buffer/return_mean"] == 2.0
    assert "explained_variance" in result
    cases = evaluation_cases(10, 12)
    assert len(cases) == 48 and {seat for _seed, seat, _recipe in cases} == {0, 1, 2, 3}
    merged = merge_evaluation_summaries([
        {"eval/match/count": 1.0, "eval/match/rank_mean": 1.0, "eval/kyoku/count": 2.0},
        {"eval/match/count": 1.0, "eval/match/rank_mean": 3.0, "eval/kyoku/count": 2.0},
    ])
    assert merged["eval/match/rank_mean"] == 2.0
    assert merged["eval/match/rank_mean_stderr"] == 1.0
