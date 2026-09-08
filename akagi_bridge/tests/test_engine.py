"""mjai 事件流 → Observation → V18 决策 的适配层测试。

需要本地已安装 `riichienv` / `riichi` 原生扩展与 torch;缺失时整文件跳过。
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("riichienv")
pytest.importorskip("riichi")

import numpy as np  # noqa: E402

from akagi_bridge.engine import (  # noqa: E402
    AkagiDecisionEngine,
    AkagiProtocolError,
    build_observation,
)
from riichi_lab_bot.policy import Candidate, InferenceResult  # noqa: E402

HAND = [
    "1m", "1m", "1m", "2m", "3m", "4m", "5m",
    "6m", "7m", "8m", "9m", "9m", "9m",
]


def start_kyoku() -> dict:
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyoutaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "dora_marker": "1p",
        # 对手手牌按 Akagi 的约定隐藏为 "?"
        "tehais": [HAND, ["?"] * 13, ["?"] * 13, ["?"] * 13],
    }


def dealer_events() -> list[dict]:
    return [
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        start_kyoku(),
        {"type": "tsumo", "actor": 0, "pai": "5m"},
    ]


class FirstLegalPolicy:
    """取合法掩码里的第一个动作,并按 topk 给出均匀候选。"""

    def infer(self, prepared, *, topk: int = 0) -> InferenceResult:
        ids = [int(value) for value in np.flatnonzero(prepared.legal_mask)]
        candidates: tuple[Candidate, ...] = ()
        if topk > 0:
            width = min(int(topk), len(ids))
            candidates = tuple(
                Candidate(action_id, 1.0 / width) for action_id in ids[:width]
            )
        return InferenceResult(ids[0], 0.1, candidates)


def test_build_observation_exposes_pending_decision() -> None:
    observation = build_observation(dealer_events(), 0)
    assert int(observation.player_id) == 0
    actions = list(observation.legal_actions())
    assert actions, "dealer must have a discard decision after the opening draw"
    events = list(observation.new_events())
    # 全新 Observation 的 new_events() 必须是本局完整事件流,
    # OnlineStateBridge.prepare 依赖这一点重建状态机。
    assert json.loads(events[0])["type"] == "start_game"
    assert any(json.loads(raw)["type"] == "start_kyoku" for raw in events)


def test_build_observation_accepts_json_strings() -> None:
    events = [json.dumps(event) for event in dealer_events()]
    observation = build_observation(events, 0)
    assert list(observation.legal_actions())


def test_build_observation_rejects_stream_without_decision() -> None:
    with pytest.raises(AkagiProtocolError, match="no pending decision"):
        build_observation([{"type": "start_game", "names": []}], 0)


def test_build_observation_rejects_bad_events() -> None:
    with pytest.raises(AkagiProtocolError, match=r"events\[1\]"):
        build_observation([{"type": "start_game"}, {"type": "???"}], 0)
    with pytest.raises(AkagiProtocolError, match="must be a JSON object"):
        build_observation([{"type": "start_game"}, 42], 0)  # type: ignore[list-item]


def test_build_observation_rejects_bad_seat() -> None:
    with pytest.raises(AkagiProtocolError, match="player_id"):
        build_observation(dealer_events(), 4)


def test_engine_react_returns_mjai_action_and_candidates() -> None:
    engine = AkagiDecisionEngine(FirstLegalPolicy(), topk=3)
    decision = engine.react(dealer_events(), 0)
    assert decision.reaction is not None
    assert decision.reaction["type"] in {"dahai", "reach"}
    assert decision.reaction["actor"] == 0
    assert decision.candidates
    assert all(candidate.prob > 0.0 for candidate in decision.candidates)
    assert decision.elapsed_ms >= 0.0


def test_engine_react_for_other_seat_has_no_decision() -> None:
    engine = AkagiDecisionEngine(FirstLegalPolicy(), topk=3)
    with pytest.raises(AkagiProtocolError, match="no pending decision"):
        engine.react(dealer_events(), 1)
