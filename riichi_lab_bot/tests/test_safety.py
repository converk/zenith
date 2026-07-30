from __future__ import annotations

import json

import numpy as np

from riichi_lab_bot.bridge import EventContext, PreparedDecision
from riichi_lab_bot.safety import (
    action_to_response,
    choose_safe_response,
    possible_action_matches,
)


class FakeAction:
    def __init__(self, payload: dict, tile: int | None = None) -> None:
        self.payload = payload
        self.tile = tile

    def to_mjai(self) -> str:
        return json.dumps(self.payload)


class FakeObservation:
    player_id = 1
    drawn_tile = 16

    def __init__(self, actions: list[FakeAction]) -> None:
        self.actions = actions

    def legal_actions(self) -> list[FakeAction]:
        return self.actions

    def select_action_from_mjai(self, response):
        for action in self.actions:
            value = action.payload
            if value.get("type") != response.get("type"):
                continue
            if "pai" in value and value["pai"] != response.get("pai"):
                continue
            return action
        return None


def test_possible_action_ignores_response_only_fields() -> None:
    response = {
        "type": "chi",
        "actor": 1,
        "target": 0,
        "pai": "5m",
        "consumed": ["4m", "6m"],
        "request_id": 42,
    }
    possible = {
        "type": "chi",
        "pai": "5m",
        "consumed": ["6m", "4m"],
    }
    assert possible_action_matches(response, possible)


def test_discard_tsumogiri_is_checked_when_server_provides_it() -> None:
    response = {
        "type": "dahai",
        "pai": "5mr",
        "tsumogiri": True,
    }
    assert possible_action_matches(
        response, {"type": "dahai", "pai": "5mr"}
    )
    assert possible_action_matches(
        response,
        {"type": "dahai", "pai": "5mr", "tsumogiri": True},
    )
    assert not possible_action_matches(
        response,
        {"type": "dahai", "pai": "5mr", "tsumogiri": False},
    )


def test_red_five_and_consumed_tiles_remain_distinct() -> None:
    assert not possible_action_matches(
        {"type": "dahai", "pai": "5mr"},
        {"type": "dahai", "pai": "5m"},
    )
    assert not possible_action_matches(
        {"type": "pon", "pai": "5m", "consumed": ["5m", "5mr"]},
        {"type": "pon", "pai": "5m", "consumed": ["5m", "5m"]},
    )


def test_hora_server_details_are_not_required_in_response() -> None:
    assert possible_action_matches(
        {"type": "hora", "actor": 0, "request_id": 7},
        {"type": "hora", "actor": 0, "target": 2, "pai": "5p"},
    )


def test_action_response_enrichment_for_all_action_families() -> None:
    obs = FakeObservation([])
    discard = action_to_response(
        FakeAction({"type": "dahai", "actor": 1, "pai": "5mr"}, 16),
        obs,
        EventContext("tsumo", 1, "5mr"),
        1,
    )
    assert discard["tsumogiri"] is True
    assert discard["request_id"] == 1

    for action_type in ("chi", "pon", "daiminkan"):
        response = action_to_response(
            FakeAction(
                {
                    "type": action_type,
                    "actor": 1,
                    "pai": "5m",
                    "consumed": ["4m", "6m"],
                }
            ),
            obs,
            EventContext("dahai", 0, "5m"),
            2,
        )
        assert response["target"] == 0

    ron = action_to_response(
        FakeAction({"type": "hora", "actor": 1}),
        obs,
        EventContext("dahai", 3, "C"),
        3,
    )
    assert ron["target"] == 3
    assert ron["pai"] == "C"

    tsumo = action_to_response(
        FakeAction({"type": "hora", "actor": 1}),
        obs,
        EventContext("tsumo", 1, "C"),
        4,
    )
    assert "target" not in tsumo

    for action_type in (
        "reach",
        "ankan",
        "kakan",
        "ryukyoku",
        "none",
    ):
        response = action_to_response(
            FakeAction({"type": action_type, "actor": 1}),
            obs,
            EventContext(),
            5,
        )
        assert response["type"] == action_type
        assert response["request_id"] == 5


def test_safe_fallback_prefers_pass_over_unmatched_model_action() -> None:
    primary = FakeAction({"type": "pon", "actor": 1, "pai": "5m"})
    pass_action = FakeAction({"type": "none"})
    observation = FakeObservation([primary, pass_action])
    prepared = PreparedDecision(
        observation=observation,
        seat=1,
        token_factors=np.zeros((0, 10), dtype=np.uint8),
        token_numeric=np.zeros((0, 8), dtype=np.float32),
        token_length=0,
        legal_mask=np.ones(241, dtype=bool),
        legal_jsons=(),
        event_context=EventContext("dahai", 0, "5m"),
    )
    result = choose_safe_response(
        prepared,
        primary,
        [{"type": "none"}],
        99,
    )
    assert result.source == "fallback"
    assert result.payload == {"type": "none", "request_id": 99}
