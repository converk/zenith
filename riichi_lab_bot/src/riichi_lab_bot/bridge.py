"""Single-seat RiichiEnv observation to semantic-token bridge."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from typing import Any

import numpy as np

from .features import encode_public_summary
from .model import NUM_ACTIONS, NUMERIC_WIDTH, TOKEN_WIDTH

NUM_PLAYERS = 4
_DECISION_ACTION_TYPES = frozenset(
    {"dahai", "reach", "ankan", "kakan", "ryukyoku"}
)
_MODEL_EVENT_TYPES = frozenset(
    {
        "start_game",
        "start_kyoku",
        "tsumo",
        "dahai",
        "chi",
        "pon",
        "daiminkan",
        "ankan",
        "kakan",
        "dora",
        "reach",
        "reach_accepted",
        "hora",
        "ryukyoku",
        "end_kyoku",
        "end_game",
    }
)


@dataclass(frozen=True)
class EventContext:
    last_type: str | None = None
    actor: int | None = None
    pai: str | None = None


@dataclass(frozen=True)
class PreparedDecision:
    observation: Any
    seat: int
    token_factors: np.ndarray
    token_numeric: np.ndarray
    token_length: int
    legal_mask: np.ndarray
    legal_jsons: tuple[str, ...]
    event_context: EventContext


def tile_id_to_mjai(tile_id: int | None) -> str | None:
    if tile_id is None:
        return None
    tile = int(tile_id)
    red = {16: "5mr", 52: "5pr", 88: "5sr"}
    if tile in red:
        return red[tile]
    if not 0 <= tile < 136:
        raise ValueError(f"invalid RiichiEnv tile id {tile}")
    suit = tile // 36
    if suit < 3:
        return f"{tile % 36 // 4 + 1}{('m', 'p', 's')[suit]}"
    honors = ("E", "S", "W", "N", "P", "F", "C")
    return honors[(tile - 108) // 4]


@lru_cache(maxsize=1024)
def _normalized_action_json(
    raw_action: str, tsumogiri: bool
) -> tuple[str, str]:
    value = json.loads(raw_action)
    action_type = str(value.get("type", ""))
    if action_type == "dahai":
        value["tsumogiri"] = bool(tsumogiri)
    expected_consumed = {
        "chi": 2,
        "pon": 2,
        "daiminkan": 3,
    }.get(action_type)
    consumed = value.get("consumed")
    if (
        expected_consumed is not None
        and isinstance(consumed, list)
        and len(consumed) == expected_consumed + 1
    ):
        called = value.get("pai")
        try:
            consumed.remove(called)
        except ValueError as exc:
            raise ValueError(
                f"{action_type} replay action lacks its called tile"
            ) from exc
        value["consumed"] = consumed
    return (
        json.dumps(value, separators=(",", ":"), sort_keys=True),
        action_type,
    )


def action_jsons_and_flag(observation: Any) -> tuple[list[str], int]:
    drawn = getattr(observation, "drawn_tile", None)
    result: list[str] = []
    has_decision_action = False
    for action in observation.legal_actions():
        tsumogiri = (
            action.tile is not None
            and drawn is not None
            and int(action.tile) == int(drawn)
        )
        action_json, action_type = _normalized_action_json(
            action.to_mjai(), tsumogiri
        )
        result.append(action_json)
        has_decision_action |= action_type in _DECISION_ACTION_TYPES
    return result, int(has_decision_action)


def snapshot_json(observation: Any, decision_flags: int) -> str:
    pid = int(observation.player_id)
    hands = getattr(observation, "hands", None)
    if hands is None:
        raise RuntimeError("Observation must expose hands")
    data = {
        "player_id": pid,
        "oya": int(observation.oya),
        "round_wind": int(observation.round_wind),
        "kyoku_index": int(observation.kyoku_index),
        "honba": int(observation.honba),
        "riichi_sticks": int(observation.riichi_sticks),
        "scores": [int(x) for x in observation.scores],
        "dora_indicators": [
            tile_id_to_mjai(x) for x in observation.dora_indicators
        ],
        "hand": [tile_id_to_mjai(x) for x in hands[pid]],
        "drawn_tile": tile_id_to_mjai(
            getattr(observation, "drawn_tile", None)
        ),
        "riichi_declared": [
            bool(x) for x in observation.riichi_declared
        ],
        "decision_flags": int(decision_flags),
    }
    return json.dumps(data, separators=(",", ":"))


def _parse_event_context(events: list[str]) -> EventContext:
    context = EventContext()
    for raw in events:
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        event_type = value.get("type")
        if event_type in {"tsumo", "dahai"}:
            actor = value.get("actor")
            context = EventContext(
                str(event_type),
                int(actor) if isinstance(actor, int) else None,
                value.get("pai") if isinstance(value.get("pai"), str) else None,
            )
    return context


class OnlineStateBridge:
    """Stateful bridge for exactly one bot seat in one connection."""

    def __init__(self, seat: int) -> None:
        if not 0 <= int(seat) < NUM_PLAYERS:
            raise ValueError("4-player seat must be in [0, 3]")
        try:
            import riichi
        except ImportError as exc:
            raise RuntimeError(
                "the local riichi native extension is not installed"
            ) from exc
        self.seat = int(seat)
        self.manager = riichi.MjaiKyokuStateMachineManager(1)
        self.event_context = EventContext()

    def prepare(self, observation: Any) -> PreparedDecision:
        if int(observation.player_id) != self.seat:
            raise ValueError(
                "Observation seat mismatch: "
                f"expected {self.seat}, got {observation.player_id}"
            )
        legal_actions = list(observation.legal_actions())
        if not legal_actions:
            raise ValueError("request_action observation has no legal actions")

        raw_events = list(observation.new_events())
        accepted_events: list[str] = []
        for raw in raw_events:
            try:
                event_type = json.loads(raw).get("type")
            except (TypeError, json.JSONDecodeError):
                continue
            if event_type in _MODEL_EVENT_TYPES:
                accepted_events.append(raw)
        new_context = _parse_event_context(accepted_events)
        if new_context.last_type is not None:
            self.event_context = new_context

        events_by_player = [
            accepted_events if player == self.seat else []
            for player in range(NUM_PLAYERS)
        ]
        self.manager.apply_events_batch([0], [events_by_player])

        legal_jsons, decision_flag = action_jsons_and_flag(observation)
        factors, numeric, lengths, mask, _generation = (
            self.manager.prepare_decisions(
                [self.seat],
                [legal_jsons],
                [snapshot_json(observation, decision_flag)],
            )
        )
        factors_a = np.asarray(factors, dtype=np.uint8)
        numeric_a = np.asarray(numeric, dtype=np.float32)
        lengths_a = np.asarray(lengths, dtype=np.int64)
        mask_a = np.asarray(mask, dtype=np.bool_)
        if factors_a.ndim != 3 or factors_a.shape[0] != 1:
            raise RuntimeError(
                f"state machine returned malformed factors {factors_a.shape}"
            )
        if factors_a.shape[2] != TOKEN_WIDTH:
            raise RuntimeError("state machine returned wrong token width")
        if numeric_a.shape != (*factors_a.shape[:2], NUMERIC_WIDTH):
            raise RuntimeError("state machine returned malformed numeric data")
        if lengths_a.shape != (1,) or mask_a.shape != (1, NUM_ACTIONS):
            raise RuntimeError("state machine returned malformed metadata")
        if not mask_a[0].any():
            raise RuntimeError("state machine returned an empty legal mask")

        base_length = int(lengths_a[0])
        public = encode_public_summary(observation, self.seat)
        total_length = base_length + len(public)
        if total_length + 1 > 4096:
            raise RuntimeError(
                f"public token context overflow: {total_length + 1} > 4096"
            )
        output_factors = np.zeros(
            (total_length, TOKEN_WIDTH), dtype=np.uint8
        )
        output_numeric = np.zeros(
            (total_length, NUMERIC_WIDTH), dtype=np.float32
        )
        output_factors[:base_length] = factors_a[0, :base_length]
        output_numeric[:base_length] = numeric_a[0, :base_length]
        if len(public):
            output_factors[base_length:] = public
        return PreparedDecision(
            observation=observation,
            seat=self.seat,
            token_factors=output_factors,
            token_numeric=output_numeric,
            token_length=total_length,
            legal_mask=mask_a[0].copy(),
            legal_jsons=tuple(legal_jsons),
            event_context=self.event_context,
        )

    def decode(self, prepared: PreparedDecision, action_id: int) -> Any:
        if not prepared.legal_mask[int(action_id)]:
            raise ValueError(f"model selected masked action id {action_id}")
        mjai = self.manager.decode_actions(
            [self.seat], [int(action_id)]
        )[0]
        action = prepared.observation.select_action_from_mjai(mjai)
        if action is None:
            raise RuntimeError(
                f"RiichiEnv rejected decoded action id={action_id}: {mjai}"
            )
        return action

