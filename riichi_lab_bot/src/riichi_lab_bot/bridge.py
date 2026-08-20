"""单席位在线 Observation 到 V16 Actor 输入的桥接。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from riichi_ppo_v1.model.bridge import (
    NUM_PLAYERS,
    BatchedStateBridge,
    Decision,
    action_jsons_and_decision_flag,
)
from riichi_ppo_v1.model.semantic_validation import (
    assert_v16_actor_input_semantics,
)

from .observation import ObservationView, ThreatSnapshotTracker

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

_ALWAYS_REBUILT_FIELDS = frozenset(
    {
        "riichi_declared",
        "riichi_accepted",
        "riichi_declaration_indices",
        "riichi_sutehais",
        "last_tedashis",
        "tsumogiri_flags",
        "tiles_left",
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
    history_factors: np.ndarray
    history_numeric: np.ndarray
    history_length: int
    snapshot_kinds: np.ndarray
    snapshot_cat: np.ndarray
    snapshot_num: np.ndarray
    snapshot_length: int
    query_rows: np.ndarray
    query_action_ids: np.ndarray
    query_pair_count: int
    legal_mask: np.ndarray
    legal_jsons: tuple[str, ...]
    event_context: EventContext


def _parse_event_context(events: list[str]) -> EventContext:
    context = EventContext()
    for raw in events:
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        event_type = value.get("type")
        if event_type in {"tsumo", "dahai", "kakan", "ankan"}:
            actor = value.get("actor")
            context = EventContext(
                str(event_type),
                int(actor) if isinstance(actor, int) else None,
                value.get("pai") if isinstance(value.get("pai"), str) else None,
            )
    return context


def _accepted_model_events(raw_events: list[str]) -> tuple[list[str], bool]:
    accepted: list[str] = []
    saw_start_kyoku = False
    for raw in raw_events:
        try:
            event_type = json.loads(raw).get("type")
        except (TypeError, json.JSONDecodeError):
            continue
        if event_type in _MODEL_EVENT_TYPES:
            accepted.append(raw)
            saw_start_kyoku |= event_type == "start_kyoku"
    return accepted, saw_start_kyoku


class OnlineStateBridge:
    """一个连接、一个 bot 席位的有状态 V16 桥接器。"""

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
        self.training_bridge = BatchedStateBridge(self.manager, 1)
        self.event_context = EventContext()
        self.threats = ThreatSnapshotTracker(self.seat)

    def prepare(self, observation: Any) -> PreparedDecision:
        if int(observation.player_id) != self.seat:
            raise ValueError(
                "Observation seat mismatch: "
                f"expected {self.seat}, got {observation.player_id}"
            )
        if not list(observation.legal_actions()):
            raise ValueError("request_action observation has no legal actions")

        accepted_events, saw_start_kyoku = _accepted_model_events(
            list(observation.new_events())
        )
        if saw_start_kyoku:
            self.event_context = EventContext()
        self.threats.apply_events(accepted_events)
        self.threats.refine_drawn_tile(observation)
        new_context = _parse_event_context(accepted_events)
        if new_context.last_type is not None:
            self.event_context = new_context
        self.threats.apply_current_response_window(
            observation,
            last_type=self.event_context.last_type,
            actor=self.event_context.actor,
            pai=self.event_context.pai,
        )
        observation = self._with_rebuilt_fields(observation)

        events_by_player = [
            accepted_events if player == self.seat else []
            for player in range(NUM_PLAYERS)
        ]
        self.manager.apply_events_batch([0], [events_by_player])

        legal_jsons, _decision_flag = action_jsons_and_decision_flag(
            observation
        )
        batch = self.training_bridge.prepare_v16(
            [Decision(0, self.seat, observation)]
        )
        history_length = int(batch.history_lengths[0])
        snapshot_length = int(batch.snapshot_lengths[0])
        query_pair_count = int(batch.query_pair_counts[0])
        assert_v16_actor_input_semantics(
            batch.history_factors,
            batch.history_numeric,
            batch.history_lengths,
            batch.snapshot_kinds,
            batch.snapshot_cat,
            batch.snapshot_num,
            batch.snapshot_lengths,
            batch.query_rows,
            batch.query_action_ids,
            batch.query_pair_counts,
            batch.legal_mask,
        )
        return PreparedDecision(
            observation=observation,
            seat=self.seat,
            history_factors=batch.history_factors[0, :history_length].copy(),
            history_numeric=batch.history_numeric[0, :history_length].copy(),
            history_length=history_length,
            snapshot_kinds=batch.snapshot_kinds[0, :snapshot_length].copy(),
            snapshot_cat=batch.snapshot_cat[0, :snapshot_length].copy(),
            snapshot_num=batch.snapshot_num[0, :snapshot_length].copy(),
            snapshot_length=snapshot_length,
            query_rows=batch.query_rows[0, : 2 * query_pair_count].copy(),
            query_action_ids=batch.query_action_ids[
                0, :query_pair_count
            ].copy(),
            query_pair_count=query_pair_count,
            legal_mask=batch.legal_mask[0].copy(),
            legal_jsons=tuple(legal_jsons),
            event_context=self.event_context,
        )

    def _with_rebuilt_fields(self, observation: Any) -> Any:
        rebuilt = self.threats.fields()
        if isinstance(observation, ObservationView):
            missing = set(observation.missing_fields)
            observation.set_fields(
                {
                    name: value
                    for name, value in rebuilt.items()
                    if name in missing or name in _ALWAYS_REBUILT_FIELDS
                }
            )
            return observation

        overrides = {
            name: value
            for name, value in rebuilt.items()
            if name in _ALWAYS_REBUILT_FIELDS
            and getattr(observation, name, None) != value
        }
        if overrides:
            return ObservationView(observation, overrides)
        return observation

    def record_response(
        self, prepared: PreparedDecision, response: dict[str, Any] | None,
    ) -> None:
        self.threats.record_response(
            prepared.observation,
            last_type=prepared.event_context.last_type,
            actor=prepared.event_context.actor,
            legal_jsons=prepared.legal_jsons,
            response=response,
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
