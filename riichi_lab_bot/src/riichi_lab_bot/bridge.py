"""单席位在线 Observation 到 V19 Actor 输入的桥接。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from riichi_ppo_v1.model.bridge import (
    NUM_PLAYERS,
    action_jsons_and_decision_flag,
)
from riichi_ppo_v1.model.current_state import encode_batch
from riichi_ppo_v1.model.semantic_validation import (
    assert_actor_input_semantics,
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
    """单次决策的 V19 Actor 输入(只保留有效行,不含 padding)。

    与训练/评测共用 ``riichi_ppo_v1.model.current_state.encode_batch`` 编码路径;
    critic 私有行不参与在线推理。
    """

    observation: Any
    seat: int
    actor_factors: np.ndarray
    actor_numeric: np.ndarray
    actor_length: int
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
    """一个连接、一个 bot 席位的有状态 V19 桥接器。"""

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

        # 状态机仅需推进 own-seat 事件流;管理器负责合法掩码与动作解码。
        events_by_player = [
            accepted_events if player == self.seat else []
            for player in range(NUM_PLAYERS)
        ]
        self.manager.apply_events_batch([0], [events_by_player])

        # 与训练 rollout 同一条编码路径:action 对象→action id 映射→
        # current_state.encode_batch 整批装配(无 critic 私有行)。
        legal_jsons, _decision_flag = action_jsons_and_decision_flag(
            observation
        )
        legal_objects = list(observation.legal_actions())
        mask = np.asarray(
            self.manager.prepare_decisions([self.seat], [legal_jsons]),
            dtype=np.bool_,
        )[0]
        index_rows = self.manager.action_ids_with_source_indices([self.seat])
        actions_by_id: list[tuple[Any, int]] = []
        for raw_action_id, raw_source_index in index_rows[0]:
            action_id = int(raw_action_id)
            source_index = int(raw_source_index)
            if not 0 <= source_index < len(legal_objects):
                raise RuntimeError(
                    "state machine returned invalid legal action index "
                    f"{source_index}"
                )
            actions_by_id.append((legal_objects[source_index], action_id))
        expected_ids = np.flatnonzero(mask).tolist()
        if [action_id for _action, action_id in actions_by_id] != expected_ids:
            raise RuntimeError(
                "state machine action-id mapping disagrees with legal mask"
            )
        batch = encode_batch([(observation, actions_by_id)])

        actor_length = int(batch.actor_lengths[0])
        query_pair_count = int(batch.query_pair_counts[0])
        # 与 PPO rollout 同约定:当前局面路径不产生 query_rows,传 None
        # 跳过逐 query 行一致性校验,其余结构/域/action 集合校验照常生效。
        assert_actor_input_semantics(
            batch.actor_factors,
            batch.actor_numeric,
            batch.actor_lengths,
            None,
            batch.action_ids,
            batch.query_pair_counts,
            batch.legal_mask,
        )
        return PreparedDecision(
            observation=observation,
            seat=self.seat,
            actor_factors=batch.actor_factors[0, :actor_length].copy(),
            actor_numeric=batch.actor_numeric[0, :actor_length].copy(),
            actor_length=actor_length,
            query_action_ids=batch.action_ids[
                0, :query_pair_count
            ].copy(),
            query_pair_count=query_pair_count,
            legal_mask=batch.legal_mask[0].copy(),
            legal_jsons=tuple(legal_jsons),
            event_context=self.event_context,
        )

    def _with_rebuilt_fields(self, observation: Any) -> Any:
        rebuilt = self.threats.fields()
        discards = getattr(observation, "discards", ((), (), (), ()))
        rebuilt_flags = rebuilt["tsumogiri_flags"]
        if all(
            len(discard_row) == len(flag_row)
            for discard_row, flag_row in zip(discards, rebuilt_flags, strict=True)
        ):
            rebuilt["last_tedashis"] = [
                next(
                    (
                        int(tile)
                        for tile, tsumogiri in zip(
                            reversed(discard_row), reversed(flag_row), strict=True,
                        )
                        if not tsumogiri
                    ),
                    None,
                )
                for discard_row, flag_row in zip(discards, rebuilt_flags, strict=True)
            ]
        if isinstance(observation, ObservationView):
            missing = set(observation.missing_fields)
            flags = getattr(observation, "tsumogiri_flags", ((), (), (), ()))
            malformed_flags = any(
                len(discard_row) != len(flag_row)
                for discard_row, flag_row in zip(discards, flags, strict=True)
            )
            observation.set_fields(
                {
                    name: value
                    for name, value in rebuilt.items()
                    if name in missing
                    or name in _ALWAYS_REBUILT_FIELDS
                    or (name == "tsumogiri_flags" and malformed_flags)
                }
            )
            return observation

        flags = getattr(observation, "tsumogiri_flags", ((), (), (), ()))
        malformed_flags = any(
            len(discard_row) != len(flag_row)
            for discard_row, flag_row in zip(discards, flags, strict=True)
        )
        overrides = {
            name: value
            for name, value in rebuilt.items()
            if (name in _ALWAYS_REBUILT_FIELDS or (name == "tsumogiri_flags" and malformed_flags))
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
