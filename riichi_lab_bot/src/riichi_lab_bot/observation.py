"""RiichiLab 在线 Observation 归一化与公开状态重建。"""

from __future__ import annotations

import base64
import json
from typing import Any

NUM_PLAYERS = 4

_RED_PHYSICAL = {"5mr": 16, "5pr": 52, "5sr": 88}
_HONOR_PHYSICAL = {
    "E": 108, "S": 112, "W": 116, "N": 120,
    "P": 124, "F": 128, "C": 132,
}


def mjai_to_physical_id(pai: str | None) -> int | None:
    """把 MJAI 牌字符串映射为 RiichiEnv 的 136 牌物理编号。"""
    if not pai or pai == "?":
        return None
    if pai in _RED_PHYSICAL:
        return _RED_PHYSICAL[pai]
    if len(pai) == 2 and pai[1] in "mps":
        suit = "mps".index(pai[1])
        rank = int(pai[0])
        if 1 <= rank <= 9:
            return suit * 36 + (rank - 1) * 4
        return None
    if pai in _HONOR_PHYSICAL:
        return _HONOR_PHYSICAL[pai]
    return None


class ThreatSnapshotTracker:
    """累积线上 payload 可能缺失的本局公开/自家状态事实。"""

    def __init__(self, seat: int) -> None:
        self.seat = int(seat)
        self.reset_kyoku()

    def reset_kyoku(self) -> None:
        self.riichi_declared = [False] * NUM_PLAYERS
        self.riichi_accepted = [False] * NUM_PLAYERS
        self.riichi_declaration_indices = [None] * NUM_PLAYERS
        self.last_tedashis = [None] * NUM_PLAYERS
        self.tsumogiri_flags: list[list[bool]] = [
            [] for _ in range(NUM_PLAYERS)
        ]
        self.riichi_sutehais = [None] * NUM_PLAYERS
        self.discard_counts = [0] * NUM_PLAYERS
        self.tsumo_count = 0
        self.missed_agari_doujun = False
        self.missed_agari_riichi = False
        self.drawn_tile = None
        self.drawn_pai = None

    def apply_event(self, raw: Any) -> None:
        try:
            event = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(event, dict):
            return
        kind = str(event.get("type", ""))
        if kind == "start_kyoku":
            self.reset_kyoku()
            return
        actor_raw = event.get("actor")
        actor = int(actor_raw) if isinstance(actor_raw, int) else -1
        if not 0 <= actor < NUM_PLAYERS:
            return
        if kind == "tsumo":
            self.tsumo_count += 1
            if actor == self.seat:
                pai = event.get("pai")
                self.drawn_pai = pai if isinstance(pai, str) else None
                self.drawn_tile = mjai_to_physical_id(self.drawn_pai)
        elif kind == "reach":
            self.riichi_declared[actor] = True
            self.riichi_declaration_indices[actor] = self.discard_counts[actor]
        elif kind == "reach_accepted":
            self.riichi_accepted[actor] = True
        elif kind in {"chi", "pon", "daiminkan"} and actor == self.seat:
            self.missed_agari_doujun = False
            self.drawn_tile = None
            self.drawn_pai = None
        elif kind == "dahai":
            pai = event.get("pai")
            tsumogiri = bool(event.get("tsumogiri", False))
            self.tsumogiri_flags[actor].append(tsumogiri)
            physical = mjai_to_physical_id(pai)
            if not tsumogiri and physical is not None:
                self.last_tedashis[actor] = physical
            if (
                self.riichi_declared[actor]
                and self.riichi_sutehais[actor] is None
                and physical is not None
            ):
                self.riichi_sutehais[actor] = physical
            self.discard_counts[actor] += 1
            if actor == self.seat:
                self.missed_agari_doujun = False
                self.drawn_tile = None
                self.drawn_pai = None

    def apply_events(self, events: list[Any]) -> None:
        for raw in events:
            self.apply_event(raw)

    def refine_drawn_tile(self, observation: Any) -> None:
        """用当前自家手牌把 MJAI 摸牌补成精确物理牌。"""
        if self.drawn_tile is None:
            return
        hand = getattr(observation, "hands", None)
        if hand is None:
            return
        candidates = [
            int(tile)
            for tile in hand[self.seat]
            if _physical_matches_pai(int(tile), self.drawn_pai)
        ]
        if candidates:
            self.drawn_tile = candidates[-1]

    def apply_current_response_window(
        self,
        observation: Any,
        *,
        last_type: str | None,
        actor: int | None,
        pai: str | None,
    ) -> None:
        """按当前可见窗口补齐内核在发起响应前设置的见逃标记。"""
        if actor is None or actor == self.seat:
            return
        if last_type not in {"dahai", "kakan", "ankan"}:
            return
        if self.missed_agari_doujun or self.missed_agari_riichi:
            return
        if _has_legal_hora(observation):
            return
        win_tile = mjai_to_physical_id(pai)
        if win_tile is None:
            return
        own_discards = getattr(observation, "discards", ([], [], [], []))
        if any(int(tile) // 4 == win_tile // 4 for tile in own_discards[self.seat]):
            return
        if _has_win_shape_without_legal_ron(observation, self.seat, win_tile):
            self.missed_agari_doujun = True

    def record_response(
        self,
        observation: Any,
        *,
        last_type: str | None,
        actor: int | None,
        legal_jsons: tuple[str, ...],
        response: dict[str, Any] | None,
    ) -> None:
        """记录本 bot 已发送的动作,补齐线上不会回传的 pass 语义。"""
        response_type = response.get("type") if isinstance(response, dict) else None
        if (
            actor is not None
            and actor != self.seat
            and last_type in {"dahai", "kakan", "ankan"}
            and response_type != "hora"
            and any(_json_type(value) == "hora" for value in legal_jsons)
        ):
            self.missed_agari_doujun = True
            declared = getattr(observation, "riichi_declared", (False,) * NUM_PLAYERS)
            if bool(declared[self.seat]):
                self.missed_agari_riichi = True
        if response_type in {"chi", "pon", "daiminkan", "dahai"}:
            self.missed_agari_doujun = False

    def fields(self) -> dict[str, Any]:
        declared = [
            bool(self.riichi_declared[seat] and self.riichi_sutehais[seat] is not None)
            or bool(self.riichi_accepted[seat])
            for seat in range(NUM_PLAYERS)
        ]
        return {
            "riichi_declared": declared,
            "riichi_accepted": list(self.riichi_accepted),
            "riichi_declaration_indices": list(
                self.riichi_declaration_indices
            ),
            "last_tedashis": list(self.last_tedashis),
            "tsumogiri_flags": [list(row) for row in self.tsumogiri_flags],
            "riichi_sutehais": list(self.riichi_sutehais),
            "missed_agari_doujun": self.missed_agari_doujun,
            "missed_agari_riichi": self.missed_agari_riichi,
            "tiles_left": max(70 - self.tsumo_count, 0),
            "waits": [],
            "is_tenpai": False,
            "drawn_tile": self.drawn_tile,
        }


def _json_type(value: str) -> str | None:
    try:
        payload = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    action_type = payload.get("type")
    return action_type if isinstance(action_type, str) else None


def _physical_matches_pai(tile: int, pai: str | None) -> bool:
    reference = mjai_to_physical_id(pai)
    if reference is None:
        return False
    if pai in _RED_PHYSICAL:
        return int(tile) == reference
    return int(tile) // 4 == reference // 4


def _has_legal_hora(observation: Any) -> bool:
    for action in observation.legal_actions():
        try:
            payload = json.loads(action.to_mjai())
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("type") == "hora":
            return True
    return False


def _has_win_shape_without_legal_ron(
    observation: Any, seat: int, win_tile: int,
) -> bool:
    """复刻 RiichiEnv 对“有和形但当前不能荣和”的同巡见逃判定。"""
    try:
        from riichienv import Conditions, HandEvaluator

        declared = getattr(observation, "riichi_declared", (False,) * NUM_PLAYERS)
        conditions = Conditions(
            tsumo=False,
            riichi=bool(declared[seat]),
            player_wind=(seat - int(getattr(observation, "oya", 0))) % NUM_PLAYERS,
            round_wind=int(getattr(observation, "round_wind", 0)),
            houtei=int(getattr(observation, "tiles_left", 1)) == 0,
            riichi_sticks=int(getattr(observation, "riichi_sticks", 0)),
            honba=int(getattr(observation, "honba", 0)),
        )
        evaluator = HandEvaluator(
            sorted(int(tile) for tile in observation.hands[seat]),
            list(observation.melds[seat]),
        )
        result = evaluator.calc(
            int(win_tile),
            [int(tile) for tile in getattr(observation, "dora_indicators", ())],
            conditions,
        )
    except Exception:
        return False
    return bool(getattr(result, "has_win_shape", False)) and not bool(
        getattr(result, "is_win", False)
    )


_REQUIRED_DEFAULTS = {
    "riichi_accepted": [False] * NUM_PLAYERS,
    "riichi_declaration_indices": [None] * NUM_PLAYERS,
    "missed_agari_doujun": False,
    "missed_agari_riichi": False,
    "tiles_left": 70,
    "tsumogiri_flags": [[] for _ in range(NUM_PLAYERS)],
    "last_tedashis": [None] * NUM_PLAYERS,
    "riichi_sutehais": [None] * NUM_PLAYERS,
    "waits": [],
    "is_tenpai": False,
    "drawn_tile": None,
}


def normalize_observation_base64(encoded: str) -> str:
    """返回本地反序列化器可以读取的 base64 Observation。"""
    value = json.loads(base64.b64decode(encoded))
    if not isinstance(value, dict):
        raise ValueError("observation payload must be a JSON object")
    for field, default in _REQUIRED_DEFAULTS.items():
        if field not in value:
            value[field] = default
    return base64.b64encode(
        json.dumps(value, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")


def missing_observation_fields(encoded: str) -> frozenset[str]:
    """返回线上 payload 缺失的本地 Observation 字段。"""
    value = json.loads(base64.b64decode(encoded))
    if not isinstance(value, dict):
        raise ValueError("observation payload must be a JSON object")
    return frozenset(
        field for field in _REQUIRED_DEFAULTS if field not in value
    )


class ObservationView:
    """代理反序列化后的 Observation,并覆盖重建字段。"""

    def __init__(
        self,
        observation: Any,
        fields: dict[str, Any] | None = None,
        missing_fields: frozenset[str] = frozenset(),
    ) -> None:
        object.__setattr__(self, "_observation", observation)
        object.__setattr__(self, "_fields", dict(fields or {}))
        object.__setattr__(
            self, "_missing_fields", frozenset(missing_fields)
        )

    @property
    def missing_fields(self) -> frozenset[str]:
        return object.__getattribute__(self, "_missing_fields")

    @property
    def native_observation(self) -> Any:
        """物化覆盖字段,供 Rust 编码器读取在线重建后的公开状态。"""
        observation = object.__getattribute__(self, "_observation")
        fields = object.__getattribute__(self, "_fields")
        if not fields:
            return observation
        from riichienv import Observation

        payload = json.loads(
            base64.b64decode(observation.serialize_to_base64()).decode("utf-8")
        )
        payload.update(fields)
        encoded = base64.b64encode(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        return Observation.deserialize_from_base64(encoded)

    def set_fields(self, fields: dict[str, Any]) -> None:
        merged = dict(object.__getattribute__(self, "_fields"))
        merged.update(fields)
        object.__setattr__(self, "_fields", merged)

    def __getattr__(self, name: str) -> Any:
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return fields[name]
        observation = object.__getattribute__(self, "_observation")
        return getattr(observation, name)
