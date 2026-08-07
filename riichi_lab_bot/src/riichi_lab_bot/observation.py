"""Online RiichiEnv observation normalization for the lab server schema.

The lab server's base64 Observation predates several schema-13 snapshot
fields.  This module (a) injects the fields the local deserializer requires,
and (b) rebuilds the full per-player snapshot from the accumulated MJAI event
stream so the V13 actor encoder still receives correct semantics.
"""

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
    """Map an MJAI tile string to a physical RiichiEnv tile id (copy 0)."""
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
    """Accumulate the per-player snapshot facts the server omits."""

    def __init__(self) -> None:
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
        elif kind == "reach":
            self.riichi_declared[actor] = True
            self.riichi_declaration_indices[actor] = self.discard_counts[actor]
        elif kind == "reach_accepted":
            self.riichi_accepted[actor] = True
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

    def apply_events(self, events: list[Any]) -> None:
        for raw in events:
            self.apply_event(raw)

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
        }


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
    """Return a base64 Observation the local deserializer accepts."""
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
    """Return the schema-13 snapshot fields absent from the server payload."""
    value = json.loads(base64.b64decode(encoded))
    if not isinstance(value, dict):
        raise ValueError("observation payload must be a JSON object")
    return frozenset(
        field for field in _REQUIRED_DEFAULTS if field not in value
    )


class ObservationView:
    """Delegate to a deserialized Observation but override derived fields."""

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
