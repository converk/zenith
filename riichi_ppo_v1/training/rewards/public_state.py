"""Incremental public-tile accounting for rewards and heuristic defence.

The tracker consumes only bridge-extracted event JSON.  It deliberately has no
reference to observations, so it cannot accidentally read an opponent hand.
"""

from __future__ import annotations

import json

import numpy as np

from ...model.bridge import NUM_PLAYERS

_TILE_TYPES = {
    **{f"{rank}{suit}": (suit_index * 9 + rank - 1) for suit_index, suit in enumerate("mps") for rank in range(1, 10)},
    "E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33,
}


def tile_type(tile: str | None) -> int | None:
    if tile is None:
        return None
    value = str(tile)
    if value in {"5mr", "5pr", "5sr"}:
        value = value[:2]
    return _TILE_TYPES.get(value)


class PublicStateTracker:
    """Per-worker state updated once after each native environment step."""

    def __init__(self, num_envs: int) -> None:
        self.visible = np.zeros((int(num_envs), 34), dtype=np.int16)
        self.discard_masks = np.zeros((int(num_envs), NUM_PLAYERS), dtype=object)
        self.riichi = np.zeros((int(num_envs), NUM_PLAYERS), dtype=np.bool_)
        self.post_riichi_safe_masks = np.zeros((int(num_envs), NUM_PLAYERS), dtype=object)
        self.discard_counts = np.zeros(int(num_envs), dtype=np.int16)
        self.completed_discard_counts = np.zeros(int(num_envs), dtype=np.int16)
        self.discard_counts_by_seat = np.zeros(
            (int(num_envs), NUM_PLAYERS), dtype=np.int16,
        )
        self.completed_discard_counts_by_seat = np.zeros(
            (int(num_envs), NUM_PLAYERS), dtype=np.int16,
        )
        # Only open melds count as furo. Kakan upgrades an existing pon and
        # therefore must not increment this table-level count.
        self.open_meld_counts = np.zeros((int(num_envs), NUM_PLAYERS), dtype=np.int8)
        self.completed_open_meld_counts = np.zeros(int(num_envs), dtype=np.int8)
        self.completed_open_meld_counts_by_seat = np.zeros(
            (int(num_envs), NUM_PLAYERS), dtype=np.int8,
        )
        self.events = 0

    def reset(self, indices: list[int] | tuple[int, ...]) -> None:
        for index in indices:
            self.visible[int(index)].fill(0)
            self.discard_masks[int(index)].fill(0)
            self.riichi[int(index)].fill(False)
            self.post_riichi_safe_masks[int(index)].fill(0)
            self.discard_counts[int(index)] = 0
            self.discard_counts_by_seat[int(index)].fill(0)
            self.open_meld_counts[int(index)].fill(0)

    def remaining(self, env_index: int, own_counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(own_counts, dtype=np.int16)
        if counts.shape != (34,):
            raise ValueError("own_counts must have 34 tile types")
        return np.maximum(0, 4 - counts - self.visible[int(env_index)]).astype(np.int16, copy=False)

    def update(self, events_by_env: list[list[list[str]]]) -> None:
        for env_index, by_seat in enumerate(events_by_env):
            seen: set[str] = set()
            for source_seat, rows in enumerate(by_seat):
                for raw in rows:
                    if raw in seen:
                        continue
                    seen.add(raw)
                    self._apply(env_index, source_seat, raw)

    def _apply(self, env_index: int, source_seat: int, raw: str) -> None:
        try:
            event = json.loads(raw)
        except (TypeError, ValueError):
            return
        kind = str(event.get("type", ""))
        actor = int(event.get("actor", source_seat))
        if kind in {"start_kyoku", "end_kyoku"}:
            if kind == "end_kyoku":
                # RiichiEnv may include the next ``start_kyoku`` in the same
                # event batch. Snapshot before that reset so completed-hand
                # metrics cannot be reported as zero.
                self.completed_discard_counts[env_index] = self.discard_counts[env_index]
                self.completed_discard_counts_by_seat[env_index] = (
                    self.discard_counts_by_seat[env_index]
                )
                self.completed_open_meld_counts[env_index] = self.open_meld_counts[env_index].sum()
                self.completed_open_meld_counts_by_seat[env_index] = (
                    self.open_meld_counts[env_index]
                )
            if kind == "start_kyoku":
                self.reset([env_index])
                marker = tile_type(event.get("dora_marker"))
                if marker is not None:
                    self.visible[env_index, marker] += 1
            return
        if kind in {"reach", "reach_accepted"}:
            self.riichi[env_index, actor] = True
        elif kind in {"chi", "pon", "daiminkan"}:
            self.open_meld_counts[env_index, actor] += 1
        if kind == "dahai":
            self.discard_counts[env_index] += 1
            self.discard_counts_by_seat[env_index, actor] += 1
        tile_values: list[str | None] = []
        if kind in {"dahai", "dora"}:
            tile_values.append(event.get("pai"))
        elif kind in {"chi", "pon", "daiminkan", "ankan"}:
            tile_values.extend(event.get("consumed", ()))
        elif kind == "kakan":
            # The consumed triplet was made public by the original pon.
            # Only the added tile becomes newly visible here.
            tile_values.append(event.get("pai"))
        for value in tile_values:
            kind_index = tile_type(value)
            if kind_index is not None:
                self.visible[env_index, kind_index] += 1
                if kind == "dahai":
                    self.discard_masks[env_index, actor] = int(self.discard_masks[env_index, actor]) | (1 << kind_index)
                    for opponent in np.flatnonzero(self.riichi[env_index]):
                        if int(opponent) != actor:
                            self.post_riichi_safe_masks[env_index, opponent] = (
                                int(self.post_riichi_safe_masks[env_index, opponent])
                                | (1 << kind_index)
                            )
        self.events += 1

    def is_genbutsu_to_all_riichi(self, env_index: int, tile: int) -> bool:
        threats = np.flatnonzero(self.riichi[int(env_index)])
        return bool(len(threats)) and all(int(self.discard_masks[int(env_index), seat]) & (1 << int(tile)) for seat in threats)

    def genbutsu_coverage(self, env_index: int, tile: int) -> int:
        return sum(bool(int(self.discard_masks[int(env_index), seat]) & (1 << int(tile))) for seat in np.flatnonzero(self.riichi[int(env_index)]))

    def has_riichi_threat(self, env_index: int, seat: int) -> bool:
        return bool(np.any(np.delete(self.riichi[int(env_index)], int(seat))))

    def passed_after_riichi(self, env_index: int, opponent: int, tile: int) -> bool:
        return bool(
            int(self.post_riichi_safe_masks[int(env_index), int(opponent)])
            & (1 << int(tile))
        )

    def metrics(self) -> dict[str, float]:
        return {"public_state/events": float(self.events)}
