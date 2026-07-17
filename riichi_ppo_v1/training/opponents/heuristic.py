"""CPU-only, public-information heuristic opponents.

The policy receives an observation only for its own seat and an incremental
``PublicStateTracker``.  It intentionally never accepts the all-seat table
observation used by the privileged critic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..rewards.efficiency import DiscardAnalysisBatch, EfficiencyAnalyzer
from ..rewards.public_state import PublicStateTracker


def _kind(action: Any) -> str:
    value = getattr(action, "action_type", getattr(action, "type", ""))
    return str(getattr(value, "name", value)).lower()


def _is_discard(action: Any) -> bool:
    return _kind(action) in {"dahai", "riichi", "reach"} or "discard" in _kind(action)


def _own_counts(observation: Any) -> np.ndarray:
    counts = np.zeros(34, dtype=np.uint8)
    seat = int(observation.player_id)
    for physical in getattr(observation, "hands")[seat]:
        counts[int(physical) // 4] += 1
    return counts


def _open_melds(observation: Any) -> int:
    seat = int(observation.player_id)
    melds = getattr(observation, "melds", ())
    if len(melds) > seat and isinstance(melds[seat], (list, tuple)):
        return len(melds[seat])
    return sum(int(getattr(meld, "seat", -1)) == seat for meld in melds)


def _suji_safety(tile: int, discard_mask: int) -> int:
    """A small public-only suji score; honours have no suji guarantee."""
    if tile >= 27:
        return 0
    number = tile % 9 + 1
    suit_base = tile // 9 * 9
    partners: tuple[int, ...]
    if number <= 3:
        partners = (suit_base + number + 2,)
    elif number >= 7:
        partners = (suit_base + number - 4,)
    else:
        partners = (suit_base + number - 4, suit_base + number + 2)
    return sum(bool(discard_mask & (1 << partner)) for partner in partners)


class HeuristicPolicy:
    def __init__(self, analyzer: EfficiencyAnalyzer, public: PublicStateTracker, *, defensive: bool) -> None:
        self.analyzer = analyzer
        self.public = public
        self.defensive = bool(defensive)

    def select_batch(self, decisions: list[Any], analysis_batch: DiscardAnalysisBatch | None = None) -> list[Any]:
        """Select one legal native action per decision without GPU/RPC work."""
        if not decisions:
            return []
        analysis_batch = analysis_batch or DiscardAnalysisBatch.build(
            decisions, analyzer=self.analyzer, public=self.public,
        )
        output: list[Any | None] = [None] * len(decisions)
        for index, decision in enumerate(decisions):
            row = analysis_batch.for_decision(decision)
            actions = row.actions
            immediate = next((action for action in actions if _kind(action) in {"tsumo", "ron", "hora"}), None)
            if immediate is not None:
                output[index] = immediate
                continue
            if not row.candidates:
                # Pass calls when they cannot be cheaply evaluated from the
                # public legal-action shape.  This is a safe, deterministic
                # fallback for chi/pon/kan reaction windows.
                output[index] = next((action for action in actions if _kind(action) in {"none", "pass"}), actions[0])
                continue
            env_index = int(decision.env_index)
            seat = int(decision.seat_id)
            threat = self.defensive and self.public.has_riichi_threat(env_index, seat)
            def key(candidate: Any) -> tuple[int, int, int, int]:
                action = candidate.action
                tile = int(action.tile) // 4
                if threat:
                    genbutsu_all = int(self.public.is_genbutsu_to_all_riichi(env_index, tile))
                    coverage = self.public.genbutsu_coverage(env_index, tile)
                    suji = max((_suji_safety(tile, int(self.public.discard_masks[env_index, opponent]))
                                for opponent in range(4) if opponent != seat and self.public.riichi[env_index, opponent]), default=0)
                    return (genbutsu_all, coverage, suji, -candidate.shanten * 1000 + candidate.ukeire)
                return (0, 0, 0, -candidate.shanten * 1000 + candidate.ukeire)
            output[index] = max(row.candidates, key=key).action
        if any(action is None for action in output):
            raise RuntimeError("heuristic failed to choose a legal action")
        return [action for action in output if action is not None]
