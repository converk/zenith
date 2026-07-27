"""CPU-only, public-information heuristic opponents.

The policy receives an observation only for its own seat and an incremental
``PublicStateTracker``.  It intentionally never accepts the all-seat table
observation used by the privileged critic.
"""

from __future__ import annotations

import json
from typing import Any

from ..rewards.decision import DecisionAnalysisBatch
from ..rewards.efficiency import DiscardAnalysisBatch, EfficiencyAnalyzer
from ..rewards.public_state import PublicStateTracker


def _kind(action: Any) -> str:
    try:
        return str(json.loads(action.to_mjai()).get("type", "")).lower()
    except (AttributeError, TypeError, ValueError):
        pass
    value = getattr(action, "action_type", getattr(action, "type", ""))
    return str(getattr(value, "name", value)).lower().rsplit(".", 1)[-1]


def _is_discard(action: Any) -> bool:
    return _kind(action) in {"dahai", "riichi", "reach"} or "discard" in _kind(action)


def _is_aka_dora(physical_tile: int) -> bool:
    """Return whether a physical tile is one of the three red fives."""
    return int(physical_tile) in {16, 52, 88}


def _dora_type(indicator: int) -> int:
    """Return the tile type indicated by a public dora indicator."""
    tile = int(indicator) // 4
    if tile >= 27:
        # East -> South -> West -> North -> East; white -> green -> red.
        return 27 + ((tile - 27 + 1) % 4) if tile < 31 else 31 + ((tile - 31 + 1) % 3)
    suit, number = divmod(tile, 9)
    return suit * 9 + ((number + 1) % 9)


def _is_dora(observation: Any, tile: int) -> bool:
    indicators = getattr(observation, "dora_indicators", None) or ()
    return any(_dora_type(indicator) == int(tile) for indicator in indicators)


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

    @staticmethod
    def _efficiency_key(candidate: Any, observation: Any) -> tuple[int, int, int, int, int]:
        """Shanten/ukeire first, then preserve known scoring value deterministically."""
        action = candidate.action
        physical_tile = int(action.tile)
        tile = physical_tile // 4
        return (
            -int(getattr(candidate, "structural_shanten", getattr(candidate, "shanten", 99))),
            int(candidate.ukeire),
            -int(_is_dora(observation, tile)),
            -int(_is_aka_dora(physical_tile)),
            -physical_tile,
        )

    def _defense_key(self, candidate: Any, row: Any, env_index: int, seat: int) -> tuple[int, ...]:
        """Rank safety against every declared riichi before hand efficiency.

        A fully visible tile cannot be in any opponent's concealed hand.  For
        non-exhausted tiles, genbutsu is certain safety; suji is only a weak
        public signal and is counted across *all* riichi opponents rather than
        taking the previous, misleading maximum for one opponent.
        """
        action = candidate.action
        physical_tile = int(action.tile)
        tile = physical_tile // 4
        threats = [
            opponent
            for opponent in range(4)
            if opponent != seat and self.public.riichi[env_index, opponent]
        ]
        exhausted = int(getattr(candidate, "four_visible", False))
        if hasattr(row, "counts"):
            exhausted = int(int(row.counts[tile]) + int(self.public.visible[env_index, tile]) >= 4)
        genbutsu_all = int(bool(threats) and self.public.is_genbutsu_to_all_riichi(env_index, tile))
        coverage = self.public.genbutsu_coverage(env_index, tile)
        suji_coverage = sum(
            _suji_safety(tile, int(self.public.discard_masks[env_index, opponent]))
            for opponent in threats
        )
        return (
            exhausted,
            genbutsu_all,
            coverage,
            suji_coverage,
            *self._efficiency_key(candidate, row.decision.observation),
        )

    def _candidate_key(self, candidate: Any, row: Any, env_index: int, seat: int, threat: bool) -> tuple[int, ...]:
        if threat:
            return self._defense_key(candidate, row, env_index, seat)
        return self._efficiency_key(candidate, row.decision.observation)

    def _safe_tenpai_under_riichi(self, candidate: Any, row: Any, env_index: int, seat: int) -> bool:
        """Whether defence can commit to riichi without giving up certain safety."""
        if int(getattr(candidate, "effective_shanten", getattr(candidate, "shanten", 99))) != 0:
            return False
        tile = int(candidate.action.tile) // 4
        exhausted = bool(getattr(candidate, "four_visible", False))
        if hasattr(row, "counts"):
            exhausted = int(row.counts[tile]) + int(self.public.visible[env_index, tile]) >= 4
        return exhausted or self.public.is_genbutsu_to_all_riichi(env_index, tile)

    def select_batch(
        self,
        decisions: list[Any],
        analysis_batch: DiscardAnalysisBatch | DecisionAnalysisBatch | None = None,
    ) -> list[Any]:
        """Select one legal native action per decision without GPU/RPC work."""
        if not decisions:
            return []
        analysis_batch = analysis_batch or DecisionAnalysisBatch.build(
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
            if isinstance(analysis_batch, DecisionAnalysisBatch):
                defensive_discards = [
                    candidate for candidate in row.candidates
                    if _is_discard(candidate.action)
                    and getattr(candidate.action, "tile", None) is not None
                ]
                if threat and defensive_discards:
                    best = max(
                        defensive_discards,
                        key=lambda candidate: self._defense_key(
                            candidate, row, env_index, seat,
                        ),
                    )
                else:
                    best = min(row.candidates, key=lambda candidate: candidate.rank)
            else:
                best = max(
                    row.candidates,
                    key=lambda candidate: self._candidate_key(candidate, row, env_index, seat, threat),
                )
            riichi = next((action for action in actions if _kind(action) in {"riichi", "reach"}), None)
            if riichi is not None and (
                not threat or self._safe_tenpai_under_riichi(best, row, env_index, seat)
            ):
                # Riichi is a separate legal action in RiichiEnv.  The next
                # decision is restricted to tenpai-preserving discards, so
                # declare it explicitly instead of accidentally treating it
                # as a tied discard candidate.
                output[index] = riichi
            else:
                output[index] = best.action
        if any(action is None for action in output):
            raise RuntimeError("heuristic failed to choose a legal action")
        return [action for action in output if action is not None]
