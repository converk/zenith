"""Cached, batched shanten-first public-ukeire discard rewards."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class HandAnalysis:
    shanten: int
    improving_mask: int


@dataclass(frozen=True, slots=True)
class DiscardCandidate:
    action: object
    analysis: HandAnalysis
    shanten: int
    ukeire: int


@dataclass(frozen=True, slots=True)
class DiscardDecisionAnalysis:
    decision: object
    actions: tuple[object, ...]
    counts: np.ndarray
    opened: int
    candidates: tuple[DiscardCandidate, ...]
    best_shanten: int | None
    best_ukeire: int

    def selected_reward(self, chosen: object) -> float:
        chosen_tile = getattr(chosen, "tile", None)
        if chosen_tile is None or not _is_discard(chosen) or self.best_shanten is None:
            return 0.0
        kind = _action_kind(chosen)
        tile = int(chosen_tile)
        for candidate in self.candidates:
            action = candidate.action
            if _action_kind(action) == kind and int(getattr(action, "tile")) == tile:
                return efficiency_reward(candidate.shanten, candidate.ukeire, self.best_shanten, self.best_ukeire)
        return 0.0


def efficiency_reward(shanten: int, ukeire: int, best_shanten: int, best_ukeire: int) -> float:
    """Return the prescribed [-6, 0] / [-2, 0] shaping reward."""
    if int(shanten) > int(best_shanten):
        return float(np.clip(-3.0 * (int(shanten) - int(best_shanten)), -6.0, 0.0))
    if int(shanten) < int(best_shanten):
        raise ValueError("selected shanten cannot beat the candidate minimum")
    loss = max(0, int(best_ukeire) - int(ukeire)) / max(int(best_ukeire), 1)
    return float(np.clip(-2.0 * loss, -2.0, 0.0))


def remaining_ukeire(improving_mask: int, remaining_counts: np.ndarray) -> int:
    """Count publicly remaining improving tile copies from a 34-element vector."""
    if remaining_counts.shape != (34,):
        raise ValueError("remaining_counts must have 34 tile types")
    return int(sum(int(remaining_counts[tile]) for tile in range(34) if int(improving_mask) & (1 << tile)))


def _action_kind(action: object) -> str:
    value = getattr(action, "action_type", getattr(action, "type", ""))
    return str(getattr(value, "name", value)).lower()


def _is_discard(action: object) -> bool:
    kind = _action_kind(action)
    return kind in {"dahai", "riichi", "reach"} or "discard" in kind


def _counts_and_melds(observation: object) -> tuple[np.ndarray, int]:
    counts = np.zeros(34, dtype=np.uint8)
    seat = int(getattr(observation, "player_id"))
    for physical in getattr(observation, "hands")[seat]:
        counts[int(physical) // 4] += 1
    melds = getattr(observation, "melds", ())
    if len(melds) > seat and isinstance(melds[seat], (list, tuple)):
        opened = len(melds[seat])
    else:
        opened = sum(int(getattr(meld, "seat", -1)) == seat for meld in melds)
    return counts, opened


def selected_efficiency_rewards(
    decisions: list[object],
    selected_actions: list[object],
    *,
    analyzer: "EfficiencyAnalyzer",
    public: object,
    analysis: "DiscardAnalysisBatch | None" = None,
) -> list[float]:
    """Batch reward selected discard actions; non-discard decisions receive zero."""
    if len(decisions) != len(selected_actions):
        raise ValueError("decision/action length mismatch")
    batch = analysis or DiscardAnalysisBatch.build(decisions, analyzer=analyzer, public=public)
    return [batch.for_decision(decision).selected_reward(action) for decision, action in zip(decisions, selected_actions, strict=True)]


class DiscardAnalysisBatch:
    """Step-local discard analysis shared by learner rewards and heuristic opponents."""

    def __init__(self, rows: dict[int, DiscardDecisionAnalysis]) -> None:
        self._rows = rows

    @classmethod
    def build(cls, decisions: Iterable[object], *, analyzer: "EfficiencyAnalyzer", public: object) -> "DiscardAnalysisBatch":
        decision_rows: list[tuple[object, tuple[object, ...], np.ndarray, int, list[object]]] = []
        hands: list[np.ndarray] = []
        melds: list[int] = []
        candidate_refs: list[tuple[int, object]] = []
        for row, decision in enumerate(decisions):
            observation = decision.observation
            actions = tuple(observation.legal_actions())
            counts, opened = _counts_and_melds(observation)
            discard_actions = [
                action for action in actions
                if getattr(action, "tile", None) is not None and _is_discard(action)
            ]
            decision_rows.append((decision, actions, counts, opened, discard_actions))
            for action in discard_actions:
                post = counts.copy()
                post[int(action.tile) // 4] -= 1
                hands.append(post)
                melds.append(opened)
                candidate_refs.append((row, action))

        analyzed = analyzer.analyze(hands, melds) if hands else []
        grouped: list[list[tuple[object, HandAnalysis]]] = [[] for _ in decision_rows]
        for (row, action), hand_analysis in zip(candidate_refs, analyzed, strict=True):
            grouped[row].append((action, hand_analysis))

        rows: dict[int, DiscardDecisionAnalysis] = {}
        for row, (decision, actions, counts, opened, _discard_actions) in enumerate(decision_rows):
            remaining = public.remaining(decision.env_index, counts) if grouped[row] else np.zeros(34, dtype=np.int16)
            candidates = tuple(
                DiscardCandidate(
                    action,
                    hand_analysis,
                    int(hand_analysis.shanten),
                    remaining_ukeire(hand_analysis.improving_mask, remaining),
                )
                for action, hand_analysis in grouped[row]
            )
            if candidates:
                best_shanten = min(candidate.shanten for candidate in candidates)
                best_ukeire = max(candidate.ukeire for candidate in candidates if candidate.shanten == best_shanten)
            else:
                best_shanten = None
                best_ukeire = 0
            rows[id(decision)] = DiscardDecisionAnalysis(
                decision, actions, counts, opened, candidates, best_shanten, best_ukeire,
            )
        return cls(rows)

    def for_decision(self, decision: object) -> DiscardDecisionAnalysis:
        row = self._rows.get(id(decision))
        if row is None:
            counts, opened = _counts_and_melds(decision.observation)
            return DiscardDecisionAnalysis(
                decision,
                tuple(decision.observation.legal_actions()),
                counts,
                opened,
                (),
                None,
                0,
            )
        return row


class EfficiencyAnalyzer:
    """Worker-local bounded cache around the Rust vectorised hand analyzer."""

    def __init__(self, capacity: int = 131_072) -> None:
        self.capacity = max(1, int(capacity))
        self._cache: OrderedDict[tuple[bytes, int], HandAnalysis] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def analyze(self, hands: Iterable[np.ndarray], open_melds: Iterable[int]) -> list[HandAnalysis]:
        rows = [(np.ascontiguousarray(hand, dtype=np.uint8), int(melds)) for hand, melds in zip(hands, open_melds, strict=True)]
        result: list[HandAnalysis | None] = [None] * len(rows)
        misses: dict[tuple[bytes, int], tuple[np.ndarray, int, list[int]]] = {}
        for index, (hand, melds) in enumerate(rows):
            key = (hand.tobytes(), melds)
            cached = self._cache.get(key)
            if cached is not None:
                self.hits += 1
                self._cache.move_to_end(key)
                result[index] = cached
            else:
                self.misses += 1
                entry = misses.get(key)
                if entry is None:
                    misses[key] = (hand, melds, [index])
                else:
                    entry[2].append(index)
        if misses:
            unique = list(misses.items())
            values = self._analyze_unique(unique)
            for (key, (_hand, _melds, indices)), value in zip(unique, values, strict=True):
                self._cache[key] = value
                self._cache.move_to_end(key)
                while len(self._cache) > self.capacity:
                    self._cache.popitem(last=False)
                for index in indices:
                    result[index] = value
        return [value for value in result if value is not None]

    @staticmethod
    def _analyze_unique(unique: list[tuple[tuple[bytes, int], tuple[np.ndarray, int, list[int]]]]) -> list[HandAnalysis]:
        """Use the batched project analyzer when installed, with a safe fallback.

        The current slim state-machine extension intentionally exposes no hand
        analysis API.  Its editable development environment may therefore
        temporarily provide only RiichiEnv's compiled scalar shanten helper.
        The fallback is still native code and is protected by this LRU; it is
        retained solely until the vectorized analyzer extension is installed.
        """
        try:
            import riichi

            analyze_hands = getattr(riichi, "analyze_hands", None)
            if analyze_hands is not None:
                analysis = analyze_hands(
                    np.ascontiguousarray([item[1][0] for item in unique], dtype=np.uint8),
                    np.asarray([item[1][1] for item in unique], dtype=np.uint8),
                )
                return [HandAnalysis(int(analysis.shanten[row, 0]), int(analysis.improving_type_mask[row]))
                        for row in range(len(unique))]
        except ImportError:
            pass
        from riichienv import calculate_shanten

        def physical_tiles(counts: np.ndarray) -> list[int]:
            return [tile * 4 + copy for tile, count in enumerate(counts) for copy in range(int(count))]

        result = []
        for _key, (counts, _melds, _indices) in unique:
            base = int(calculate_shanten(physical_tiles(counts)))
            mask = 0
            for tile, count in enumerate(counts):
                if int(count) >= 4:
                    continue
                next_counts = counts.copy()
                next_counts[tile] += 1
                if int(calculate_shanten(physical_tiles(next_counts))) < base:
                    mask |= 1 << tile
            result.append(HandAnalysis(base, mask))
        return result

    def metrics(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "reward_analysis/cache_hits": float(self.hits),
            "reward_analysis/cache_misses": float(self.misses),
            "reward_analysis/cache_hit_rate": float(self.hits / max(total, 1)),
            "reward_analysis/cache_entries": float(len(self._cache)),
        }
