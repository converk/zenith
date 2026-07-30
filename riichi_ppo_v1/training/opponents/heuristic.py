"""CPU-only, public-information heuristic opponents.

The policy receives an observation only for its own seat and an incremental
``PublicStateTracker``.  It intentionally never accepts the all-seat table
observation used by the privileged critic.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..rewards.decision import (
    DecisionAnalysisBatch,
    consumed_tiles,
    public_remaining,
)
from ..rewards.efficiency import (
    DiscardAnalysisBatch,
    EfficiencyAnalyzer,
    remaining_ukeire,
)
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
        self._second_order_cache_row: Any | None = None
        self._second_order_cache: dict[int, int] = {}

    @staticmethod
    def _efficiency_key(candidate: Any, observation: Any) -> tuple[int, ...]:
        """Rank offence without search or additional hand-evaluator calls."""
        action = candidate.action
        raw_tile = getattr(action, "tile", None)
        physical_tile = -1 if raw_tile is None else int(raw_tile)
        tile = -1 if physical_tile < 0 else physical_tile // 4
        effective = int(
            getattr(candidate, "effective_shanten", getattr(candidate, "shanten", 99))
        )
        structural = int(
            getattr(candidate, "structural_shanten", getattr(candidate, "shanten", 99))
        )
        ukeire = int(getattr(candidate, "ukeire", 0))
        preserve_dora = int(getattr(candidate, "preserve_dora", 0))
        preserve_red = int(getattr(candidate, "preserve_red", 0))
        if not hasattr(candidate, "preserve_dora"):
            # Compatibility with the older discard-only analysis and its test
            # doubles.  A discarded value tile means one fewer preserved copy.
            preserve_dora = -int(tile >= 0 and _is_dora(observation, tile))
            preserve_red = -int(
                physical_tile >= 0 and _is_aka_dora(physical_tile)
            )
        if effective == 0:
            opportunity_value = (
                int(getattr(candidate, "ron_value_sum", 0))
                + int(getattr(candidate, "tsumo_value_sum", 0))
            )
            if (
                bool(getattr(candidate, "riichi_route", False))
                and not bool(getattr(candidate, "has_yaku", False))
            ):
                opportunity_value = (
                    int(getattr(candidate, "riichi_ron_value_sum", 0))
                    + int(getattr(candidate, "tsumo_value_sum", 0))
                )
            live_waits = max(
                int(getattr(candidate, "live_ron", 0)),
                int(getattr(candidate, "live_tsumo", 0)),
            )
            quality = opportunity_value
        else:
            live_waits = 0
            quality = 100 * ukeire + 250 * preserve_dora + 150 * preserve_red
        return (
            -effective,
            -structural,
            quality,
            live_waits,
            ukeire,
            preserve_dora,
            preserve_red,
            -physical_tile,
        )

    @staticmethod
    def _own_counts(observation: Any) -> list[int]:
        counts = [0] * 34
        seat = int(getattr(observation, "player_id", 0))
        hands = getattr(observation, "hands", ())
        hand = hands[seat] if len(hands) > seat else getattr(observation, "hand", ())
        for physical in hand:
            counts[int(physical) // 4] += 1
        return counts

    @staticmethod
    def _sequence_pairs(tile: int) -> tuple[tuple[int, int], ...]:
        """Return all two-tile shapes that can wait on this suited tile."""
        if tile >= 27:
            return ()
        suit_base = tile // 9 * 9
        number = tile - suit_base
        pairs: list[tuple[int, int]] = []
        for start in range(number - 2, number + 1):
            if 0 <= start <= 6:
                sequence = [suit_base + start + offset for offset in range(3)]
                sequence.remove(tile)
                pairs.append((sequence[0], sequence[1]))
        return tuple(pairs)

    def _danger_score(self, candidate: Any, row: Any, env_index: int, seat: int) -> int:
        """Estimate deal-in danger from public copies, walls, suji and genbutsu."""
        tile = int(candidate.action.tile) // 4
        observation = row.decision.observation
        own_counts = self._own_counts(observation)
        remaining = self.public.remaining(
            env_index, np.asarray(own_counts, dtype=np.uint8),
        )
        exhausted = bool(getattr(candidate, "four_visible", False))
        if hasattr(row, "counts"):
            exhausted = int(row.counts[tile]) + int(self.public.visible[env_index, tile]) >= 4
        if exhausted:
            return 0
        total = 0
        for opponent in range(4):
            if opponent == seat or not self.public.riichi[env_index, opponent]:
                continue
            discard_mask = int(self.public.discard_masks[env_index, opponent])
            if discard_mask & (1 << tile):
                continue
            unseen = int(remaining[tile])
            if tile >= 27:
                total += 4 * unseen
                continue
            # A one-chance sequence contributes half the ordinary danger; a
            # hard wall removes the shape altogether.
            live_pattern_units = sum(
                min(
                    2,
                    int(remaining[left]),
                    int(remaining[right]),
                )
                for left, right in self._sequence_pairs(tile)
            )
            suji = _suji_safety(tile, discard_mask)
            total += max(1, 3 * unseen + 2 * live_pattern_units - 2 * suji)
        return total

    @staticmethod
    def _is_call(candidate: Any) -> bool:
        return _kind(candidate.action) in {"chi", "pon", "daiminkan"}

    @staticmethod
    def _is_value_honour(action: Any, observation: Any, seat: int) -> bool:
        physical = getattr(action, "tile", None)
        if physical is None:
            return False
        tile = int(physical) // 4
        if tile >= 31:
            return True
        round_wind = int(getattr(observation, "round_wind", -99))
        oya = int(getattr(observation, "oya", -99))
        player_wind = (int(seat) - oya) % 4 if oya >= 0 else -99
        return tile in {27 + round_wind, 27 + player_wind}

    def _call_is_worthwhile(
        self, best: Any, fallback: Any, observation: Any, seat: int,
    ) -> bool:
        """Apply conservative, cheap opening thresholds after candidate scoring."""
        kind = _kind(best.action)
        if bool(getattr(best, "open_no_yaku", False)):
            return False
        best_effective = int(
            getattr(best, "effective_shanten", getattr(best, "shanten", 99))
        )
        pass_effective = int(
            getattr(
                fallback, "effective_shanten",
                getattr(fallback, "shanten", 99),
            )
        )
        if kind == "daiminkan":
            return (
                best_effective == 0
                and bool(getattr(best, "has_yaku", False))
            )
        was_open = not bool(getattr(fallback, "closed", True))
        value_honour = self._is_value_honour(best.action, observation, seat)
        if best_effective < pass_effective:
            return best_effective <= 1 or was_open or value_honour
        return (
            best_effective == pass_effective
            and (was_open or value_honour)
            and int(getattr(best, "ukeire", 0))
            >= int(getattr(fallback, "ukeire", 0)) + 4
            and int(getattr(best, "preserve_dora", 0))
            >= int(getattr(fallback, "preserve_dora", 0))
            and int(getattr(best, "preserve_red", 0))
            >= int(getattr(fallback, "preserve_red", 0))
        )

    @staticmethod
    def _live_waits(candidate: Any) -> int:
        return max(
            int(getattr(candidate, "live_ron", 0)),
            int(getattr(candidate, "live_tsumo", 0)),
        )

    @staticmethod
    def _max_points(candidate: Any) -> int:
        return max(
            int(getattr(candidate, "ron_points", 0)),
            int(getattr(candidate, "tsumo_points", 0)),
        )

    @classmethod
    def _potential_points(cls, candidate: Any) -> int:
        return max(
            cls._max_points(candidate),
            int(getattr(candidate, "riichi_ron_points", 0)),
        )

    def _should_push(
        self, candidate: Any, observation: Any, env_index: int,
    ) -> bool:
        effective = int(
            getattr(candidate, "effective_shanten", getattr(candidate, "shanten", 99))
        )
        seat = int(getattr(observation, "player_id", 0))
        dealer = seat == int(getattr(observation, "oya", -1))
        if effective == 0:
            wait_threshold = 4 if dealer else 5
            point_threshold = 2_900 if dealer else 3_900
            enough = (
                self._live_waits(candidate) >= wait_threshold
                or self._potential_points(candidate) >= point_threshold
            )
            threat_count = int(self.public.riichi[env_index].sum())
            if threat_count >= 2:
                return enough and (
                    self._live_waits(candidate) >= wait_threshold + 1
                    or self._potential_points(candidate)
                    >= (5_800 if dealer else 7_700)
                )
            return enough
        if effective != 1 or int(self.public.discard_counts[env_index]) >= 48:
            return False
        value_tiles = (
            int(getattr(candidate, "preserve_dora", 0))
            + int(getattr(candidate, "preserve_red", 0))
        )
        return (
            int(getattr(candidate, "ukeire", 0)) >= (10 if dealer else 12)
            and value_tiles >= (1 if dealer else 2)
        )

    def _should_riichi(
        self, candidate: Any, row: Any, env_index: int, seat: int, threat: bool,
    ) -> bool:
        if int(
            getattr(candidate, "effective_shanten", getattr(candidate, "shanten", 99))
        ) != 0:
            return False
        observation = row.decision.observation
        if threat and (
            self._danger_score(candidate, row, env_index, seat) != 0
            and not self._should_push(candidate, observation, env_index)
        ):
            return False
        has_yaku = bool(getattr(candidate, "has_yaku", False))
        if not has_yaku:
            return bool(getattr(candidate, "riichi_route", True))
        dealer = seat == int(getattr(observation, "oya", -1))
        if self._max_points(candidate) >= (11_600 if dealer else 7_700):
            return False
        if (
            int(self.public.discard_counts[env_index]) >= 56
            and self._live_waits(candidate) <= 4
        ):
            return False
        return True

    @staticmethod
    def _physical_hand(observation: Any) -> list[int]:
        seat = int(getattr(observation, "player_id", 0))
        hands = getattr(observation, "hands", ())
        if len(hands) > seat:
            return [int(tile) for tile in hands[seat]]
        return [int(tile) for tile in getattr(observation, "hand", ())]

    @staticmethod
    def _opened_melds(observation: Any) -> int:
        seat = int(getattr(observation, "player_id", 0))
        melds = getattr(observation, "melds", ())
        return len(melds[seat]) if len(melds) > seat else 0

    def _second_order_bonuses(
        self, candidates: list[Any], row: Any,
    ) -> dict[int, int]:
        """Score the next-best ukeire for two tied early one-shanten cuts."""
        analyze = getattr(self.analyzer, "analyze", None)
        observation = row.decision.observation
        hand = self._physical_hand(observation)
        env_index = int(row.decision.env_index)
        if (
            not callable(analyze)
            or not hand
            or self.defensive
            or int(self.public.discard_counts[env_index]) >= 24
        ):
            return {}
        ranked = sorted(
            (
                candidate for candidate in candidates
                if _kind(candidate.action) == "dahai"
                and int(getattr(candidate, "effective_shanten", 99)) == 1
                and int(getattr(candidate, "improving_mask", 0))
            ),
            key=lambda candidate: self._efficiency_key(candidate, observation),
            reverse=True,
        )
        if len(ranked) < 2:
            return {}
        best_key = self._efficiency_key(ranked[0], observation)
        ranked = [
            candidate for candidate in ranked[:2]
            if int(getattr(candidate, "structural_shanten", 99))
            == int(getattr(ranked[0], "structural_shanten", 99))
            and best_key[2]
            == self._efficiency_key(candidate, observation)[2]
        ]
        if len(ranked) < 2:
            return {}

        remaining = public_remaining(observation)
        opened = self._opened_melds(observation)
        jobs: list[np.ndarray] = []
        refs: list[tuple[int, int, int]] = []
        draw_remaining: dict[int, np.ndarray] = {}
        for candidate in ranked:
            post = np.zeros(34, dtype=np.uint8)
            post_hand = hand.copy()
            try:
                post_hand.remove(int(candidate.action.tile))
            except ValueError:
                continue
            for physical in post_hand:
                post[physical // 4] += 1
            mask = int(getattr(candidate, "improving_mask", 0))
            for draw in range(34):
                copies = int(remaining[draw])
                if not (mask & (1 << draw)) or copies <= 0:
                    continue
                after_draw = post.copy()
                after_draw[draw] += 1
                next_remaining = remaining.copy()
                next_remaining[draw] = max(0, next_remaining[draw] - 1)
                draw_remaining[draw] = next_remaining
                for discard in np.flatnonzero(after_draw):
                    next_hand = after_draw.copy()
                    next_hand[int(discard)] -= 1
                    jobs.append(next_hand)
                    refs.append((id(candidate), draw, copies))
        if not jobs:
            return {}
        analyses = analyze(jobs, [opened] * len(jobs))
        best_by_draw: dict[tuple[int, int], tuple[int, int]] = {}
        copies_by_draw: dict[tuple[int, int], int] = {}
        for ref, analysis in zip(refs, analyses, strict=True):
            candidate_id, draw, copies = ref
            next_ukeire = remaining_ukeire(
                int(analysis.improving_mask), draw_remaining[draw],
            )
            key = (int(analysis.shanten), -next_ukeire)
            group = (candidate_id, draw)
            current = best_by_draw.get(group)
            if current is None or key < current:
                best_by_draw[group] = key
                copies_by_draw[group] = copies

        weighted: dict[int, int] = {}
        for (candidate_id, _draw), (_shanten, negative_ukeire) in best_by_draw.items():
            weighted[candidate_id] = (
                weighted.get(candidate_id, 0)
                + copies_by_draw[(candidate_id, _draw)] * -negative_ukeire
            )
        return {
            id(candidate): 10 * weighted.get(id(candidate), 0)
            // max(int(getattr(candidate, "ukeire", 0)), 1)
            for candidate in ranked
        }

    def _best_offence(self, candidates: list[Any], row: Any) -> Any:
        if self._second_order_cache_row is row:
            bonuses = self._second_order_cache
        else:
            bonuses = self._second_order_bonuses(candidates, row)
            self._second_order_cache_row = row
            self._second_order_cache = bonuses

        def key(candidate: Any) -> tuple[int, ...]:
            base = self._efficiency_key(
                candidate, row.decision.observation,
            )
            if id(candidate) not in bonuses:
                return base
            return (
                base[0], base[1], base[2] + bonuses[id(candidate)],
                *base[3:],
            )

        return max(
            candidates,
            key=key,
        )

    def _kan_is_worthwhile(
        self, action: Any, best: Any, observation: Any,
    ) -> bool:
        """Analyze the rare post-kan shape without changing candidate tokens."""
        analyze = getattr(self.analyzer, "analyze", None)
        if not callable(analyze):
            return False
        kind = _kind(action)
        hand = self._physical_hand(observation)
        raw_tile = getattr(action, "tile", None)
        if raw_tile is None or not hand:
            return False
        tile_type = int(raw_tile) // 4
        post = hand.copy()
        if kind == "ankan":
            removed = [tile for tile in post if tile // 4 == tile_type][:4]
            if len(removed) != 4:
                removed = list(consumed_tiles(action))
            try:
                for tile in removed:
                    post.remove(int(tile))
            except ValueError:
                return False
            opened = self._opened_melds(observation) + 1
        elif kind == "kakan":
            added = next(
                (tile for tile in post if tile // 4 == tile_type),
                None,
            )
            if added is None:
                return False
            post.remove(added)
            opened = self._opened_melds(observation)
        else:
            return False
        counts = np.zeros(34, dtype=np.uint8)
        for tile in post:
            counts[tile // 4] += 1
        analysis = analyze([counts], [opened])[0]
        structural = int(getattr(best, "structural_shanten", 99))
        effective = int(getattr(best, "effective_shanten", structural))
        value_tiles = (
            int(getattr(best, "preserve_dora", 0))
            + int(getattr(best, "preserve_red", 0))
        )
        return (
            int(analysis.shanten) <= structural
            and effective == structural
            and (effective == 0 or value_tiles >= 2)
        )

    def _best_safe(
        self, candidates: list[Any], row: Any, env_index: int, seat: int,
    ) -> Any:
        return min(
            candidates,
            key=lambda candidate: (
                self._danger_score(candidate, row, env_index, seat),
                tuple(
                    -value for value in self._efficiency_key(
                        candidate, row.decision.observation,
                    )
                ),
            ),
        )

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
                pass_action = next(
                    (action for action in actions if _kind(action) in {"none", "pass"}),
                    None,
                )
                has_reaction_call = any(
                    _kind(action) in {"chi", "pon", "daiminkan"} for action in actions
                )
                if threat and has_reaction_call and pass_action is not None:
                    # Calling into an established riichi removes a safe
                    # reaction and exposes a forced discard.  The defensive
                    # baseline folds unless it can win immediately.
                    output[index] = pass_action
                    continue
                defensive_discards = [
                    candidate for candidate in row.candidates
                    if _is_discard(candidate.action)
                    and getattr(candidate.action, "tile", None) is not None
                ]
                if threat and defensive_discards:
                    offensive = self._best_offence(defensive_discards, row)
                    best = (
                        offensive
                        if (
                            self._danger_score(offensive, row, env_index, seat) == 0
                            or self._should_push(
                                offensive, decision.observation, env_index,
                            )
                        )
                        else self._best_safe(
                            defensive_discards, row, env_index, seat,
                        )
                    )
                else:
                    candidates = list(row.candidates)
                    pass_candidates = [
                        candidate for candidate in candidates
                        if _kind(candidate.action) in {"none", "pass"}
                    ]
                    if pass_candidates:
                        fallback = self._best_offence(pass_candidates, row)
                        candidates = [
                            candidate for candidate in candidates
                            if (
                                not self._is_call(candidate)
                                or self._call_is_worthwhile(
                                    candidate, fallback,
                                    decision.observation, seat,
                                )
                            )
                        ]
                    best = self._best_offence(candidates, row)
            else:
                candidates = list(row.candidates)
                offensive = self._best_offence(candidates, row)
                best = (
                    self._best_safe(candidates, row, env_index, seat)
                    if threat
                    and self._danger_score(
                        offensive, row, env_index, seat,
                    ) != 0
                    and not self._should_push(
                        offensive, decision.observation, env_index,
                    )
                    else offensive
                )
            riichi = next((action for action in actions if _kind(action) in {"riichi", "reach"}), None)
            if riichi is not None and self._should_riichi(
                best, row, env_index, seat, threat,
            ):
                # Riichi is a separate legal action in RiichiEnv.  The next
                # decision is restricted to tenpai-preserving discards, so
                # declare it explicitly instead of accidentally treating it
                # as a tied discard candidate.
                output[index] = riichi
            else:
                kans = [
                    action for action in actions
                    if _kind(action) in {"ankan", "kakan"}
                ]
                own_riichi = bool(
                    (
                        getattr(decision.observation, "riichi_declared", ())
                        or (False,) * 4
                    )[seat]
                )
                chosen_kan = None
                if (
                    kans
                    and not self.public.has_riichi_threat(env_index, seat)
                    and _is_discard(best.action)
                ):
                    ankan = next(
                        (action for action in kans if _kind(action) == "ankan"),
                        None,
                    )
                    if own_riichi:
                        # RiichiEnv exposes only wait-preserving post-riichi
                        # ankan, so no further shape simulation is necessary.
                        chosen_kan = ankan
                    else:
                        ordered = sorted(
                            kans, key=lambda action: _kind(action) != "ankan",
                        )
                        chosen_kan = next(
                            (
                                action for action in ordered
                                if self._kan_is_worthwhile(
                                    action, best, decision.observation,
                                )
                            ),
                            None,
                        )
                output[index] = chosen_kan or best.action
        if any(action is None for action in output):
            raise RuntimeError("heuristic failed to choose a legal action")
        return [action for action in output if action is not None]
