"""Rule-aware public decision analysis shared by rollout, teachers and opponents.

The analyzer deliberately uses only fields visible in the acting player's
``Observation``.  Structural shanten is batched through ``EfficiencyAnalyzer``;
exact waits, yaku and score estimates are evaluated only for structural-tenpai
candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np

from ...model.validation import TILE34, TILE37, _chi_pairs, _pon_pairs
from .efficiency import EfficiencyAnalyzer, HandAnalysis, remaining_ukeire

NUM_ACTIONS = 241
SCHEMA_VERSION = 8


def _action_type_code(aid: int) -> int:
    if aid == 0:
        return 1
    if 1 <= aid <= 74:
        return 2
    if aid == 75:
        return 3
    if 76 <= aid <= 132:
        return 4
    if 133 <= aid <= 169:
        return 5
    if aid == 170:
        return 6
    if 171 <= aid <= 204:
        return 7
    if 205 <= aid <= 238:
        return 8
    if aid == 239:
        return 9
    return 10


def action_kind(action: object) -> str:
    """Return the canonical MJAI action kind."""
    try:
        return str(json.loads(action.to_mjai()).get("type", "")).lower()
    except (AttributeError, TypeError, ValueError):
        value = getattr(action, "action_type", getattr(action, "type", ""))
        return str(getattr(value, "name", value)).lower().rsplit(".", 1)[-1]


def consumed_tiles(action: object) -> tuple[int, ...]:
    values = getattr(action, "consume_tiles", getattr(action, "consumed", ())) or ()
    return tuple(int(value) for value in values)


def action_key(action: object) -> tuple[str, int, tuple[int, ...]]:
    """Stable semantic key across independently decoded native actions."""
    kind = action_kind(action)
    tile = getattr(action, "tile", None)
    tile_type = -1 if tile is None else int(tile) // 4
    return kind, tile_type, tuple(sorted(value // 4 for value in consumed_tiles(action)))


def action_id(action: object, observation: object) -> int | None:
    """Map a RiichiEnv action to the fixed 241-way policy id."""
    kind = action_kind(action)
    try:
        row = json.loads(action.to_mjai())
    except (AttributeError, TypeError, ValueError):
        row = {}
    if kind in {"none", "pass"}:
        return 0
    if kind == "dahai":
        pai = str(row.get("pai", ""))
        drawn = getattr(observation, "drawn_tile", None)
        tile = getattr(action, "tile", None)
        mode = int(drawn is not None and tile is not None and int(drawn) == int(tile))
        try:
            return 1 + 2 * TILE37.index(pai) + mode
        except ValueError:
            return None
    if kind == "reach":
        return 75
    if kind == "chi":
        consumed = tuple(sorted(str(value) for value in row.get("consumed", ())))
        for index, pair in enumerate(_chi_pairs()):
            if consumed == tuple(sorted(pair)):
                return 76 + index
        return None
    if kind == "pon":
        consumed = tuple(sorted(str(value) for value in row.get("consumed", ())))
        for index, pair in enumerate(_pon_pairs()):
            if consumed == tuple(sorted(pair)):
                return 133 + index
        return None
    if kind == "daiminkan":
        return 170
    if kind in {"ankan", "kakan"}:
        values = consumed_tiles(action)
        tile = getattr(action, "tile", None)
        tile_type = (values[0] if values else int(tile)) // 4
        return (171 if kind == "ankan" else 205) + tile_type
    if kind in {"hora", "ron", "tsumo"}:
        return 239
    if kind in {"ryukyoku", "kyushukyuhai", "kyushu_kyuhai"}:
        return 240
    return None


def _own_melds(observation: object) -> list[object]:
    seat = int(getattr(observation, "player_id"))
    rows = getattr(observation, "melds", ()) or ()
    return list(rows[seat]) if len(rows) > seat else []


def _physical_hand(observation: object) -> list[int]:
    seat = int(getattr(observation, "player_id"))
    hands = getattr(observation, "hands", None)
    if hands is not None:
        return [int(value) for value in hands[seat]]
    return [int(value) for value in getattr(observation, "hand")]


def _counts(hand: Iterable[int]) -> np.ndarray:
    result = np.zeros(34, dtype=np.uint8)
    for tile in hand:
        result[int(tile) // 4] += 1
    return result


def public_remaining(observation: object, hand: Iterable[int] | None = None) -> np.ndarray:
    """Build physical-id-deduplicated public remaining counts.

    A called discard may exist in both a river and a meld representation.  Its
    physical id is counted once, unlike event-level additive trackers.
    """
    own = list(_physical_hand(observation) if hand is None else hand)
    visible: set[int] = set()
    for river in getattr(observation, "discards", ()) or ():
        visible.update(int(tile) for tile in river)
    for meld_rows in getattr(observation, "melds", ()) or ():
        for meld in meld_rows:
            visible.update(int(tile) for tile in getattr(meld, "tiles", ()) or ())
    visible.update(int(tile) for tile in getattr(observation, "dora_indicators", ()) or ())
    result = np.full(34, 4, dtype=np.int16)
    for tile in own:
        result[tile // 4] -= 1
    for tile in visible.difference(own):
        result[tile // 4] -= 1
    return np.maximum(result, 0)


def _is_closed(melds: Iterable[object]) -> bool:
    return not any(bool(getattr(meld, "opened", True)) for meld in melds)


def _river_types(observation: object) -> set[int]:
    seat = int(getattr(observation, "player_id"))
    rows = getattr(observation, "discards", ()) or ()
    return {int(tile) // 4 for tile in rows[seat]} if len(rows) > seat else set()


def _score_total(result: object, tsumo: bool, dealer: bool) -> int:
    if not bool(getattr(result, "is_win", False)):
        return 0
    if not tsumo:
        return int(getattr(result, "ron_agari", 0))
    ko = int(getattr(result, "tsumo_agari_ko", 0))
    oya = int(getattr(result, "tsumo_agari_oya", 0))
    return 3 * ko if dealer else oya + 2 * ko


@dataclass(frozen=True, slots=True)
class RuleState:
    effective_shanten: int
    waits: tuple[int, ...] = ()
    live_ron: int = 0
    live_tsumo: int = 0
    ron_points: int = 0
    tsumo_points: int = 0
    has_yaku: bool = False
    riichi_route: bool = False
    open_no_yaku: bool = False
    furiten: bool = False


def _rule_state(
    observation: object,
    hand: list[int],
    melds: list[object],
    structural_shanten: int,
    remaining: np.ndarray,
) -> RuleState:
    if structural_shanten != 0:
        return RuleState(structural_shanten)
    try:
        from riichienv import Conditions, HandEvaluator

        evaluator = HandEvaluator(sorted(hand), melds)
        waits = tuple(sorted({int(wait) for wait in evaluator.get_waits()}))
        if not waits:
            return RuleState(1)
        seat = int(getattr(observation, "player_id"))
        closed = _is_closed(melds)
        riichi_now = bool((getattr(observation, "riichi_declared", ()) or (False,) * 4)[seat])
        furiten = bool(set(waits) & _river_types(observation))
        ron_live = tsumo_live = ron_points = tsumo_points = 0
        intrinsic_yaku = False
        riichi_route = False
        dora = [int(tile) for tile in getattr(observation, "dora_indicators", ()) or ()]
        base = dict(
            player_wind=(seat - int(getattr(observation, "oya", 0))) % 4,
            round_wind=int(getattr(observation, "round_wind", 0)),
            honba=int(getattr(observation, "honba", 0)),
            riichi_sticks=int(getattr(observation, "riichi_sticks", 0)),
        )
        dealer = seat == int(getattr(observation, "oya", 0))
        for wait in waits:
            copies = max(0, int(remaining[wait]))
            # Copy 0 is the red five for 5m/5p/5s; use a normal copy for a
            # conservative public score estimate.
            win_tile = wait * 4 + int(wait in {4, 13, 22})
            ron = evaluator.calc(win_tile, dora, Conditions(tsumo=False, riichi=riichi_now, **base))
            tsumo = evaluator.calc(win_tile, dora, Conditions(tsumo=True, riichi=riichi_now, **base))
            intrinsic_yaku |= bool(ron.is_win or tsumo.is_win)
            if ron.is_win and not furiten:
                ron_live += copies
                ron_points = max(ron_points, _score_total(ron, False, dealer))
            if tsumo.is_win:
                tsumo_live += copies
                tsumo_points = max(tsumo_points, _score_total(tsumo, True, dealer))
            if closed and not riichi_now:
                reach = evaluator.calc(win_tile, dora, Conditions(tsumo=False, riichi=True, **base))
                riichi_route |= bool(reach.is_win)
        open_no_yaku = not closed and not intrinsic_yaku
        effective = 1 if open_no_yaku else 0
        return RuleState(
            effective, waits, ron_live, tsumo_live, ron_points, tsumo_points,
            intrinsic_yaku, riichi_route, open_no_yaku, furiten,
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        # Unit-test doubles and old serialized observations may not carry the
        # full evaluator context.  They retain structural semantics.
        return RuleState(structural_shanten)


def _dora_types(observation: object) -> set[int]:
    result: set[int] = set()
    for raw in getattr(observation, "dora_indicators", ()) or ():
        tile = int(raw) // 4
        if tile < 27:
            base = tile // 9 * 9
            result.add(base + (tile - base + 1) % 9)
        elif tile <= 30:
            result.add(27 + (tile - 27 + 1) % 4)
        else:
            result.add(31 + (tile - 31 + 1) % 3)
    return result


@dataclass(frozen=True, slots=True)
class Candidate:
    action: object
    action_id: int
    structural_shanten: int
    effective_shanten: int
    ukeire: int
    live_ron: int = 0
    live_tsumo: int = 0
    ron_points: int = 0
    tsumo_points: int = 0
    has_yaku: bool = False
    riichi_route: bool = False
    open_no_yaku: bool = False
    furiten: bool = False
    closed: bool = True
    preserve_dora: int = 0
    preserve_red: int = 0
    genbutsu_coverage: int = 0
    four_visible: bool = False

    @property
    def rank(self) -> tuple[int, ...]:
        """Smaller tuple is better; ordering is fixed and deterministic."""
        return (
            self.structural_shanten,
            self.effective_shanten,
            -self.live_ron,
            -self.live_tsumo,
            -self.ukeire,
            -max(self.ron_points, self.tsumo_points),
            -self.preserve_dora,
            -self.preserve_red,
        )


@dataclass(frozen=True, slots=True)
class DecisionAnalysis:
    decision: object
    actions: tuple[object, ...]
    candidates: tuple[Candidate, ...]
    best_rank: tuple[int, ...] | None
    teacher_mask: np.ndarray
    selected_kind: str

    def candidate_for(self, action: object) -> Candidate | None:
        key = action_key(action)
        return next((candidate for candidate in self.candidates if action_key(candidate.action) == key), None)

    def selected_regrets(self, action: object) -> tuple[float, float]:
        kind = action_kind(action)
        candidate = self.candidate_for(action)
        if candidate is None or self.best_rank is None:
            return 0.0, 0.0
        regret = _bounded_regret(candidate.rank, self.best_rank)
        if kind == "dahai":
            return regret, 0.0
        if kind in {"none", "pass", "chi", "pon", "daiminkan"}:
            return 0.0, regret
        return 0.0, 0.0


def _bounded_regret(rank: tuple[int, ...], best: tuple[int, ...]) -> float:
    if rank == best:
        return 0.0
    if rank[0] != best[0]:
        return float(max(-1.0, -0.5 * (rank[0] - best[0])))
    if rank[1] != best[1]:
        return float(max(-1.0, -0.5 * (rank[1] - best[1])))
    # Lexicographically worse within the same shanten is a softer regret.
    for value, optimum in zip(rank[2:], best[2:], strict=True):
        if value != optimum:
            denominator = max(abs(optimum), 1)
            return float(np.clip(-0.25 * abs(value - optimum) / denominator, -0.5, -0.02))
    return 0.0


def _candidate(
    decision: object,
    action: object,
    aid: int,
    hand: list[int],
    melds: list[object],
    analysis: HandAnalysis,
    remaining: np.ndarray,
    public: object | None,
) -> Candidate:
    observation = decision.observation
    discarded = getattr(action, "tile", None) if action_kind(action) == "dahai" else None
    rules = _rule_state(observation, hand, melds, int(analysis.shanten), remaining)
    kept_tiles = [
        *hand,
        *(int(tile) for meld in melds for tile in (getattr(meld, "tiles", ()) or ())),
    ]
    kept = _counts(kept_tiles)
    dora = _dora_types(observation)
    preserve_dora = sum(int(kept[tile]) for tile in dora)
    preserve_red = sum(int(tile in {16, 52, 88}) for tile in kept_tiles)
    tile_type = -1 if discarded is None else int(discarded) // 4
    coverage = 0
    four_visible = False
    if tile_type >= 0:
        four_visible = int(remaining[tile_type]) == 0
        if public is not None:
            coverage = int(public.genbutsu_coverage(decision.env_index, tile_type))
    return Candidate(
        action, aid, int(analysis.shanten), rules.effective_shanten,
        remaining_ukeire(analysis.improving_mask, remaining),
        rules.live_ron, rules.live_tsumo, rules.ron_points, rules.tsumo_points,
        rules.has_yaku, rules.riichi_route, rules.open_no_yaku, rules.furiten,
        _is_closed(melds), preserve_dora, preserve_red, coverage, four_visible,
    )


def _make_call_meld(action: object) -> object | None:
    try:
        from riichienv import Meld, MeldType

        kind = action_kind(action)
        meld_type = {
            "chi": MeldType.Chi,
            "pon": MeldType.Pon,
            "daiminkan": MeldType.Daiminkan,
        }.get(kind)
        if meld_type is None:
            return None
        called = int(getattr(action, "tile"))
        return Meld(meld_type, sorted([*consumed_tiles(action), called]), True)
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def _kuikae_forbidden_types(action: object) -> set[int]:
    """Mirror RiichiEnv's standard post-chi/pon kuikae rule."""
    kind = action_kind(action)
    called = getattr(action, "tile", None)
    if called is None or kind not in {"chi", "pon"}:
        return set()
    called_type = int(called) // 4
    forbidden = {called_type}
    if kind == "chi":
        used = sorted(tile // 4 for tile in consumed_tiles(action))
        if len(used) == 2 and used == [called_type + 1, called_type + 2]:
            if called_type % 9 <= 5:
                forbidden.add(called_type + 3)
        elif (
            len(used) == 2
            and called_type >= 2
            and used == [called_type - 2, called_type - 1]
            and called_type % 9 >= 3
        ):
            forbidden.add(called_type - 3)
    return forbidden


class DecisionAnalysisBatch:
    """One step-local source of truth for reward, tokens, metrics and teachers."""

    def __init__(self, rows: dict[int, DecisionAnalysis]) -> None:
        self._rows = rows

    @classmethod
    def build(
        cls,
        decisions: Iterable[object],
        *,
        analyzer: EfficiencyAnalyzer,
        public: object | None = None,
    ) -> "DecisionAnalysisBatch":
        decisions = list(decisions)
        remaining_by_row = [
            public_remaining(decision.observation) for decision in decisions
        ]
        jobs: list[tuple[int, object, int, list[int], list[object]]] = []
        hands: list[np.ndarray] = []
        opened: list[int] = []
        actions_by_row: list[tuple[object, ...]] = []
        for row, decision in enumerate(decisions):
            observation = decision.observation
            actions = tuple(observation.legal_actions())
            actions_by_row.append(actions)
            hand = _physical_hand(observation)
            melds = _own_melds(observation)
            call_window = any(action_kind(action) in {"chi", "pon", "daiminkan"} for action in actions)
            for action in actions:
                kind = action_kind(action)
                aid = action_id(action, observation)
                if aid is None:
                    continue
                if kind == "dahai":
                    post = hand.copy()
                    try:
                        post.remove(int(getattr(action, "tile")))
                    except ValueError:
                        continue
                    jobs.append((row, action, aid, post, melds))
                    hands.append(_counts(post))
                    opened.append(len(melds))
                elif call_window and kind in {"none", "pass"}:
                    jobs.append((row, action, aid, hand, melds))
                    hands.append(_counts(hand))
                    opened.append(len(melds))
                elif kind in {"chi", "pon"}:
                    post_call = hand.copy()
                    try:
                        for tile in consumed_tiles(action):
                            post_call.remove(tile)
                    except ValueError:
                        continue
                    new_meld = _make_call_meld(action)
                    if new_meld is None:
                        continue
                    call_melds = [*melds, new_meld]
                    # The forced discard is represented by the best resulting
                    # hand, but the candidate action remains the call itself.
                    forbidden = _kuikae_forbidden_types(action)
                    for discard in sorted(set(post_call)):
                        if discard // 4 in forbidden:
                            continue
                        post = post_call.copy()
                        post.remove(discard)
                        jobs.append((row, action, aid, post, call_melds))
                        hands.append(_counts(post))
                        opened.append(len(call_melds))
                elif kind == "daiminkan":
                    post = hand.copy()
                    try:
                        for tile in consumed_tiles(action):
                            post.remove(tile)
                    except ValueError:
                        continue
                    new_meld = _make_call_meld(action)
                    if new_meld is None:
                        continue
                    call_melds = [*melds, new_meld]
                    jobs.append((row, action, aid, post, call_melds))
                    hands.append(_counts(post))
                    opened.append(len(call_melds))

        values = analyzer.analyze(hands, opened) if hands else []
        grouped: list[dict[tuple[str, int, tuple[int, ...]], list[Candidate]]] = [dict() for _ in decisions]
        for (row, action, aid, hand, melds), structural in zip(jobs, values, strict=True):
            # Moving a tile from the concealed hand into a discard or meld
            # does not change how many copies are available.  Always derive
            # remaining counts from the original observation hand so simulated
            # consumed/forced-discard tiles cannot reappear as ukeire.
            remaining = remaining_by_row[row]
            candidate = _candidate(decisions[row], action, aid, hand, melds, structural, remaining, public)
            grouped[row].setdefault(action_key(action), []).append(candidate)

        rows: dict[int, DecisionAnalysis] = {}
        for row, decision in enumerate(decisions):
            collapsed: list[Candidate] = []
            for variants in grouped[row].values():
                collapsed.append(min(variants, key=lambda item: item.rank))
            candidates = tuple(collapsed)
            best = min((candidate.rank for candidate in candidates), default=None)
            teacher = np.zeros(NUM_ACTIONS, dtype=np.bool_)
            if best is not None:
                for candidate in candidates:
                    if candidate.rank == best:
                        teacher[candidate.action_id] = True
            kinds = {action_kind(action) for action in actions_by_row[row]}
            excluded = kinds & {"reach", "ankan", "kakan", "hora", "ron", "tsumo"}
            supervised = (
                "" if excluded else
                ("dahai" if "dahai" in kinds else
                 ("call" if kinds & {"chi", "pon", "daiminkan"} else ""))
            )
            if not supervised:
                teacher.fill(False)
            rows[id(decision)] = DecisionAnalysis(
                decision, actions_by_row[row], candidates, best, teacher, supervised,
            )
        return cls(rows)

    def for_decision(self, decision: object) -> DecisionAnalysis:
        row = self._rows.get(id(decision))
        if row is not None:
            return row
        return DecisionAnalysis(
            decision, tuple(decision.observation.legal_actions()), (), None,
            np.zeros(NUM_ACTIONS, dtype=np.bool_), "",
        )

    def teacher_masks(self, decisions: Iterable[object]) -> np.ndarray:
        return np.stack([self.for_decision(decision).teacher_mask for decision in decisions])

    def candidate_tokens(
        self,
        decisions: Iterable[object],
        legal_masks: np.ndarray,
        *,
        public: object | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Encode one deterministic public candidate token per legal action."""
        factor_rows: list[np.ndarray] = []
        numeric_rows: list[np.ndarray] = []
        for row_index, decision in enumerate(decisions):
            analysis = self.for_decision(decision)
            by_id = {candidate.action_id: candidate for candidate in analysis.candidates}
            legal_ids = np.flatnonzero(legal_masks[row_index])
            factors = np.zeros((len(legal_ids), 10), dtype=np.uint8)
            numeric = np.zeros((len(legal_ids), 8), dtype=np.float32)
            threat_count = sum(bool(value) for value in getattr(decision.observation, "riichi_declared", ()))
            best_shanten = min((item.structural_shanten for item in analysis.candidates), default=0)
            for token, aid in enumerate(legal_ids):
                candidate = by_id.get(int(aid))
                factors[token, 0] = 7  # candidate-token segment
                factors[token, 1] = _action_type_code(int(aid))
                factors[token, 2] = min(int(aid) + 1, 255)
                factors[token, 3] = min(threat_count, 3)
                if candidate is None:
                    continue
                factors[token, 4] = min(candidate.structural_shanten + 1, 7)
                factors[token, 5] = min(candidate.effective_shanten + 1, 15)
                factors[token, 6] = int(candidate.has_yaku) + 2 * int(candidate.riichi_route)
                factors[token, 7] = (
                    int(candidate.open_no_yaku)
                    + 2 * int(candidate.furiten)
                    + 4 * int(candidate.closed)
                )
                # Separate 1k-point ron/tsumo buckets packed into one public
                # categorical channel (low/high nibble).
                factors[token, 8] = (
                    min(candidate.ron_points // 1000, 15)
                    + 16 * min(candidate.tsumo_points // 1000, 15)
                )
                factors[token, 9] = int(candidate.four_visible)
                numeric[token] = (
                    candidate.structural_shanten / 6.0,
                    candidate.effective_shanten / 6.0,
                    (candidate.structural_shanten - best_shanten) / 3.0,
                    candidate.ukeire / 40.0,
                    candidate.live_ron / 16.0,
                    candidate.live_tsumo / 16.0,
                    max(candidate.ron_points, candidate.tsumo_points) / 12000.0,
                    candidate.genbutsu_coverage / 3.0,
                )
            factor_rows.append(factors)
            numeric_rows.append(numeric)
        return factor_rows, numeric_rows
