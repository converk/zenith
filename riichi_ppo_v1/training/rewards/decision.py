"""Rule-aware public decision analysis shared by rollout, teachers and opponents.

The analyzer deliberately uses only fields visible in the acting player's
``Observation``.  Structural shanten is batched through ``EfficiencyAnalyzer``;
exact waits, yaku and score estimates are evaluated only for structural-tenpai
candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import OrderedDict
import json
import time
from typing import Iterable

import numpy as np

from ...model.validation import TILE37, _chi_pairs, _pon_pairs
from .efficiency import EfficiencyAnalyzer, HandAnalysis, remaining_ukeire
from ...model.feature_schema import (
    DECISION_ANALYSIS_VERSION,
    RUST_ANALYSIS_VERSION,
    encode_shanten,
)
from ...model.dora import dora_type_multiplicities

NUM_ACTIONS = 241
SCHEMA_VERSION = DECISION_ANALYSIS_VERSION
_RULE_CACHE_CAPACITY = 131_072
_RULE_CACHE: OrderedDict[tuple[object, ...], "RuleState"] = OrderedDict()


def _unit(value: int | float, scale: int | float, *, signed: bool = False) -> float:
    lower = -1.0 if signed else 0.0
    return float(np.clip(float(value) / float(scale), lower, 1.0))


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
    result = [int(value) for value in values]
    kind = action_kind(action)
    expected = {"chi": 2, "pon": 2, "daiminkan": 3}.get(kind)
    called = getattr(action, "tile", None)
    if expected is not None and len(result) == expected + 1 and called is not None:
        # Offline MjaiReplay actions retain the called tile in consume_tiles;
        # live RiichiEnv actions expose only tiles taken from the actor's hand.
        # Normalize both representations before simulating the post-call hand.
        try:
            result.remove(int(called))
        except ValueError:
            pass
    return tuple(result)


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
    expected_consumed = {"chi": 2, "pon": 2, "daiminkan": 3}.get(kind)
    consumed_row = row.get("consumed")
    if (
        expected_consumed is not None
        and isinstance(consumed_row, list)
        and len(consumed_row) == expected_consumed + 1
    ):
        consumed_row = list(consumed_row)
        try:
            consumed_row.remove(row.get("pai"))
        except ValueError:
            return None
        row["consumed"] = consumed_row
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
    damaten_ron_yaku: bool = False
    can_tsumo: bool = False
    has_legal_route: bool = False
    riichi_route: bool = False
    open_no_yaku: bool = False
    furiten: bool = False
    ron_value_sum: int = 0
    tsumo_value_sum: int = 0
    riichi_ron_points: int = 0
    riichi_ron_value_sum: int = 0


def _rule_state(
    observation: object,
    hand: list[int],
    melds: list[object],
    structural_shanten: int,
    remaining: np.ndarray,
    *,
    legacy: bool = False,
) -> RuleState:
    if structural_shanten != 0:
        return RuleState(structural_shanten)
    seat = int(getattr(observation, "player_id"))
    meld_key = tuple(
        (
            str(getattr(getattr(meld, "meld_type", None), "name", getattr(meld, "meld_type", ""))),
            tuple(sorted(int(tile) for tile in (getattr(meld, "tiles", ()) or ()))),
            bool(getattr(meld, "opened", True)),
        )
        for meld in melds
    )
    cache_key = (
        tuple(sorted(int(tile) for tile in hand)), meld_key, tuple(int(value) for value in remaining),
        tuple(sorted(_river_types(observation))),
        tuple(int(tile) for tile in (getattr(observation, "dora_indicators", ()) or ())),
        seat, int(getattr(observation, "oya", 0)), int(getattr(observation, "round_wind", 0)),
        int(getattr(observation, "honba", 0)), int(getattr(observation, "riichi_sticks", 0)),
        bool((getattr(observation, "riichi_declared", ()) or (False,) * 4)[seat]),
        False if legacy else bool(getattr(observation, "missed_agari_doujun", False)),
        False if legacy else bool(getattr(observation, "missed_agari_riichi", False)),
        legacy,
    )
    cached = _RULE_CACHE.get(cache_key)
    if cached is not None:
        _RULE_CACHE.move_to_end(cache_key)
        return cached
    try:
        from riichienv import Conditions, HandEvaluator

        evaluator = HandEvaluator(sorted(hand), melds)
        waits = tuple(sorted({int(wait) for wait in evaluator.get_waits()}))
        if not waits:
            return RuleState(1)
        closed = _is_closed(melds)
        riichi_now = bool((getattr(observation, "riichi_declared", ()) or (False,) * 4)[seat])
        furiten = bool(set(waits) & _river_types(observation))
        if not legacy:
            furiten = bool(
                furiten
                or getattr(observation, "missed_agari_doujun", False)
                or getattr(observation, "missed_agari_riichi", False)
            )
        ron_live = tsumo_live = ron_points = tsumo_points = 0
        ron_value_sum = tsumo_value_sum = 0
        riichi_ron_points = riichi_ron_value_sum = 0
        damaten_ron_yaku = False
        current_ron_yaku = False
        can_tsumo = False
        legacy_intrinsic_yaku = False
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
            damaten_ron = evaluator.calc(
                win_tile, dora, Conditions(tsumo=False, riichi=False, **base),
            )
            current_ron_yaku |= bool(ron.is_win)
            damaten_ron_yaku |= bool(damaten_ron.is_win)
            can_tsumo |= bool(tsumo.is_win)
            legacy_intrinsic_yaku |= bool(ron.is_win or tsumo.is_win)
            if ron.is_win and not furiten:
                ron_live += copies
                points = _score_total(ron, False, dealer)
                ron_points = max(ron_points, points)
                ron_value_sum += copies * points
            if tsumo.is_win:
                tsumo_live += copies
                points = _score_total(tsumo, True, dealer)
                tsumo_points = max(tsumo_points, points)
                tsumo_value_sum += copies * points
            if closed and not riichi_now:
                reach = evaluator.calc(win_tile, dora, Conditions(tsumo=False, riichi=True, **base))
                riichi_route |= bool(reach.is_win)
                if reach.is_win and not furiten:
                    points = _score_total(reach, False, dealer)
                    riichi_ron_points = max(riichi_ron_points, points)
                    riichi_ron_value_sum += copies * points
        encoded_yaku = legacy_intrinsic_yaku if legacy else damaten_ron_yaku
        open_no_yaku = not closed and not (damaten_ron_yaku or can_tsumo)
        effective = 1 if open_no_yaku else 0
        has_legal_route = current_ron_yaku or can_tsumo or riichi_route
        result = RuleState(
            effective, waits, ron_live, tsumo_live, ron_points, tsumo_points,
            encoded_yaku, can_tsumo, has_legal_route,
            riichi_route, open_no_yaku, furiten,
            ron_value_sum, tsumo_value_sum,
            riichi_ron_points, riichi_ron_value_sum,
        )
        _RULE_CACHE[cache_key] = result
        _RULE_CACHE.move_to_end(cache_key)
        while len(_RULE_CACHE) > _RULE_CACHE_CAPACITY:
            _RULE_CACHE.popitem(last=False)
        return result
    except ImportError:
        # Keep a minimal audit fallback for environments where RiichiEnv has
        # not been installed. Evaluator/data errors are deliberately fatal in
        # schema 13 instead of being converted to plausible-looking zeros.
        return RuleState(structural_shanten)


def _dora_types(observation: object) -> dict[int, int]:
    """Return public dora types with one count per equal indicator."""
    return dora_type_multiplicities(
        getattr(observation, "dora_indicators", ()) or ()
    )


def _defense_feature_batch(
    observation: object,
    legal_ids: np.ndarray,
    action_by_id: dict[int, object],
    remaining: np.ndarray,
    opponents: tuple[int, int, int],
    offline_pass_masks: list[int],
    features_by_id: dict[int, "Candidate"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the schema-13 Rust kernel, retaining an exact audit fallback."""
    river_rows = getattr(observation, "discards", ()) or ()
    river_masks = np.zeros((len(legal_ids), 3), dtype=np.uint64)
    passed_masks = np.zeros_like(river_masks)
    for column, opponent in enumerate(opponents):
        mask = 0
        for tile in river_rows[opponent] if len(river_rows) > opponent else ():
            mask |= 1 << (int(tile) // 4)
        river_masks[:, column] = mask
        passed_masks[:, column] = int(offline_pass_masks[opponent])
    discard_types = np.full(len(legal_ids), -1, dtype=np.int16)
    for row, aid_value in enumerate(legal_ids):
        action = action_by_id.get(int(aid_value))
        if action is not None and action_kind(action) == "dahai" and getattr(action, "tile", None) is not None:
            discard_types[row] = int(getattr(action, "tile")) // 4
    current_hand = _counts(_physical_hand(observation))
    current_opened = len(_own_melds(observation))
    hands = np.zeros((len(legal_ids), 34), dtype=np.uint8)
    opened = np.zeros(len(legal_ids), dtype=np.uint8)
    for row, aid_value in enumerate(legal_ids):
        feature = features_by_id.get(int(aid_value))
        if feature is not None and len(feature.concealed_counts) == 34:
            hands[row] = np.asarray(feature.concealed_counts, dtype=np.uint8)
            opened[row] = int(feature.open_meld_count)
        else:
            hands[row] = current_hand
            opened[row] = current_opened
    try:
        import riichi
    except ImportError as exc:
        raise RuntimeError(
            "schema-13 feature analysis requires the versioned riichi Rust extension"
        ) from exc
    kernel = getattr(riichi, "analyze_features", None)
    if kernel is None:
        raise RuntimeError("installed riichi extension does not provide analyze_features")
    result = kernel(
        np.ascontiguousarray(legal_ids, dtype=np.uint16),
        np.ascontiguousarray(hands),
        np.ascontiguousarray(opened),
        np.repeat(np.asarray(remaining, dtype=np.uint8)[None], len(legal_ids), axis=0),
        discard_types,
        river_masks,
        passed_masks,
    )
    if int(result.analysis_version) != RUST_ANALYSIS_VERSION:
        raise RuntimeError(
            f"installed riichi feature-analysis version is not {RUST_ANALYSIS_VERSION}"
        )
    rust_shanten = np.asarray(result.shanten, dtype=np.int16)
    rust_masks = np.asarray(result.improving_type_mask, dtype=np.uint64)
    rust_ukeire = np.asarray(result.ukeire, dtype=np.int64)
    for row, aid_value in enumerate(legal_ids):
        feature = features_by_id.get(int(aid_value))
        if feature is None:
            continue
        expected = (
            feature.structural_shanten,
            feature.standard_shanten,
            feature.seven_pairs_shanten,
            feature.thirteen_orphans_shanten,
        )
        if tuple(int(value) for value in rust_shanten[row]) != expected:
            raise RuntimeError(f"Rust/Python shanten parity failure for action {int(aid_value)}")
        if int(rust_masks[row]) != int(feature.improving_mask):
            raise RuntimeError(f"Rust/Python improving-mask parity failure for action {int(aid_value)}")
        if int(rust_ukeire[row]) != int(feature.ukeire):
            raise RuntimeError(f"Rust/Python ukeire parity failure for action {int(aid_value)}")
    return (
        np.asarray(result.defense, dtype=np.uint8),
        np.asarray(result.categorical, dtype=np.uint8),
        np.asarray(result.numeric, dtype=np.float32),
    )


def _defense_feature_reference(
    legal_ids: np.ndarray,
    remaining: np.ndarray,
    discard_types: np.ndarray,
    river_masks: np.ndarray,
    passed_masks: np.ndarray,
    hand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Auditable scalar reference for full Rust defense-row parity tests."""
    defense = np.zeros((len(legal_ids), 5), dtype=np.uint8)
    for row, tile_value in enumerate(discard_types):
        tile = int(tile_value)
        if tile < 0:
            continue
        for column in range(3):
            river = int(river_masks[row, column])
            if river & (1 << tile):
                defense[row, 0] |= 1 << column
            if tile < 27:
                rank = tile % 9
                anchors = [
                    anchor for anchor in (
                        tile - 3 if rank >= 3 else None,
                        tile + 3 if rank <= 5 else None,
                    ) if anchor is not None
                ]
                if anchors and all(river & (1 << anchor) for anchor in anchors):
                    defense[row, 1] |= 1 << column
            if int(passed_masks[row, column]) & (1 << tile):
                defense[row, 4] |= 1 << column
        visible = 4 - int(remaining[tile])
        if tile < 27:
            rank, base = tile % 9, tile - tile % 9
            adjacent_visible = max(
                (4 - int(remaining[base + other])
                 for other in range(9)
                 if other != rank and abs(other - rank) <= 2),
                default=0,
            )
            defense[row, 2] = 2 if adjacent_visible >= 4 else (1 if adjacent_visible >= 3 else 0)
        defense[row, 3] = max(visible - int(hand[tile]), 0) if tile >= 27 else 0
    categorical = np.zeros((len(legal_ids), 10), dtype=np.uint8)
    numeric = np.zeros((len(legal_ids), 8), dtype=np.float32)
    for row, (aid_value, tile_value) in enumerate(zip(legal_ids, discard_types, strict=True)):
        aid, tile = int(aid_value), int(tile_value)
        categorical[row, (0, 1, 2, 9)] = (7, _action_type_code(aid), min(aid + 1, 255), 2)
        categorical[row, 4:7] = defense[row, :3]
        categorical[row, 7] = defense[row, 4]
        if tile >= 0:
            numeric[row, :4] = (
                int(defense[row, 0]).bit_count() / 3.0,
                int(defense[row, 1]).bit_count() / 3.0,
                (4 - int(remaining[tile])) / 4.0,
                int(defense[row, 3]) / 4.0,
            )
    return defense, categorical, numeric


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
    can_tsumo: bool = False
    has_legal_route: bool = False
    riichi_route: bool = False
    open_no_yaku: bool = False
    furiten: bool = False
    closed: bool = True
    preserve_dora: int = 0
    preserve_red: int = 0
    genbutsu_coverage: int = 0
    four_visible: bool = False
    # Heuristic-only opportunity-weighted values.  They deliberately do not
    # participate in ``rank`` so SFT teacher masks and candidate-token schemas
    # remain stable.
    ron_value_sum: int = 0
    tsumo_value_sum: int = 0
    riichi_ron_points: int = 0
    riichi_ron_value_sum: int = 0
    improving_mask: int = 0
    standard_shanten: int = 0
    seven_pairs_shanten: int = 0
    thirteen_orphans_shanten: int = 0
    wait_count: int = 0
    legal_discard_count: int = 0
    structure_known_mask: int = 0
    structure_break_mask: int = 0
    foregone_shanten_improvement: int = 0
    foregone_ukeire_improvement: int = 0
    foregone_win: bool = False
    concealed_counts: tuple[int, ...] = ()
    open_meld_count: int = 0

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
        observation = getattr(self.decision, "observation", None)
        mapped = action_id(action, observation) if observation is not None else None
        if mapped is not None:
            candidate = next(
                (item for item in self.candidates if item.action_id == mapped), None,
            )
            if candidate is not None:
                return candidate
        key = action_key(action)
        return next((candidate for candidate in self.candidates if action_key(candidate.action) == key), None)

    def selected_efficiency_reward(self, action: object) -> float:
        """Small dense discard-efficiency reward (E4); 0 for non-discards.

        Mirrors ``efficiency.efficiency_reward``: shanten regression is -1.0,
        otherwise -0.25 * relative ukeire loss versus the best same-shanten
        candidate. Only tile-carrying discard decisions (dahai/riichi/reach)
        receive the signal.
        """
        if action_kind(action) not in {"dahai", "riichi", "reach"}:
            return 0.0
        candidate = self.candidate_for(action)
        if candidate is None or self.best_rank is None:
            return 0.0
        shanten_gap = int(candidate.structural_shanten) - int(self.best_rank[0])
        if shanten_gap > 0:
            return -1.0
        if shanten_gap < 0:
            return 0.0
        best_ukeire = max(
            (
                int(item.ukeire)
                for item in self.candidates
                if int(item.structural_shanten) == int(self.best_rank[0])
            ),
            default=0,
        )
        loss = max(0, int(best_ukeire) - int(candidate.ukeire)) / max(int(best_ukeire), 1)
        return -0.25 * float(loss)

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
    *,
    legacy: bool = False,
) -> Candidate:
    observation = decision.observation
    discarded = getattr(action, "tile", None) if action_kind(action) == "dahai" else None
    rules = _rule_state(
        observation, hand, melds, int(analysis.shanten), remaining, legacy=legacy,
    )
    kept_tiles = [
        *hand,
        *(int(tile) for meld in melds for tile in (getattr(meld, "tiles", ()) or ())),
    ]
    kept = _counts(kept_tiles)
    dora = _dora_types(observation)
    preserve_dora = sum(int(kept[tile]) * multiplier for tile, multiplier in dora.items())
    preserve_red = sum(int(tile in {16, 52, 88}) for tile in kept_tiles)
    tile_type = -1 if discarded is None else int(discarded) // 4
    coverage = 0
    four_visible = False
    if tile_type >= 0:
        four_visible = int(remaining[tile_type]) == 0
        if public is not None:
            coverage = int(public.genbutsu_coverage(decision.env_index, tile_type))
    structure_known = structure_break = 0
    if discarded is not None:
        from ...model.feature_schema import BREAK_MELD, BREAK_PAIR, BREAK_RYANMEN
        before = _counts(_physical_hand(observation))
        before_count = int(before[tile_type])
        structure_known = BREAK_MELD | BREAK_PAIR | BREAK_RYANMEN
        if before_count == 2:
            structure_break |= BREAK_PAIR
        if before_count == 3:
            structure_break |= BREAK_MELD
        if tile_type < 27 and before_count == 1:
            rank = tile_type % 9
            base = tile_type - rank
            for start in range(max(0, rank - 2), min(rank, 6) + 1):
                sequence = [base + start, base + start + 1, base + start + 2]
                if tile_type in sequence and all(before[value] > 0 for value in sequence if value != tile_type):
                    structure_break |= BREAK_MELD
            for left in range(1, 7):
                pair = {base + left, base + left + 1}
                if tile_type in pair and any(before[value] > 0 for value in pair if value != tile_type):
                    structure_break |= BREAK_RYANMEN
    return Candidate(
        action, aid, int(analysis.shanten), rules.effective_shanten,
        remaining_ukeire(analysis.improving_mask, remaining),
        rules.live_ron, rules.live_tsumo, rules.ron_points, rules.tsumo_points,
        rules.damaten_ron_yaku, rules.can_tsumo, rules.has_legal_route,
        rules.riichi_route, rules.open_no_yaku, rules.furiten,
        _is_closed(melds), preserve_dora, preserve_red, coverage, four_visible,
        rules.ron_value_sum, rules.tsumo_value_sum,
        rules.riichi_ron_points, rules.riichi_ron_value_sum,
        int(analysis.improving_mask),
        int(analysis.standard_shanten if analysis.standard_shanten is not None else analysis.shanten),
        int(analysis.seven_pairs_shanten if analysis.seven_pairs_shanten is not None else analysis.shanten),
        int(analysis.thirteen_orphans_shanten if analysis.thirteen_orphans_shanten is not None else analysis.shanten),
        len(rules.waits),
        0, structure_known, structure_break,
        0, 0, False, tuple(int(value) for value in _counts(hand)), len(melds),
    )


def _make_call_meld(action: object) -> object | None:
    try:
        from riichienv import Meld, MeldType

        kind = action_kind(action)
        meld_type = {
            "chi": MeldType.Chi,
            "pon": MeldType.Pon,
            "daiminkan": MeldType.Daiminkan,
            "ankan": MeldType.Ankan,
            "kakan": MeldType.Kakan,
        }.get(kind)
        if meld_type is None:
            return None
        called_raw = getattr(action, "tile", None)
        values = list(consumed_tiles(action))
        if called_raw is not None and int(called_raw) not in values:
            values.append(int(called_raw))
        return Meld(meld_type, sorted(values), kind != "ankan")
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def _make_kan_meld(kind: str, tiles: Iterable[int]) -> object | None:
    """Build a complete four-tile kan meld for rule/value simulation."""
    try:
        from riichienv import Meld, MeldType

        meld_type = {"ankan": MeldType.Ankan, "kakan": MeldType.Kakan}[kind]
        values = sorted(int(tile) for tile in tiles)
        if len(values) != 4 or len({tile // 4 for tile in values}) != 1:
            return None
        return Meld(meld_type, values, kind != "ankan")
    except (ImportError, KeyError, TypeError, ValueError):
        return None


def _upgrade_kakan_melds(
    melds: list[object], action: object, *, added_tile: int | None = None,
) -> list[object] | None:
    tile = getattr(action, "tile", None)
    values = list(consumed_tiles(action))
    physical = (
        int(added_tile) if added_tile is not None else
        (int(tile) if tile is not None else (int(values[-1]) if values else -1))
    )
    tile_type = physical // 4 if physical >= 0 else -1
    result = list(melds)
    for index, meld in enumerate(result):
        kind = str(getattr(getattr(meld, "meld_type", None), "name", getattr(meld, "meld_type", ""))).lower()
        existing = [int(value) for value in (getattr(meld, "tiles", ()) or ())]
        meld_types = {value // 4 for value in existing}
        if "pon" in kind and tile_type in meld_types:
            upgraded = _make_kan_meld("kakan", [*existing, physical])
            if upgraded is None:
                return None
            result[index] = upgraded
            return result
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

    def __init__(self, rows: dict[int, DecisionAnalysis], analyzer: EfficiencyAnalyzer | None = None, public: object | None = None) -> None:
        self._rows = rows
        self._analyzer = analyzer or EfficiencyAnalyzer()
        self._public = public

    @classmethod
    def build(
        cls,
        decisions: Iterable[object],
        *,
        analyzer: EfficiencyAnalyzer,
        public: object | None = None,
        profiler: object | None = None,
        _legacy_v11: bool = False,
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
            response_kinds = (
                {"chi", "pon", "daiminkan"}
                if _legacy_v11
                else {"chi", "pon", "daiminkan", "hora", "ron"}
            )
            response_window = any(
                action_kind(action) in response_kinds
                for action in actions
            )
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
                elif response_window and kind in {"none", "pass"}:
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
                elif not _legacy_v11 and kind == "ankan":
                    post = hand.copy()
                    values = list(consumed_tiles(action))
                    tile = getattr(action, "tile", None)
                    tile_type = (
                        values[0] // 4 if values else
                        (int(tile) // 4 if tile is not None else -1)
                    )
                    concealed_values = [
                        physical for physical in hand if physical // 4 == tile_type
                    ]
                    if len(concealed_values) != 4:
                        raise RuntimeError("legal ankan does not identify four concealed copies")
                    try:
                        for physical in concealed_values:
                            post.remove(physical)
                    except ValueError:
                        raise RuntimeError("legal ankan tiles are absent from the concealed hand") from None
                    new_meld = _make_kan_meld("ankan", concealed_values)
                    if new_meld is None:
                        raise RuntimeError("legal ankan could not be normalized to one tile type")
                    call_melds = [*melds, new_meld]
                    jobs.append((row, action, aid, post, call_melds))
                    hands.append(_counts(post))
                    opened.append(len(call_melds))
                elif not _legacy_v11 and kind == "kakan":
                    # Kakan consumes one concealed tile and upgrades an existing
                    # pon, so the fixed-meld count does not change.
                    post = hand.copy()
                    tile = getattr(action, "tile", None)
                    values = list(consumed_tiles(action))
                    added = int(tile) if tile is not None else (int(values[-1]) if values else -1)
                    matching = next(
                        (physical for physical in post if physical == added),
                        next((physical for physical in post if physical // 4 == added // 4), None),
                    )
                    if matching is None:
                        raise RuntimeError("legal kakan tile is absent from the concealed hand")
                    try:
                        post.remove(matching)
                    except ValueError:
                        raise RuntimeError("legal kakan tile is absent from the concealed hand") from None
                    call_melds = _upgrade_kakan_melds(melds, action, added_tile=matching)
                    if call_melds is None:
                        raise RuntimeError("legal kakan could not be matched to an existing pon")
                    jobs.append((row, action, aid, post, call_melds))
                    hands.append(_counts(post))
                    opened.append(len(call_melds))

        values = analyzer.analyze(hands, opened) if hands else []
        grouped: list[dict[int, list[Candidate]]] = [dict() for _ in decisions]
        scoring_started = time.perf_counter()
        for (row, action, aid, hand, melds), structural in zip(jobs, values, strict=True):
            # Moving a tile from the concealed hand into a discard or meld
            # does not change how many copies are available.  Always derive
            # remaining counts from the original observation hand so simulated
            # consumed/forced-discard tiles cannot reappear as ukeire.
            remaining = remaining_by_row[row]
            candidate = _candidate(
                decisions[row], action, aid, hand, melds, structural, remaining, public,
                legacy=_legacy_v11,
            )
            grouped[row].setdefault(aid, []).append(candidate)
        if profiler is not None:
            profiler.add("features/hand_evaluator_and_points", time.perf_counter() - scoring_started)

        rows: dict[int, DecisionAnalysis] = {}
        for row, decision in enumerate(decisions):
            collapsed: list[Candidate] = []
            for variants in grouped[row].values():
                selected = min(variants, key=lambda item: item.rank)
                collapsed.append(replace(selected, legal_discard_count=len(variants)))
            kinds = {action_kind(action) for action in actions_by_row[row]}
            foregone_win = bool(
                not _legacy_v11 and kinds & {"hora", "ron", "tsumo"}
            )
            non_pass = [item for item in collapsed if action_kind(item.action) not in {"none", "pass"}]
            if non_pass:
                best_non_pass = min(non_pass, key=lambda item: item.rank)
                collapsed = [
                    replace(
                        item,
                        foregone_shanten_improvement=max(0, item.structural_shanten - best_non_pass.structural_shanten),
                        foregone_ukeire_improvement=max(0, best_non_pass.ukeire - item.ukeire),
                        foregone_win=foregone_win or best_non_pass.structural_shanten < 0,
                    ) if action_kind(item.action) in {"none", "pass"} else item
                    for item in collapsed
                ]
            elif foregone_win:
                collapsed = [
                    replace(item, foregone_win=True)
                    if action_kind(item.action) in {"none", "pass"} else item
                    for item in collapsed
                ]
            candidates = tuple(collapsed)
            best = min((candidate.rank for candidate in candidates), default=None)
            teacher = np.zeros(NUM_ACTIONS, dtype=np.bool_)
            if _legacy_v11 and best is not None:
                for candidate in candidates:
                    if candidate.rank == best:
                        teacher[candidate.action_id] = True
            elif candidates:
                best_structural = min(candidate.structural_shanten for candidate in candidates)
                structural = [candidate for candidate in candidates if candidate.structural_shanten == best_structural]
                best_effective = min(candidate.effective_shanten for candidate in structural)
                effective = [candidate for candidate in structural if candidate.effective_shanten == best_effective]
                best_ukeire = max(candidate.ukeire for candidate in effective)
                for candidate in effective:
                    if candidate.ukeire == best_ukeire:
                        teacher[candidate.action_id] = True
            excluded = kinds & {"reach", "ankan", "kakan", "hora", "ron", "tsumo"}
            supervised = (
                "" if excluded else
                ("dahai" if "dahai" in kinds else
                 ("call" if kinds & {"chi", "pon", "daiminkan"} else ""))
            )
            if not supervised or (not _legacy_v11 and len(candidates) < 2):
                teacher.fill(False)
            rows[id(decision)] = DecisionAnalysis(
                decision, actions_by_row[row], candidates, best, teacher, supervised,
            )
        return cls(rows, analyzer, public)

    @classmethod
    def build_legacy_v11(
        cls,
        decisions: Iterable[object],
        *,
        analyzer: EfficiencyAnalyzer,
        public: object | None = None,
        profiler: object | None = None,
    ) -> "DecisionAnalysisBatch":
        """Frozen compatibility hook; only ``legacy.v11`` may call this."""
        return cls.build(
            decisions, analyzer=analyzer, public=public, profiler=profiler,
            _legacy_v11=True,
        )

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

    def state_tokens(self, decisions: Iterable[object]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        from ...model.actor_features import encode_actor_state_summary

        factor_rows: list[np.ndarray] = []
        numeric_rows: list[np.ndarray] = []
        for decision in decisions:
            factors, numeric = encode_actor_state_summary(
                decision.observation, self.for_decision(decision), self._analyzer,
            )
            factor_rows.append(factors)
            numeric_rows.append(numeric)
        return factor_rows, numeric_rows

    def candidate_tokens(
        self,
        decisions: Iterable[object],
        legal_masks: np.ndarray,
        *,
        public: object | None = None,
        profiler: object | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Encode two isolated public query tokens per legal action."""
        public = public if public is not None else self._public
        factor_rows: list[np.ndarray] = []
        numeric_rows: list[np.ndarray] = []
        for row_index, decision in enumerate(decisions):
            normalization_started = time.perf_counter()
            analysis = self.for_decision(decision)
            by_id = {candidate.action_id: candidate for candidate in analysis.candidates}
            legal_ids = np.flatnonzero(legal_masks[row_index])
            factors = np.zeros((2 * len(legal_ids), 10), dtype=np.uint8)
            numeric = np.zeros((2 * len(legal_ids), 8), dtype=np.float32)
            observation = decision.observation
            seat = int(getattr(observation, "player_id", 0))
            riichi = tuple(getattr(observation, "riichi_declared", ()) or (False,) * 4)
            threat_count = sum(bool(riichi[player]) for player in range(min(4, len(riichi))) if player != seat)
            action_by_id = {
                action_id(action, observation): action for action in analysis.actions
                if action_id(action, observation) is not None
            }
            dora_types = _dora_types(observation)
            remaining = public_remaining(observation)
            offline_pass_masks = [0] * 4
            history_available = hasattr(observation, "events")
            if history_available:
                from .public_state import tile_type as mjai_tile_type

                reached: set[int] = set()
                for event in getattr(observation, "events", ()) or ():
                    if isinstance(event, str):
                        try:
                            event = json.loads(event)
                        except ValueError:
                            continue
                    if not isinstance(event, dict):
                        continue
                    kind = str(event.get("type", ""))
                    actor = int(event.get("actor", -1))
                    if kind in {"reach", "reach_accepted"} and 0 <= actor < 4:
                        reached.add(actor)
                    elif kind == "dahai":
                        safe_type = mjai_tile_type(event.get("pai"))
                        if safe_type is not None:
                            for opponent in reached:
                                if opponent != actor:
                                    offline_pass_masks[opponent] |= 1 << safe_type
            if public is not None and hasattr(public, "post_riichi_safe_masks"):
                for opponent in range(4):
                    offline_pass_masks[opponent] |= int(
                        public.post_riichi_safe_masks[decision.env_index, opponent]
                    )
            legal_riichi_tiles: set[int] = set()
            if 75 in legal_ids:
                from riichienv import check_riichi_candidates

                legal_riichi_tiles = {
                    int(tile) for tile in check_riichi_candidates(_physical_hand(observation))
                }
            riichi_discards = tuple(
                item for item in analysis.candidates
                if action_kind(item.action) == "dahai" and item.structural_shanten == 0
                and item.closed and item.riichi_route
                and int(getattr(item.action, "tile")) in legal_riichi_tiles
            )
            best_discard = min(
                riichi_discards,
                key=lambda item: item.rank, default=None,
            )
            opponents = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
            rust_features = dict(by_id)
            if best_discard is not None:
                rust_features[75] = best_discard
            if profiler is not None:
                profiler.add("features/action_normalization", time.perf_counter() - normalization_started)
            rust_started = time.perf_counter()
            defense_batch, rust_factors, rust_numeric = _defense_feature_batch(
                observation, legal_ids, action_by_id, remaining, opponents, offline_pass_masks,
                rust_features,
            )
            if profiler is not None:
                profiler.add("features/rust_batch_analysis", time.perf_counter() - rust_started)
            fill_started = time.perf_counter()
            for pair, aid_value in enumerate(legal_ids):
                aid = int(aid_value)
                offense, defense = 2 * pair, 2 * pair + 1
                factors[defense] = rust_factors[pair]
                numeric[defense] = rust_numeric[pair]
                candidate = by_id.get(int(aid))
                action = action_by_id.get(aid)
                # A reach query uses the best legal declaration-discard proxy;
                # its own action remains independently identified by aid.
                feature = best_discard if aid == 75 and candidate is None else candidate
                factors[offense, 0] = factors[defense, 0] = 7
                factors[offense, 1] = factors[defense, 1] = _action_type_code(aid)
                factors[offense, 2] = factors[defense, 2] = min(aid + 1, 255)
                factors[offense, 3] = min(feature.preserve_dora + 1, 7) if feature is not None else 0
                factors[defense, 3] = min(threat_count, 3) if feature is not None else 0
                factors[offense, 9] = 1
                factors[defense, 9] = 2
                if feature is not None:
                    factors[offense, 4] = encode_shanten(feature.structural_shanten, maximum=7)
                    factors[offense, 5] = encode_shanten(feature.effective_shanten, maximum=15)
                    factors[offense, 6] = int(feature.has_yaku) + 2 * int(feature.riichi_route)
                    factors[offense, 7] = (
                        int(feature.open_no_yaku) + 2 * int(feature.furiten)
                        + 4 * int(feature.closed)
                    )
                    packed_points = min(feature.ron_points // 1000, 15) + 16 * min(feature.tsumo_points // 1000, 15)
                    factors[offense, 8] = packed_points
                    numeric[offense] = (
                        _unit(feature.standard_shanten, 6, signed=True),
                        _unit(feature.seven_pairs_shanten, 6, signed=True),
                        _unit(feature.thirteen_orphans_shanten, 13, signed=True),
                        _unit(feature.ukeire, 40),
                        _unit(feature.wait_count, 13),
                        _unit(feature.live_ron, 16),
                        _unit(feature.live_tsumo, 16),
                        _unit(feature.preserve_red, 3),
                    )
                kind = action_kind(action) if action is not None else ""
                tile = getattr(action, "tile", None) if action is not None else None
                tile_type = int(tile) // 4 if tile is not None and kind == "dahai" else -1
                if tile_type >= 0:
                    defense_values = defense_batch[pair]
                    genbutsu_mask, suji_mask = int(defense_values[0]), int(defense_values[1])
                    factors[defense, 4] = genbutsu_mask
                    factors[defense, 5] = suji_mask
                    visible = 4 - int(remaining[tile_type])
                    factors[defense, 6] = int(defense_values[2])
                    factors[defense, 7] = int(defense_values[4])
                    factors[defense, 8] = (
                        1 + int(feature.structure_known_mask) + 8 * int(feature.structure_break_mask)
                        if feature else 0
                    )
                    numeric[defense] = (
                        _unit(genbutsu_mask.bit_count(), 3), _unit(suji_mask.bit_count(), 3), _unit(visible, 4),
                        _unit(int(defense_values[3]), 4),
                        float(tile_type in dora_types), float(int(tile) in {16, 52, 88}),
                        _unit(threat_count, 3), (_unit(feature.structure_known_mask, 7) if feature else 0.0),
                    )
                elif feature is not None:
                    # Non-discard action-risk summary; zero-valued numeric
                    # fields are valid only because categorical factor[4]
                    # explicitly says this is the non-discard variant.
                    factors[defense, 4] = 1
                    factors[defense, 5] = 1 + int(kind in {"chi", "pon", "daiminkan"})
                    factors[defense, 6] = 1 + int(bool(feature and feature.open_no_yaku))
                    factors[defense, 7] = 1 + int(kind in {"ankan", "kakan", "daiminkan"} and threat_count > 0)
                    numeric[defense, 0] = _unit(feature.structural_shanten, 6, signed=True) if feature else 0.0
                    numeric[defense, 1] = _unit(feature.ukeire, 40) if feature else 0.0
                    numeric[defense, 2] = _unit(threat_count, 3)
                    numeric[defense, 3] = _unit(len(riichi_discards), 14) if aid == 75 else 0.0
                    numeric[defense, 4] = _unit(feature.legal_discard_count, 14) if feature else 0.0
                    numeric[defense, 5] = _unit(feature.foregone_shanten_improvement, 6) if feature else 0.0
                    numeric[defense, 6] = _unit(feature.foregone_ukeire_improvement, 40) if feature else 0.0
                    numeric[defense, 7] = float(bool(feature and feature.foregone_win))
            factor_rows.append(factors)
            numeric_rows.append(numeric)
            if profiler is not None:
                profiler.add("features/token_fill", time.perf_counter() - fill_started)
        return factor_rows, numeric_rows
