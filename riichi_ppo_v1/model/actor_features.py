"""Compact deterministic public-state summaries for v13 actor inputs."""

from __future__ import annotations

import json
import numpy as np

from ..training.rewards.efficiency import EfficiencyAnalyzer, remaining_ukeire
from .dora import dora_type_multiplicities
from .feature_schema import (
    HAND_CLOSED,
    HAND_RIICHI,
    STATE_HAND,
    STATE_PLACEMENT,
    STATE_SUMMARY_SEGMENT,
    STATE_THREAT,
    STATE_VALUE,
    VALUE_DAMATEN_YAKU,
    VALUE_RIICHI_ROUTE,
    encode_shanten,
)


def _own_hand(observation: object) -> list[int]:
    seat = int(_required_scalar(observation, "player_id"))
    hands = _required_sequence(observation, "hands", length=4)
    return [int(tile) for tile in hands[seat]]


def _own_meld_count(observation: object) -> int:
    seat = int(_required_scalar(observation, "player_id"))
    melds = _required_sequence(observation, "melds", length=4)
    return len(melds[seat])


def _unit(value: int | float, scale: int | float) -> float:
    """Normalize a non-negative contract field and enforce its [0, 1] range."""
    return float(np.clip(float(value) / float(scale), 0.0, 1.0))


def _signed_unit(value: int | float, scale: int | float) -> float:
    """Normalize a signed score/gap field and enforce its [-1, 1] range."""
    return float(np.clip(float(value) / float(scale), -1.0, 1.0))


def _events(observation: object) -> list[dict[str, object]]:
    """Return the accumulated public history or fail instead of fabricating data."""
    raw = getattr(observation, "events", None)
    if raw is None:
        raise RuntimeError("schema-13 actor summaries require accumulated public observation.events")
    result: list[dict[str, object]] = []
    for value in raw:
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise RuntimeError("observation.events contains a non-MJAI event")
        result.append(value)
    return result


def _required_sequence(observation: object, name: str, *, length: int | None = None) -> object:
    value = getattr(observation, name, None)
    if value is None:
        raise RuntimeError(f"schema-13 actor summaries require observation.{name}")
    if length is not None and len(value) != length:
        raise RuntimeError(
            f"schema-13 actor summaries require observation.{name} to have length {length}"
        )
    return value


def _required_scalar(observation: object, name: str) -> object:
    if not hasattr(observation, name) or getattr(observation, name) is None:
        raise RuntimeError(f"schema-13 actor summaries require observation.{name}")
    return getattr(observation, name)


def _event_threat_rows(observation: object) -> list[dict[str, int | bool]]:
    from ..training.rewards.public_state import tile_type

    rows = [
        {
            "declared": False, "accepted": False, "reach_turn": 0,
            "discards": 0, "melds": 0, "last_tedashi": -1,
            "tsumogiri_streak": 0, "reach_tile": -1,
        }
        for _ in range(4)
    ]
    for event in _events(observation):
        kind = str(event.get("type", ""))
        actor = int(event.get("actor", -1))
        if not 0 <= actor < 4:
            continue
        row = rows[actor]
        if kind == "reach":
            row["declared"] = True
            row["reach_turn"] = int(row["discards"])
        elif kind == "reach_accepted":
            row["declared"] = True
            row["accepted"] = True
        elif kind == "dahai":
            row["discards"] = int(row["discards"]) + 1
            tile = tile_type(event.get("pai"))
            is_tsumogiri = bool(event.get("tsumogiri", False))
            if is_tsumogiri:
                row["tsumogiri_streak"] = int(row["tsumogiri_streak"]) + 1
            else:
                row["last_tedashi"] = -1 if tile is None else int(tile)
                row["tsumogiri_streak"] = 0
            if bool(row["declared"]) and int(row["reach_tile"]) < 0:
                row["reach_tile"] = -1 if tile is None else int(tile)
        elif kind in {"chi", "pon", "daiminkan", "ankan"}:
            row["melds"] = int(row["melds"]) + 1
        # Kakan upgrades an existing pon and must not increase meld count.
    return rows


def _observation_threat_rows(observation: object) -> list[dict[str, int | bool]]:
    """Build strict threat facts from the complete observation snapshot."""
    declared = _required_sequence(observation, "riichi_declared", length=4)
    accepted = _required_sequence(observation, "riichi_accepted", length=4)
    declaration_indices = _required_sequence(
        observation, "riichi_declaration_indices", length=4,
    )
    discards = _required_sequence(observation, "discards", length=4)
    melds = _required_sequence(observation, "melds", length=4)
    last_tedashis = _required_sequence(observation, "last_tedashis", length=4)
    tsumogiri_flags = _required_sequence(observation, "tsumogiri_flags", length=4)
    reach_tiles = _required_sequence(observation, "riichi_sutehais", length=4)
    rows: list[dict[str, int | bool]] = []
    for player in range(4):
        flags = list(tsumogiri_flags[player])
        streak = 0
        for flag in reversed(flags):
            if not bool(flag):
                break
            streak += 1
        last = last_tedashis[player]
        reach_tile = reach_tiles[player]
        reach_index = declaration_indices[player]
        if bool(accepted[player]) and not bool(declared[player]):
            raise RuntimeError("riichi acceptance cannot precede declaration")
        if bool(declared[player]) and (reach_tile is None or reach_index is None):
            raise RuntimeError(
                "declared riichi requires a declaration tile and river index"
            )
        rows.append({
            "declared": bool(declared[player]),
            "accepted": bool(accepted[player]),
            "reach_turn": -1 if reach_index is None else int(reach_index),
            "discards": len(discards[player]),
            "melds": len(melds[player]),
            "last_tedashi": -1 if last is None else int(last) // 4,
            "tsumogiri_streak": streak,
            "reach_tile": -1 if reach_tile is None else int(reach_tile) // 4,
        })
    return rows


def encode_actor_state_summary(
    observation: object,
    decision_analysis: object,
    analyzer: EfficiencyAnalyzer,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exactly six compact public summary rows.

    The rows are hand structure, tenpai/value, placement, then one threat row
    for each opponent. Unknown facts use categorical N/A (zero) independently
    from numeric zero.
    """
    factors = np.zeros((6, 10), dtype=np.uint8)
    numeric = np.zeros((6, 8), dtype=np.float32)
    hand = _own_hand(observation)
    counts = np.bincount([tile // 4 for tile in hand], minlength=34).astype(np.uint8)
    opened = _own_meld_count(observation)
    analyzed = analyzer.analyze([counts], [opened])[0]
    riichi = tuple(_required_sequence(observation, "riichi_declared", length=4))
    seat = int(getattr(observation, "player_id", 0))
    _required_scalar(observation, "missed_agari_doujun")
    _required_scalar(observation, "missed_agari_riichi")
    from ..training.rewards.decision import (
        _is_closed, _own_melds, _rule_state, public_remaining,
    )

    melds = _own_melds(observation)
    remaining = public_remaining(observation, hand)
    current_rules = _rule_state(observation, hand, melds, int(analyzed.shanten), remaining)
    closed = _is_closed(melds)
    dora_multiplicities = dora_type_multiplicities(
        _required_sequence(observation, "dora_indicators")
    )
    own_value_tiles = set(hand)
    own_value_tiles.update(
        int(tile) for meld in melds for tile in (getattr(meld, "tiles", ()) or ())
    )
    hand_dora = sum(dora_multiplicities.get(tile // 4, 0) for tile in own_value_tiles)
    hand_red = sum(int(tile in {16, 52, 88}) for tile in own_value_tiles)
    represented_tiles = len(hand) + 3 * opened
    ukeire_available = represented_tiles == 13

    # Hand structure.
    factors[0, :6] = (
        STATE_SUMMARY_SEGMENT, STATE_HAND,
        encode_shanten(analyzed.shanten, maximum=255),
        encode_shanten(analyzed.standard_shanten, maximum=7),
        encode_shanten(analyzed.seven_pairs_shanten, maximum=7),
        encode_shanten(analyzed.thirteen_orphans_shanten, maximum=15),
    )
    factors[0, 6] = (
        (HAND_CLOSED if closed else 0)
        | (HAND_RIICHI if bool(riichi[seat]) else 0)
    )
    factors[0, 7] = 1 + int(current_rules.furiten)
    factors[0, 8] = int(ukeire_available)
    numeric[0, 0] = _signed_unit(analyzed.shanten, 6)
    numeric[0, 1] = _signed_unit(analyzed.standard_shanten or 0, 6)
    numeric[0, 2] = _signed_unit(analyzed.seven_pairs_shanten or 0, 6)
    numeric[0, 3] = _signed_unit(analyzed.thirteen_orphans_shanten or 0, 13)
    if ukeire_available:
        numeric[0, 4] = _unit(remaining_ukeire(analyzed.improving_mask, remaining), 40)
        numeric[0, 5] = _unit(int(analyzed.improving_mask).bit_count(), 34)

    value_flags = (
        (VALUE_DAMATEN_YAKU if current_rules.damaten_ron_yaku else 0)
        | (VALUE_RIICHI_ROUTE if current_rules.riichi_route else 0)
    )
    factors[1, 0:4] = (STATE_SUMMARY_SEGMENT, STATE_VALUE, 1, value_flags)
    numeric[1] = (
        _unit(len(current_rules.waits), 13),
        _unit(current_rules.live_ron, 16),
        _unit(current_rules.live_tsumo, 16),
        _unit(current_rules.ron_points, 12000),
        _unit(current_rules.tsumo_points, 12000),
        _unit(hand_dora, 8),
        _unit(hand_red, 3),
        0.0,
    )

    # Placement and match progress.
    scores = [int(value) for value in _required_sequence(observation, "scores", length=4)]
    order = sorted(range(4), key=lambda player: (-scores[player], player))
    rank = order.index(seat) + 1
    sorted_scores = sorted(scores, reverse=True)
    round_wind = int(_required_scalar(observation, "round_wind"))
    kyoku_index = int(_required_scalar(observation, "kyoku_index"))
    match_progress = max(0, round_wind * 4 + kyoku_index)
    factors[2, 0:7] = (
        STATE_SUMMARY_SEGMENT, STATE_PLACEMENT, rank, min(int(_required_scalar(observation, "round_wind")) + 1, 7),
        min(kyoku_index + 1, 7), min(match_progress + 1, 15),
        1 + int(round_wind >= 1 and kyoku_index >= 3),
    )
    numeric[2] = (
        _signed_unit(scores[seat], 50000),
        _signed_unit(scores[seat] - sorted_scores[0], 50000),
        _signed_unit(scores[seat] - sorted_scores[1], 50000),
        _signed_unit(scores[seat] - sorted_scores[2], 50000),
        _signed_unit(scores[seat] - sorted_scores[3], 50000),
        _unit(_required_scalar(observation, "honba"), 10),
        _unit(_required_scalar(observation, "riichi_sticks"), 10),
        _unit(_required_scalar(observation, "tiles_left"), 70),
    )

    discards = _required_sequence(observation, "discards", length=4)
    all_melds = _required_sequence(observation, "melds", length=4)
    _required_sequence(observation, "dora_indicators")
    threat_rows = _observation_threat_rows(observation)
    opponents = [(seat + offset) % 4 for offset in (1, 2, 3)]
    for row, opponent in enumerate(opponents, start=3):
        river = discards[opponent] if len(discards) > opponent else ()
        opponent_melds = all_melds[opponent] if len(all_melds) > opponent and isinstance(all_melds[opponent], (list, tuple)) else ()
        threat = threat_rows[opponent]
        last_tedashi = int(threat["last_tedashi"])
        reach_tile = int(threat["reach_tile"])
        streak = int(threat["tsumogiri_streak"])
        exposed_tiles = {int(tile) for tile in river}
        exposed_tiles.update(
            int(tile) for meld in opponent_melds
            for tile in (getattr(meld, "tiles", ()) or ())
        )
        exposed_dora = sum(
            dora_multiplicities.get(tile // 4, 0) for tile in exposed_tiles
        )
        factors[row, 0:6] = (
            STATE_SUMMARY_SEGMENT, STATE_THREAT, row - 2,
            1 + int(bool(threat["declared"])) + 2 * int(bool(threat["accepted"])),
            min(len(opponent_melds) + 1, 7), min(len(river) + 1, 15),
        )
        factors[row, 6] = min(streak, 3)
        factors[row, 8] = reach_tile + 1 if reach_tile >= 0 else 0
        numeric[row, 0] = _unit(len(river), 24)
        numeric[row, 1] = _unit(len(opponent_melds), 4)
        numeric[row, 2] = _unit(last_tedashi + 1, 35) if last_tedashi >= 0 else 0.0
        numeric[row, 3] = _unit(streak, 12)
        numeric[row, 4] = _unit(exposed_dora, 8)
        if reach_tile >= 0:
            numeric[row, 5] = _unit(int(threat["reach_turn"]), 24)
    return factors, numeric
