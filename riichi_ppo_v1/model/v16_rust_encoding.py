"""V16 Action Query 的 Rust 融合编码边界。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import riichi
import riichienv

from .action_query import (
    _action_kind,
    _count_dora_aka,
    _decompose_melds,
    _kernel_shape,
    _observation_facts,
    _physical_tiles,
    _remove_first_by_type,
    _source_seat,
    _visible_code,
)
from .encoding_protocol import ACTION_TYPE_CODES, QUERY_ROW_ANSWER_START, bucket_o5, bucket_o9
from .schema import TILE_KINDS

_MODE_FULL_OFFENSE = 0
_MODE_SIMPLE_SHANTEN = 1
_MODE_WIN = 2
_MODE_MIN_DROP = 3
_O8_FROM_ANALYSIS = 255
_KAN_KINDS = frozenset({"ankan", "daiminkan", "kakan"})


@dataclass(frozen=True)
class RustV16QueryBatch:
    """逐动作连续 query 行及融合内核的去重统计。"""

    query_rows: np.ndarray
    unique_offense_rows: int
    unique_shanten_rows: int


@dataclass(frozen=True)
class _YakuInput:
    post: list[int]
    melds: list[Any]
    dora_indicators: list[int]
    player_wind: int
    round_wind: int
    honba: int
    riichi_sticks: int


def _counts(tiles: list[int]) -> np.ndarray:
    return np.bincount(
        [int(tile) // 4 for tile in tiles], minlength=TILE_KINDS,
    ).astype(np.uint8)


def encode_action_queries_batch_rust(
    rows: list[tuple[object, object, int]],
) -> RustV16QueryBatch:
    """把 `(observation, action, action_id)` 融合编码为 `[N,2,15]`。

    Python 只提取 PyObject 中的紧凑事实;post-shape 的向听/有效牌/防守计算、
    kind 分派结果组装和 query row 写入均在单次 Rust 批调用中完成。同一批内完全
    相同的 post-shape 只计算一次,但每个 action 的输出行仍逐一保留。
    """
    if not rows:
        return RustV16QueryBatch(np.zeros((0, 2, 15), dtype=np.int32), 0, 0)

    n = len(rows)
    action_ids = np.empty(n, dtype=np.uint16)
    action_types = np.empty(n, dtype=np.uint8)
    primary_types = np.empty(n, dtype=np.int16)
    source_seats = np.empty(n, dtype=np.int8)
    modes = np.empty(n, dtype=np.uint8)
    shape_counts = np.zeros((n, TILE_KINDS), dtype=np.uint8)
    open_melds = np.zeros(n, dtype=np.uint8)
    remaining = np.empty((n, TILE_KINDS), dtype=np.uint8)
    own_rivers = np.empty(n, dtype=np.uint64)
    opponent_rivers = np.empty((n, 3), dtype=np.uint64)
    defense_counts = np.zeros((n, TILE_KINDS), dtype=np.uint8)
    discard_types = np.full(n, -1, dtype=np.int16)
    defense_visible = np.empty(n, dtype=np.uint8)
    missed_doujun = np.empty(n, dtype=np.bool_)
    missed_riichi = np.empty(n, dtype=np.bool_)
    riichi_declared = np.empty(n, dtype=np.bool_)
    scores = np.empty(n, dtype=np.int32)
    o7_values = np.empty(n, dtype=np.uint8)
    o8_values = np.empty(n, dtype=np.uint8)
    o9_values = np.empty(n, dtype=np.uint8)
    yaku_inputs: list[_YakuInput | None] = [None] * n

    facts_cache: dict[int, dict[str, object]] = {}
    for row, (observation, action, action_id) in enumerate(rows):
        facts = facts_cache.setdefault(id(observation), _observation_facts(observation))
        kind = _action_kind(action)
        hand = list(facts["hand"])
        melds = list(facts["melds"])
        rivers = np.asarray(facts["rivers"], dtype=np.uint64)
        opponents = tuple(int(value) for value in facts["opponents"])
        public_visible = np.asarray(facts["public_visible"], dtype=np.uint8)
        dora_mult = facts["dora_mult"]
        menzen = bool(facts["menzen"])
        tile = getattr(action, "tile", None)
        if tile is None and kind in {"reach", "dahai"}:
            drawn = getattr(observation, "drawn_tile", None)
            if drawn is not None:
                tile = int(drawn)
        tile = None if tile is None else int(tile)
        primary_type = -1 if tile is None else tile // 4

        action_ids[row] = int(action_id)
        action_types[row] = int(ACTION_TYPE_CODES.get(kind, 0))
        primary_types[row] = primary_type
        source = _source_seat(observation, kind)
        source_seats[row] = -1 if source is None else int(source)
        remaining[row] = np.asarray(facts["remaining"], dtype=np.uint8)
        own_rivers[row] = int(facts["own_river"])
        opponent_rivers[row] = rivers[list(opponents)]
        missed_doujun[row] = bool(facts["missed_doujun"])
        missed_riichi[row] = bool(facts["missed_riichi"])
        riichi_declared[row] = bool(facts["declared"])
        scores[row] = int(facts["score"])
        o7_values[row] = 0 if menzen else 1

        if kind in {"tsumo", "ron"}:
            modes[row] = _MODE_WIN
            full = _physical_tiles(hand, melds)
            if tile is not None:
                full.append(tile)
            o8_values[row] = 2
            o9_values[row] = bucket_o9(_count_dora_aka(full, dora_mult))
            defense_visible[row] = _visible_code(
                public_visible, None if primary_type < 0 else primary_type,
            )
            continue

        if kind in {"reach", "dahai"}:
            post = list(hand)
            if tile is not None:
                try:
                    post.remove(tile)
                except ValueError:
                    pass
            three_melds, kan_types = _decompose_melds(melds)
            shape, three_melds = _kernel_shape(post, three_melds, kan_types, target=13)
            modes[row] = _MODE_FULL_OFFENSE
            shape_counts[row] = shape
            open_melds[row] = three_melds
            defense_counts[row] = _counts(post)
            discard_types[row] = primary_type
            riichi_declared[row] = kind == "reach" or bool(facts["declared"])
            o8_values[row] = 2 if kind == "reach" else _O8_FROM_ANALYSIS
            o9_values[row] = bucket_o9(
                _count_dora_aka(_physical_tiles(post, melds), dora_mult),
            )
            defense_visible[row] = _visible_code(
                public_visible, None if primary_type < 0 else primary_type,
            )
            yaku_inputs[row] = _YakuInput(
                post=post,
                melds=melds,
                dora_indicators=list(facts["dora_indicators"]),
                player_wind=int(facts["player_wind"]),
                round_wind=int(facts["round_wind"]),
                honba=int(facts["honba"]),
                riichi_sticks=int(facts["sticks"]),
            )
            continue

        if kind in {"chi", "pon", "ankan", "daiminkan", "kakan"}:
            consumed = [int(value) for value in (getattr(action, "consume_tiles", ()) or ())]
            post = list(hand)
            called = tile
            if kind == "kakan":
                added = called if called is not None else (consumed[-1] if consumed else None)
                if added is not None:
                    _remove_first_by_type(post, added // 4, 1)
            else:
                consumed_counts: dict[int, int] = {}
                for value in consumed:
                    tile_type = value // 4
                    consumed_counts[tile_type] = consumed_counts.get(tile_type, 0) + 1
                for tile_type, count in sorted(consumed_counts.items()):
                    _remove_first_by_type(post, tile_type, count)
            three_melds, kan_types = _decompose_melds(melds)
            kan_type = (
                (called if called is not None else consumed[0]) // 4
                if called is not None or consumed
                else None
            )
            o7_values[row] = 0 if menzen and kind == "ankan" else 1
            if kind in {"chi", "pon"}:
                shape, three_melds = _kernel_shape(
                    post, three_melds + 1, kan_types, target=14,
                )
                new_meld = list(consumed)
                if called is not None and called not in new_meld:
                    new_meld.append(called)
                full = _physical_tiles(post, melds) + new_meld
            elif kind == "daiminkan":
                extra_kans = [*kan_types, kan_type] if kan_type is not None else kan_types
                shape, three_melds = _kernel_shape(
                    post, three_melds, extra_kans, target=13,
                )
                full = list(post) + ([called] * 3 if called is not None else consumed)
            elif kind == "ankan":
                extra_kans = [*kan_types, kan_type] if kan_type is not None else kan_types
                shape, three_melds = _kernel_shape(
                    post, three_melds, extra_kans, target=13,
                )
                full = list(post) + consumed
            else:
                added = called if called is not None else (consumed[-1] if consumed else None)
                if added is not None:
                    try:
                        post.remove(added)
                    except ValueError:
                        pass
                extra_kans = [*kan_types, kan_type] if kan_type is not None else kan_types
                shape, three_melds = _kernel_shape(
                    post, max(0, three_melds - 1), extra_kans, target=13,
                )
                full = _physical_tiles(post, melds) + ([added] if added is not None else [])
            modes[row] = _MODE_SIMPLE_SHANTEN
            shape_counts[row] = shape
            open_melds[row] = three_melds
            defense_counts[row] = _counts(post)
            o8_values[row] = 2 if kind in _KAN_KINDS else 1
            o9_values[row] = bucket_o9(_count_dora_aka(full, dora_mult))
            defense_visible[row] = _visible_code(
                public_visible, None if primary_type < 0 else primary_type,
            )
            continue

        full = _physical_tiles(hand, melds)
        defense_counts[row] = _counts(hand)
        defense_visible[row] = 5
        o8_values[row] = 2
        o9_values[row] = bucket_o9(_count_dora_aka(full, dora_mult))
        if len(full) == 13:
            three_melds, kan_types = _decompose_melds(melds)
            shape, three_melds = _kernel_shape(hand, three_melds, kan_types, target=13)
            modes[row] = _MODE_FULL_OFFENSE
            shape_counts[row] = shape
            open_melds[row] = three_melds
            yaku_inputs[row] = _YakuInput(
                post=hand,
                melds=melds,
                dora_indicators=list(facts["dora_indicators"]),
                player_wind=int(facts["player_wind"]),
                round_wind=int(facts["round_wind"]),
                honba=int(facts["honba"]),
                riichi_sticks=int(facts["sticks"]),
            )
        elif not melds:
            modes[row] = _MODE_MIN_DROP
            shape_counts[row] = _counts(hand)
        else:
            three_melds, kan_types = _decompose_melds(melds)
            shape, three_melds = _kernel_shape(hand, three_melds, kan_types, target=13)
            modes[row] = _MODE_SIMPLE_SHANTEN
            shape_counts[row] = shape
            open_melds[row] = three_melds

    encoded = riichi.encode_v16_batch(
        action_ids,
        action_types,
        primary_types,
        source_seats,
        modes,
        shape_counts,
        open_melds,
        remaining,
        own_rivers,
        opponent_rivers,
        defense_counts,
        discard_types,
        defense_visible,
        missed_doujun,
        missed_riichi,
        riichi_declared,
        scores,
        o7_values,
        o8_values,
        o9_values,
    )
    query_rows = np.asarray(encoded.query_rows, dtype=np.int32)
    wait_masks = np.asarray(encoded.wait_masks, dtype=np.uint64)
    wait_rows = [
        row for row, mask in enumerate(wait_masks)
        if int(mask) != 0 and yaku_inputs[row] is not None
    ]
    if wait_rows:
        inputs = [yaku_inputs[row] for row in wait_rows]
        yaku = riichienv.analyze_offense_v16(
            [value.post for value in inputs],
            [value.melds for value in inputs],
            np.asarray([wait_masks[row] for row in wait_rows], dtype=np.uint64),
            [value.dora_indicators for value in inputs],
            np.asarray([value.player_wind for value in inputs], dtype=np.uint8),
            np.asarray([value.round_wind for value in inputs], dtype=np.uint8),
            np.asarray([value.honba for value in inputs], dtype=np.uint8),
            np.asarray([value.riichi_sticks for value in inputs], dtype=np.uint8),
        )
        yaku_class = np.asarray(yaku.yaku_class)
        base_han = np.asarray(yaku.base_han)
        for slot, row in enumerate(wait_rows):
            query_rows[row, 0, QUERY_ROW_ANSWER_START + 4] = int(yaku_class[slot])
            query_rows[row, 0, QUERY_ROW_ANSWER_START + 5] = bucket_o5(
                int(base_han[slot]) if int(base_han[slot]) > 0 else None,
            )

    return RustV16QueryBatch(
        query_rows=query_rows,
        unique_offense_rows=int(encoded.unique_offense_rows),
        unique_shanten_rows=int(encoded.unique_shanten_rows),
    )


def _patch_native_yaku(
    query_rows: np.ndarray,
    wait_masks: np.ndarray,
    rows: list[tuple[object, object, int]],
) -> None:
    """只为听牌行重建 core 役种输入并批量回填 O4/O5。"""
    wait_rows = np.flatnonzero(wait_masks).tolist()
    if not wait_rows:
        return
    inputs: list[_YakuInput] = []
    selected_rows: list[int] = []
    for row in wait_rows:
        observation, action, _action_id = rows[row]
        kind = _action_kind(action)
        if kind not in {"reach", "dahai", "none", "pass", "ryukyoku"}:
            continue
        facts = _observation_facts(observation)
        post = list(facts["hand"])
        if kind in {"reach", "dahai"}:
            tile = getattr(action, "tile", None)
            if tile is None:
                tile = getattr(observation, "drawn_tile", None)
            if tile is not None:
                try:
                    post.remove(int(tile))
                except ValueError:
                    pass
        inputs.append(_YakuInput(
            post=post,
            melds=list(facts["melds"]),
            dora_indicators=list(facts["dora_indicators"]),
            player_wind=int(facts["player_wind"]),
            round_wind=int(facts["round_wind"]),
            honba=int(facts["honba"]),
            riichi_sticks=int(facts["sticks"]),
        ))
        selected_rows.append(row)
    if not selected_rows:
        return
    yaku = riichienv.analyze_offense_v16(
        [value.post for value in inputs],
        [value.melds for value in inputs],
        np.asarray([wait_masks[row] for row in selected_rows], dtype=np.uint64),
        [value.dora_indicators for value in inputs],
        np.asarray([value.player_wind for value in inputs], dtype=np.uint8),
        np.asarray([value.round_wind for value in inputs], dtype=np.uint8),
        np.asarray([value.honba for value in inputs], dtype=np.uint8),
        np.asarray([value.riichi_sticks for value in inputs], dtype=np.uint8),
    )
    yaku_class = np.asarray(yaku.yaku_class)
    base_han = np.asarray(yaku.base_han)
    for slot, row in enumerate(selected_rows):
        query_rows[row, 0, QUERY_ROW_ANSWER_START + 4] = int(yaku_class[slot])
        query_rows[row, 0, QUERY_ROW_ANSWER_START + 5] = bucket_o5(
            int(base_han[slot]) if int(base_han[slot]) > 0 else None,
        )


def encode_action_queries_batch_native(
    rows: list[tuple[object, object, int]],
) -> RustV16QueryBatch:
    """Observation/Action 只各跨一次 PyO3,由 core 直接生成紧凑事实。"""
    if not rows:
        return RustV16QueryBatch(np.zeros((0, 2, 15), dtype=np.int32), 0, 0)
    observations: list[object] = []
    observation_lookup: dict[int, int] = {}
    observation_indices = np.empty(len(rows), dtype=np.uint32)
    actions: list[object] = []
    action_ids = np.empty(len(rows), dtype=np.uint16)
    for row, (observation, action, action_id) in enumerate(rows):
        key = id(observation)
        index = observation_lookup.get(key)
        if index is None:
            index = len(observations)
            observation_lookup[key] = index
            observations.append(observation)
        observation_indices[row] = index
        actions.append(action)
        action_ids[row] = int(action_id)
    facts = riichienv.prepare_v16_compact_facts(
        observations, observation_indices, actions, action_ids,
    )
    encoded = riichi.encode_v16_batch(
        facts.action_ids,
        facts.action_types,
        facts.primary_types,
        facts.source_seats,
        facts.modes,
        facts.shape_counts,
        facts.open_melds,
        facts.remaining,
        facts.own_rivers,
        facts.opponent_rivers,
        facts.defense_counts,
        facts.discard_types,
        facts.defense_visible,
        facts.missed_doujun,
        facts.missed_riichi,
        facts.riichi_declared,
        facts.scores,
        facts.o7_values,
        facts.o8_values,
        facts.o9_values,
    )
    query_rows = np.asarray(encoded.query_rows, dtype=np.int32)
    wait_masks = np.asarray(encoded.wait_masks, dtype=np.uint64)
    _patch_native_yaku(query_rows, wait_masks, rows)
    return RustV16QueryBatch(
        query_rows=query_rows,
        unique_offense_rows=int(encoded.unique_offense_rows),
        unique_shanten_rows=int(encoded.unique_shanten_rows),
    )
