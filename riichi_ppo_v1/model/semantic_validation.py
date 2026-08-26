"""V18 当前局面输入的语义校验（fail closed）。

本模块只验证输入协议不变量（顺序、域、守恒、排序、信息隔离），不重新计算麻将业务
事实；事实正确性由 bridge 等价测试与真实 replay fixture 覆盖。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .encoding_protocol import (
    CATEGORY_SCHEMAS,
    KIND_ACTION_DEFENSE_QUERY,
    KIND_ACTION_OFFENSE_QUERY,
    KIND_BOS,
    KIND_CRITIC_FUTURE,
    KIND_CRITIC_HAND,
    KIND_MELD,
    KIND_OPPONENT_ANALYSIS,
    KIND_PLAYER,
    KIND_RIVER_DISCARD,
    KIND_RIVER_SUMMARY,
    KIND_SEP_ACTIONS,
    KIND_SEP_CRITIC,
    KIND_SEP_KAMICHA_RIVER,
    KIND_SEP_MELDS,
    KIND_SEP_OPPONENT_ANALYSIS,
    KIND_SEP_PLAYERS,
    KIND_SEP_RIVERS,
    KIND_SEP_SELF_HAND,
    KIND_SEP_SHIMOCHA_RIVER,
    KIND_SEP_TILE_STATE,
    KIND_SEP_TOIMEN_RIVER,
    KIND_SELF_HAND,
    KIND_SELF_STATE_ANALYSIS,
    KIND_TABLE,
    KIND_TILE_STATE,
    NUM_SEPARATORS,
    QUERY_ROW_ACTION_ID,
    QUERY_ROW_ACTION_TYPE,
    QUERY_ROW_ANSWER_START,
    QUERY_ROW_PRIMARY_TILE,
    QUERY_ROW_QUERY_TYPE,
    QUERY_ROW_SOURCE_SEAT,
    QUERY_ROW_WIDTH,
    SEGMENT_ACTIONS,
    SEGMENT_ANALYSIS,
    SEGMENT_CRITIC_FUTURE,
    SEGMENT_CRITIC_PRIVATE,
    SEGMENT_SHARED,
    SLOT_CARDINALITIES,
    SUPPLIER_REQUIRED_ACTION_TYPES,
    TOKEN_NUMERIC_WIDTH,
    TOKEN_ROW_WIDTH,
    is_separator_kind,
    separator_id_of_kind,
)
from .schema import NUM_ACTIONS


def _assert_range(value: int, maximum: int) -> int:
    if value < 0 or value >= maximum:
        raise AssertionError(f"value {value} outside domain [0,{maximum})")
    return value


def _assert_actor_canonical_order(kinds: np.ndarray, rows: np.ndarray) -> None:
    """校验 actor 序列的规范 kind 顺序（含分隔符）与类别字段顺序。"""
    values = [int(value) for value in kinds]
    if values[0] != KIND_BOS:
        raise AssertionError("actor sequence must start with BOS")
    cursor = 1
    expected = KIND_TABLE
    if values[cursor] != expected:
        raise AssertionError(f"position {cursor} expects kind {expected}, got {values[cursor]}")
    cursor += 1
    for expected_sep, next_kind in (
        (KIND_SEP_SELF_HAND, KIND_SELF_HAND),
    ):
        if values[cursor] != expected_sep or values[cursor + 1] != next_kind:
            raise AssertionError(f"position {cursor} must be {expected_sep} then {next_kind}")
        break
    cursor += 2
    # SELF_HAND 若干（非零牌种，升序）。
    self_kinds = []
    while cursor < len(values) and values[cursor] == KIND_SELF_HAND:
        tile_type = int(rows[cursor, 2])
        if self_kinds and tile_type <= self_kinds[-1]:
            raise AssertionError("SELF_HAND kinds must be strictly ascending")
        self_kinds.append(tile_type)
        cursor += 1
    if not self_kinds:
        raise AssertionError("SELF_HAND requires at least one nonzero kind")
    for expected, label in (
        (KIND_SELF_STATE_ANALYSIS, "SELF_STATE_ANALYSIS"),
        (KIND_SEP_PLAYERS, "SEP_PLAYERS"),
    ):
        if cursor >= len(values) or values[cursor] != expected:
            raise AssertionError(f"position {cursor} expects {label}")
        cursor += 1
    for _ in range(4):
        if cursor >= len(values) or values[cursor] != KIND_PLAYER:
            raise AssertionError("PLAYER token block must contain exactly four rows")
        cursor += 1
    if cursor >= len(values) or values[cursor] != KIND_SEP_RIVERS:
        raise AssertionError("PLAYER block must be followed by SEP_RIVERS")
    cursor += 1
    for river_sep in (KIND_SEP_SHIMOCHA_RIVER, KIND_SEP_TOIMEN_RIVER, KIND_SEP_KAMICHA_RIVER):
        if cursor >= len(values) or values[cursor] != river_sep:
            raise AssertionError(f"expects river separator, got {values[cursor] if cursor < len(values) else 'EOF'}")
        cursor += 1
        if cursor >= len(values) or values[cursor] != KIND_RIVER_SUMMARY:
            raise AssertionError("each river must start with FIRST_SIX summary")
        cursor += 1
        while cursor < len(values) and values[cursor] == KIND_RIVER_DISCARD:
            cursor += 1
        if cursor >= len(values) or values[cursor] != KIND_RIVER_SUMMARY:
            raise AssertionError("each river must end with RECENT_SIX summary")
        cursor += 1
    if cursor >= len(values) or values[cursor] != KIND_SEP_MELDS:
        raise AssertionError("rivers block must be followed by SEP_MELDS")
    cursor += 1
    while cursor < len(values) and values[cursor] == KIND_MELD:
        cursor += 1
    if cursor >= len(values) or values[cursor] != KIND_SEP_TILE_STATE:
        raise AssertionError("meld block must be followed by SEP_TILE_STATE")
    cursor += 1
    if cursor + 34 > len(values) or values[cursor:cursor + 34] != [KIND_TILE_STATE] * 34:
        raise AssertionError("TILE_STATE block must contain exactly 34 rows")
    tile_types = [int(rows[cursor + index, 2]) for index in range(34)]
    if tile_types != list(range(1, 35)):
        raise AssertionError("TILE_STATE kinds must be 1..34 in ascending order")
    cursor += 34
    if cursor >= len(values) or values[cursor] != KIND_SEP_OPPONENT_ANALYSIS:
        raise AssertionError("shared prefix must be followed by SEP_OPPONENT_ANALYSIS")
    cursor += 1
    if cursor + 3 > len(values) or values[cursor:cursor + 3] != [KIND_OPPONENT_ANALYSIS] * 3:
        raise AssertionError("OPPONENT_ANALYSIS block must contain exactly three rows")
    cursor += 3
    if cursor >= len(values) or values[cursor] != KIND_SEP_ACTIONS:
        raise AssertionError("analysis block must be followed by SEP_ACTIONS")
    cursor += 1
    if cursor >= len(values):
        raise AssertionError("action block is missing")
    pair_flags = values[cursor:]
    if len(pair_flags) % 2 != 0 or any(
        value not in (KIND_ACTION_OFFENSE_QUERY, KIND_ACTION_DEFENSE_QUERY) for value in pair_flags
    ):
        raise AssertionError("action block must be alternating offense/defense pairs")
    for index in range(0, len(pair_flags), 2):
        if pair_flags[index] != KIND_ACTION_OFFENSE_QUERY or pair_flags[index + 1] != KIND_ACTION_DEFENSE_QUERY:
            raise AssertionError("each action pair must be Offense followed by Defense")


def assert_actor_input_semantics(
    actor_factors: np.ndarray,
    actor_numeric: np.ndarray,
    actor_lengths: np.ndarray,
    query_rows: np.ndarray,
    query_action_ids: np.ndarray,
    query_pair_counts: np.ndarray,
    legal_mask: np.ndarray,
    *,
    context_tokens: int = 256,
) -> None:
    """校验 V18 Actor 完整序列的结构、域、排序与 action 集合。"""
    actor_factors = np.asarray(actor_factors)
    actor_numeric = np.asarray(actor_numeric)
    actor_lengths = np.asarray(actor_lengths).astype(np.int64, copy=False)
    query_rows = np.asarray(query_rows)
    query_action_ids = np.asarray(query_action_ids).astype(np.int64, copy=False)
    query_pair_counts = np.asarray(query_pair_counts).astype(np.int64, copy=False)
    legal_mask = np.asarray(legal_mask)

    if actor_factors.ndim != 3 or actor_factors.shape[-1] != TOKEN_ROW_WIDTH:
        raise AssertionError(f"actor_factors must be [batch, tokens, {TOKEN_ROW_WIDTH}]")
    batch, capacity, _ = actor_factors.shape
    if actor_numeric.shape != (batch, capacity, TOKEN_NUMERIC_WIDTH):
        raise AssertionError("actor_numeric shape is malformed")
    if actor_lengths.shape != (batch,) or np.any(actor_lengths <= 0) or np.any(actor_lengths > capacity):
        raise AssertionError("actor_lengths are malformed")
    if query_rows.shape[0] != batch or query_rows.shape[-1] != QUERY_ROW_WIDTH:
        raise AssertionError("query_rows shape is malformed")
    if query_action_ids.shape[0] != batch:
        raise AssertionError("query_action_ids shape is malformed")
    if legal_mask.shape != (batch, NUM_ACTIONS):
        raise AssertionError(f"legal_mask must be [batch, {NUM_ACTIONS}]")

    for row in range(batch):
        length = int(actor_lengths[row])
        if length > int(context_tokens):
            raise AssertionError(f"actor context overflow: {length} > {context_tokens}")
        rows = actor_factors[row, :length]
        kinds = rows[:, 1]
        _assert_actor_canonical_order(kinds, rows)
        # 域校验（按类别 schema）。
        for token_index in range(length):
            kind = int(kinds[token_index])
            if is_separator_kind(kind):
                continue
            schema = CATEGORY_SCHEMAS[kind]
            for field_index, field in enumerate(schema.discrete):
                value = int(rows[token_index, 2 + field_index])
                if not 0 <= value < field.cardinality:
                    raise AssertionError(f"row {token_index} kind {kind} field {field.name} out of range: {value}")
            if not np.all(np.isfinite(actor_numeric[row, token_index])):
                raise AssertionError(f"row {token_index} numeric is non-finite")
            if np.any(np.abs(actor_numeric[row, token_index]) > 1.0):
                raise AssertionError(f"row {token_index} numeric outside [-1,1]")
        # TILE_STATE 守恒。
        tile_rows = rows[kinds == KIND_TILE_STATE]
        for tile_row in tile_rows:
            known = int(tile_row[7])
            unknown = int(tile_row[8])
            if unknown != 4 - min(known, 4):
                raise AssertionError(f"tile-state unknown count violates conservation for kind {tile_row[2]}")
            if bool(tile_row[9]) != (unknown == 0):
                raise AssertionError("tile-state all_seen disagrees with unknown count")
        # RIVER_SUMMARY 有效长度域。
        for summary_index in np.flatnonzero(kinds == KIND_RIVER_SUMMARY):
            valid_length = int(rows[summary_index, 2])
            if valid_length < 0 or valid_length > 6:
                raise AssertionError("river summary valid_length must be 0..6")
            for slot in range(valid_length, 6):
                if rows[summary_index, 3 + 4 * slot:7 + 4 * slot].any():
                    raise AssertionError("river summary padding slot must be all zero")
        # action 集合与 query 行。
        pair_count = int(query_pair_counts[row])
        legal_ids = np.flatnonzero(legal_mask[row]).astype(np.int64)
        if pair_count != len(legal_ids):
            raise AssertionError("query pair count differs from legal-mask set")
        ids = query_action_ids[row, :pair_count]
        if np.any(ids < 0) or np.any(ids >= NUM_ACTIONS):
            raise AssertionError("action id outside action space")
        if len(np.unique(ids)) != pair_count or set(ids.tolist()) != set(legal_ids.tolist()):
            raise AssertionError("query action ids do not equal legal-mask set")
        query_chunk = query_rows[row, : 2 * pair_count].astype(np.int64, copy=False)
        if np.any(query_chunk[0::2, QUERY_ROW_QUERY_TYPE] != 1) or np.any(query_chunk[1::2, QUERY_ROW_QUERY_TYPE] != 2):
            raise AssertionError("query rows must alternate Offense/Defense")
        if not np.array_equal(query_chunk[0::2, QUERY_ROW_ACTION_ID], ids) or not np.array_equal(
            query_chunk[1::2, QUERY_ROW_ACTION_ID], ids
        ):
            raise AssertionError("query action ids disagree with metadata")
        # actor 行中的 action 特征与 query 行一致。
        action_positions = np.flatnonzero(np.isin(kinds, (KIND_ACTION_OFFENSE_QUERY, KIND_ACTION_DEFENSE_QUERY)))
        if len(action_positions) != 2 * pair_count:
            raise AssertionError("actor action rows count disagrees with pair count")
        for index, position in enumerate(action_positions):
            query_value = query_chunk[index]
            expected = (
                int(query_value[QUERY_ROW_ACTION_TYPE]),
                int(query_value[QUERY_ROW_PRIMARY_TILE]),
                int(query_value[QUERY_ROW_SOURCE_SEAT]),
                (int(query_value[QUERY_ROW_ACTION_ID]) - 1) % 2 if 1 <= int(query_value[QUERY_ROW_ACTION_ID]) < 75 else 0,
                *[int(value) for value in query_value[QUERY_ROW_ANSWER_START:QUERY_ROW_WIDTH]],
            )
            actual = [int(rows[position, 2 + offset]) for offset in range(len(expected))]
            if actual != list(expected):
                raise AssertionError("actor action rows disagree with query rows")
        # 域校验 action metadata/supplier。
        action_types = query_chunk[:, QUERY_ROW_ACTION_TYPE]
        source_seats = query_chunk[:, QUERY_ROW_SOURCE_SEAT]
        supplier = np.isin(action_types, tuple(SUPPLIER_REQUIRED_ACTION_TYPES))
        if np.any(source_seats[supplier] == 0) or np.any(source_seats[~supplier] != 0):
            raise AssertionError("supplier/non-supplier source seat domain violated")
        for slot_index, slot in enumerate(("O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9")):
            codes = query_chunk[0::2, QUERY_ROW_ANSWER_START + slot_index]
            if np.any(codes >= SLOT_CARDINALITIES[slot]):
                raise AssertionError(f"offense answer {slot} out of range")
        for slot_index, slot in enumerate(("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9")):
            codes = query_chunk[1::2, QUERY_ROW_ANSWER_START + slot_index]
            if np.any(codes >= SLOT_CARDINALITIES[slot]):
                raise AssertionError(f"defense answer {slot} out of range")


def assert_critic_token_semantics(factors: np.ndarray, lengths: np.ndarray) -> None:
    """Critic 私有行校验：SEP_CRITIC 开头、三家闭手、未来五张、无 Analysis/Action。"""
    factors = np.asarray(factors)
    lengths = np.asarray(lengths).astype(np.int64, copy=False)
    if factors.ndim != 3 or factors.shape[-1] != TOKEN_ROW_WIDTH or lengths.shape != (factors.shape[0],):
        raise AssertionError("critic factors or lengths are malformed")
    for index, length in enumerate(lengths):
        rows = factors[index, :length]
        kinds = rows[:, 1].astype(int)
        segments = rows[:, 0].astype(int)
        if kinds.tolist()[0] != KIND_SEP_CRITIC:
            raise AssertionError("critic rows must start with SEP_CRITIC")
        if np.any(segments[:1] != SEGMENT_CRITIC_PRIVATE):
            raise AssertionError("SEP_CRITIC must be in critic-private segment")
        hand_rows = rows[kinds == KIND_CRITIC_HAND]
        future_rows = rows[kinds == KIND_CRITIC_FUTURE]
        if not hand_rows.size or not future_rows.size:
            raise AssertionError("critic requires hands and future-wall rows")
        if set(hand_rows[:, 2].astype(int).tolist()) != {1, 2, 3}:
            raise AssertionError("critic must contain all three opponent hands")
        if np.any(hand_rows[:, 0].astype(int) != SEGMENT_CRITIC_PRIVATE):
            raise AssertionError("hand rows must be in critic-private segment")
        if np.any(future_rows[:, 0].astype(int) != SEGMENT_CRITIC_FUTURE):
            raise AssertionError("future rows must be in critic-future segment")
        positions = future_rows[:, 2].astype(int)
        if positions.tolist() != list(range(1, 6)):
            raise AssertionError("future wall must contain positions 1..5 in order")
        if np.any(~np.isin(kinds, (KIND_SEP_CRITIC, KIND_CRITIC_HAND, KIND_CRITIC_FUTURE))):
            raise AssertionError("critic contains an analysis/action/unknown kind")
        if np.any(np.isin(segments, (SEGMENT_SHARED, SEGMENT_ANALYSIS, SEGMENT_ACTIONS))):
            raise AssertionError("critic contains shared/analysis/action segments")


def summarize_tokens(factors: np.ndarray, length: int) -> dict[str, Any]:
    """返回稳定的语义摘要（CLI 诊断用）。"""
    rows = np.asarray(factors)[: int(length)]
    kinds = rows[:, 1].astype(int)
    counts = Counter(int(kind) for kind in kinds)
    return {
        "length": int(length),
        "by_kind": {kind: count for kind, count in sorted(counts.items())},
        "separators": sum(1 for kind in kinds if is_separator_kind(int(kind))),
    }
