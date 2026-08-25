"""Semantic assertions and readable summaries for model input tokens.

The Rust state machine owns public-token construction and ``critic_features``
owns centralized-value tokens.  This module deliberately validates their
contract without duplicating either encoder.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .architecture import TOKEN_CARDINALITIES, TOKEN_WIDTH
from .critic_features import (
    FIELD_FUTURE_WALL,
    FIELD_OPPONENT_HAND,
    SEGMENT_CRITIC_FUTURE_WALL,
    SEGMENT_CRITIC_PRIVATE,
    TOKEN_KIND_FUTURE_WALL,
    TOKEN_KIND_TILE_COUNT,
)
from .encoding_protocol import (
    ACTION_TYPE_CARDINALITY,
    DEFENSE_SLOT_ORDER,
    OFFENSE_SLOT_ORDER,
    QUERY_DEFENSE,
    QUERY_OFFENSE,
    QUERY_ROW_ACTION_ID,
    QUERY_ROW_ACTION_TYPE,
    QUERY_ROW_ANSWER_START,
    QUERY_ROW_PRIMARY_TILE,
    QUERY_ROW_QUERY_TYPE,
    QUERY_ROW_SOURCE_SEAT,
    QUERY_ROW_WIDTH,
    SLOT_CARDINALITIES,
    SNAPSHOT_FACTOR_CARDINALITIES,
    SNAPSHOT_FACTOR_WIDTH,
    SNAPSHOT_FIELD_COUNT,
    SNAPSHOT_FIELDS,
    SNAPSHOT_NUMERIC_WIDTH,
    SUPPLIER_REQUIRED_ACTION_TYPES,
)
from .schema import NUM_ACTIONS

SEGMENT_HISTORY = 1
SEGMENT_STATE = 2


def used_rows(factors: np.ndarray, length: int) -> np.ndarray:
    """Return a copied-free view of one row's non-padding token prefix."""
    array = np.asarray(factors)
    if array.ndim != 2 or array.shape[1] != TOKEN_WIDTH:
        raise AssertionError(f"expected [tokens, {TOKEN_WIDTH}] factors, got {array.shape}")
    if not 0 <= int(length) <= array.shape[0]:
        raise AssertionError(f"invalid token length {length} for {array.shape[0]} rows")
    return array[:int(length)]


def _assert_factor_ranges(rows: np.ndarray, *, label: str) -> None:
    if rows.size == 0:
        return
    if np.any(rows < 0):
        raise AssertionError(f"{label} contains negative categorical factors")
    for column, cardinality in enumerate(TOKEN_CARDINALITIES):
        if np.any(rows[:, column] >= cardinality):
            maximum = int(rows[:, column].max())
            raise AssertionError(f"{label} factor {column} has {maximum}, capacity is {cardinality - 1}")


def _assert_history_token_semantics(factors: np.ndarray, numeric: np.ndarray, lengths: np.ndarray) -> None:
    """校验 Objective Facts 行的可见性与基础形状。"""
    factors = np.asarray(factors)
    numeric = np.asarray(numeric)
    lengths = np.asarray(lengths)
    if factors.ndim != 3 or factors.shape[-1] != TOKEN_WIDTH:
        raise AssertionError(f"history factors must be [batch, tokens, {TOKEN_WIDTH}], got {factors.shape}")
    if numeric.shape != (*factors.shape[:2], 8) or lengths.shape != (factors.shape[0],):
        raise AssertionError("history numeric or length shape is malformed")
    for index, length in enumerate(lengths):
        rows = used_rows(factors[index], int(length))
        _assert_factor_ranges(rows, label=f"history[{index}]")
        if np.any(~np.isfinite(numeric[index, :int(length)])):
            raise AssertionError(f"history[{index}] contains non-finite numeric features")
        if np.any(~np.isin(rows[:, 0], (SEGMENT_HISTORY, SEGMENT_STATE))):
            raise AssertionError(f"history[{index}] contains a critic-only or unknown segment")
        if np.any(rows[:, 9] == 2):
            raise AssertionError(f"history[{index}] has hidden information")


def _assert_lengths(
    lengths: np.ndarray,
    *,
    batch: int,
    capacity: int,
    label: str,
) -> None:
    """校验分段长度是一行一个、左对齐且不越界。"""
    if lengths.shape != (batch,):
        raise AssertionError(f"{label} must have one entry per batch row")
    if np.any(lengths < 0) or np.any(lengths > capacity):
        raise AssertionError(f"{label} exceed supplied rows")


def assert_actor_input_semantics(
    history_factors: np.ndarray,
    history_numeric: np.ndarray,
    history_lengths: np.ndarray,
    snapshot_factors: np.ndarray,
    snapshot_numeric: np.ndarray,
    snapshot_lengths: np.ndarray,
    query_rows: np.ndarray,
    query_action_ids: np.ndarray,
    query_pair_counts: np.ndarray,
    legal_mask: np.ndarray,
    *,
    context_tokens: int = 4096,
) -> None:
    """校验 V18 Actor 输入三段结构、固定 Snapshot 与动作集合。

    本函数只验证输入协议不变量,不重新计算麻将业务事实;事实正确性由
    bridge 等价测试与 query/snapshot oracle 测试覆盖。
    """
    history_factors = np.asarray(history_factors)
    history_numeric = np.asarray(history_numeric)
    history_lengths = np.asarray(history_lengths)
    snapshot_factors = np.asarray(snapshot_factors)
    snapshot_numeric = np.asarray(snapshot_numeric)
    snapshot_lengths = np.asarray(snapshot_lengths)
    query_rows = np.asarray(query_rows)
    query_action_ids = np.asarray(query_action_ids)
    query_pair_counts = np.asarray(query_pair_counts)
    legal_mask = np.asarray(legal_mask)

    if history_factors.ndim != 3 or history_factors.shape[-1] != TOKEN_WIDTH:
        raise AssertionError(
            f"history_factors must be [batch, tokens, {TOKEN_WIDTH}]"
        )
    batch, history_capacity, _width = history_factors.shape
    if history_numeric.shape != (*history_factors.shape[:2], 8):
        raise AssertionError("history_numeric shape is malformed")
    expected_snapshot = (batch, SNAPSHOT_FIELD_COUNT, SNAPSHOT_FACTOR_WIDTH)
    if snapshot_factors.shape != expected_snapshot:
        raise AssertionError(f"snapshot_factors must be {expected_snapshot}")
    if snapshot_numeric.shape != (batch, SNAPSHOT_FIELD_COUNT, SNAPSHOT_NUMERIC_WIDTH):
        raise AssertionError("snapshot_numeric shape is malformed")
    if query_rows.ndim != 3 or query_rows.shape[-1] != QUERY_ROW_WIDTH:
        raise AssertionError(
            f"query_rows must be [batch, rows, {QUERY_ROW_WIDTH}]"
        )
    if query_rows.shape[0] != batch:
        raise AssertionError("query batch size differs from history")
    if query_action_ids.ndim != 2 or query_action_ids.shape[0] != batch:
        raise AssertionError("query_action_ids shape is malformed")
    if legal_mask.shape != (batch, NUM_ACTIONS):
        raise AssertionError(f"legal_mask must be [batch, {NUM_ACTIONS}]")
    if legal_mask.dtype != np.bool_ and not np.isin(legal_mask, (False, True)).all():
        raise AssertionError("legal_mask must contain booleans")

    _assert_lengths(
        history_lengths,
        batch=batch,
        capacity=history_capacity,
        label="history_lengths",
    )
    _assert_lengths(
        snapshot_lengths,
        batch=batch,
        capacity=SNAPSHOT_FIELD_COUNT,
        label="snapshot_lengths",
    )
    _assert_lengths(
        query_pair_counts,
        batch=batch,
        capacity=query_action_ids.shape[1],
        label="query_pair_counts",
    )
    if np.any(2 * query_pair_counts > query_rows.shape[1]):
        raise AssertionError("query_pair_counts exceed query_rows capacity")
    total_lengths = history_lengths + snapshot_lengths + 2 * query_pair_counts
    if np.any(total_lengths > int(context_tokens)):
        raise AssertionError(
            f"V18 actor context overflow: max={int(total_lengths.max())} "
            f"limit={int(context_tokens)}"
        )

    _assert_history_token_semantics(
        history_factors, history_numeric, history_lengths,
    )
    if np.any(snapshot_lengths != SNAPSHOT_FIELD_COUNT):
        raise AssertionError(f"snapshot_lengths must all equal {SNAPSHOT_FIELD_COUNT}")
    if np.any(~np.isfinite(snapshot_numeric)):
        raise AssertionError("snapshot numeric contains non-finite values")

    for row in range(batch):
        factors = snapshot_factors[row].astype(np.int64, copy=False)
        if np.any(factors < 0):
            raise AssertionError(f"snapshot[{row}] contains negative factors")
        for column, cardinality in enumerate(SNAPSHOT_FACTOR_CARDINALITIES):
            if np.any(factors[:, column] >= cardinality):
                raise AssertionError(f"snapshot[{row}] factor {column} out of range")
        expected_ids = np.arange(1, SNAPSHOT_FIELD_COUNT + 1)
        if not np.array_equal(factors[:, 0], expected_ids):
            raise AssertionError(f"snapshot[{row}] field order differs from Rust schema")
        for index, field in enumerate(SNAPSHOT_FIELDS):
            values = factors[index]
            if values[1] != field.relative_seat:
                raise AssertionError(f"snapshot[{row}] relative seat differs for {field.name}")
            if values[2] > field.categorical_max or values[3] > field.tile_max:
                raise AssertionError(f"snapshot[{row}] domain violation for {field.name}")
            numeric_value = float(snapshot_numeric[row, index, 0])
            if (not field.numeric and numeric_value != 0.0) or abs(numeric_value) > 1.0:
                raise AssertionError(f"snapshot[{row}] numeric violation for {field.name}")

        pair_count = int(query_pair_counts[row])
        legal_ids = np.flatnonzero(legal_mask[row]).astype(np.int32)
        if pair_count != len(legal_ids):
            raise AssertionError(
                f"query[{row}] pair count {pair_count} differs from "
                f"legal count {len(legal_ids)}"
            )
        action_ids = query_action_ids[row, :pair_count].astype(np.int32)
        if np.any(action_ids < 0) or np.any(action_ids >= NUM_ACTIONS):
            raise AssertionError(f"query[{row}] action id outside action space")
        if len(np.unique(action_ids)) != pair_count or set(action_ids) != set(legal_ids):
            raise AssertionError(f"query[{row}] action ids do not equal legal-mask set")
        rows = query_rows[row, : 2 * pair_count].astype(np.int64, copy=False)
        if pair_count == 0:
            raise AssertionError(f"query[{row}] has no legal action pairs")
        if np.any(rows < 0):
            raise AssertionError(f"query[{row}] contains negative values")
        if np.any(rows[0::2, QUERY_ROW_QUERY_TYPE] != QUERY_OFFENSE):
            raise AssertionError(f"query[{row}] offense rows are malformed")
        if np.any(rows[1::2, QUERY_ROW_QUERY_TYPE] != QUERY_DEFENSE):
            raise AssertionError(f"query[{row}] defense rows are malformed")
        if (
            not np.array_equal(rows[0::2, QUERY_ROW_ACTION_ID], action_ids)
            or not np.array_equal(rows[1::2, QUERY_ROW_ACTION_ID], action_ids)
        ):
            raise AssertionError(f"query[{row}] paired action ids differ")
        if not np.array_equal(rows[0::2, 2:5], rows[1::2, 2:5]):
            raise AssertionError(f"query[{row}] paired metadata differs")
        if np.any(rows[:, QUERY_ROW_ACTION_TYPE] >= ACTION_TYPE_CARDINALITY):
            raise AssertionError(f"query[{row}] action type out of range")
        if np.any(rows[:, QUERY_ROW_PRIMARY_TILE] > 34):
            raise AssertionError(f"query[{row}] primary tile out of range")
        if np.any(rows[:, QUERY_ROW_SOURCE_SEAT] > 4):
            raise AssertionError(f"query[{row}] source seat out of range")
        pair_action_types = rows[0::2, QUERY_ROW_ACTION_TYPE]
        pair_source_seats = rows[0::2, QUERY_ROW_SOURCE_SEAT]
        supplier_required = np.isin(
            pair_action_types,
            tuple(SUPPLIER_REQUIRED_ACTION_TYPES),
        )
        if np.any(pair_source_seats[supplier_required] == 0):
            raise AssertionError(f"query[{row}] supplier action lacks source seat")
        if np.any(pair_source_seats[~supplier_required] != 0):
            raise AssertionError(f"query[{row}] non-supplier action has source seat")

        offense_answers = rows[0::2, QUERY_ROW_ANSWER_START:]
        defense_answers = rows[1::2, QUERY_ROW_ANSWER_START:]
        for index, slot in enumerate(OFFENSE_SLOT_ORDER):
            if np.any(offense_answers[:, index] >= SLOT_CARDINALITIES[slot]):
                raise AssertionError(f"query[{row}] {slot} answer out of range")
        for index, slot in enumerate(DEFENSE_SLOT_ORDER):
            if np.any(defense_answers[:, index] >= SLOT_CARDINALITIES[slot]):
                raise AssertionError(f"query[{row}] {slot} answer out of range")


def assert_critic_token_semantics(factors: np.ndarray, lengths: np.ndarray, *, include_public_state: bool = False) -> None:
    """Validate that centralized critic tokens contain opponent hands and the
    critic-only ordered future live-wall snapshot."""
    factors = np.asarray(factors)
    lengths = np.asarray(lengths)
    if factors.ndim != 3 or factors.shape[-1] != TOKEN_WIDTH or lengths.shape != (factors.shape[0],):
        raise AssertionError("critic factors or lengths are malformed")
    for index, length in enumerate(lengths):
        rows = used_rows(factors[index], int(length))
        _assert_factor_ranges(rows, label=f"critic[{index}]")
        if np.any(rows[:, 9] != 1):
            raise AssertionError(f"critic[{index}] contains non-visible tokens")
        if np.any(~np.isin(rows[:, 0], (SEGMENT_CRITIC_PRIVATE, SEGMENT_CRITIC_FUTURE_WALL))):
            raise AssertionError(f"critic[{index}] contains an unknown segment")
        private = rows[rows[:, 0] == SEGMENT_CRITIC_PRIVATE]
        if private.size:
            if np.any(private[:, 1] != TOKEN_KIND_TILE_COUNT):
                raise AssertionError(f"critic[{index}] contains an unknown token kind")
            if np.any(private[:, 2] != FIELD_OPPONENT_HAND):
                raise AssertionError(f"critic[{index}] contains an unknown tile-count field")
            if np.any(~np.isin(private[:, 3], (2, 3, 4))):
                raise AssertionError(f"critic[{index}] has malformed opponent hands")
            if set(private[:, 3].tolist()) != {2, 3, 4}:
                raise AssertionError(f"critic[{index}] must contain all three opponent hands")
        future = rows[rows[:, 0] == SEGMENT_CRITIC_FUTURE_WALL]
        if future.size:
            if np.any(future[:, 1] != TOKEN_KIND_FUTURE_WALL):
                raise AssertionError(
                    f"critic[{index}] has an unknown future-wall token kind"
                )
            if np.any(future[:, 2] != FIELD_FUTURE_WALL):
                raise AssertionError(
                    f"critic[{index}] has an unknown future-wall field"
                )
            positions = future[:, 3]
            if np.any(~np.isin(positions, (1, 2, 3, 4, 5))):
                raise AssertionError(
                    f"critic[{index}] has an out-of-range future-wall position"
                )
            if len(set(positions.tolist())) != len(positions):
                raise AssertionError(
                    f"critic[{index}] repeats a future-wall position"
                )
            if not np.array_equal(positions, np.arange(1, 6)):
                raise AssertionError(
                    f"critic[{index}] must contain exactly ordered future-wall positions 1..5"
                )
            if np.any(future[:, 7] != 1) or np.any(future[:, 8] != 0):
                raise AssertionError(
                    f"critic[{index}] has malformed future-wall count/flag slots"
                )
            if np.any(~np.isin(future[:, 4], (1, 2, 3, 4))) or np.any(
                ~np.isin(future[:, 5], (1, 2, 3, 4, 5, 6, 7, 8, 9))
            ):
                raise AssertionError(
                    f"critic[{index}] has an invalid future-wall suit/rank"
                )
            honors = future[future[:, 4] == 4]
            if honors.size and np.any(honors[:, 5] > 7):
                raise AssertionError(
                    f"critic[{index}] has an invalid honor rank"
                )
            red = future[future[:, 6] == 1]
            if red.size and np.any((red[:, 5] != 5) | ~np.isin(red[:, 4], (1, 2, 3))):
                raise AssertionError(
                    f"critic[{index}] marks a non-red-five future-wall tile"
                )
        if len(future) != 5 or not len(private):
            raise AssertionError(
                f"critic[{index}] requires three hands and exactly five future tiles"
            )
        if np.any(rows[:len(private), 0] != SEGMENT_CRITIC_PRIVATE):
            raise AssertionError(f"critic[{index}] private hands must precede future wall")


def summarize_tokens(factors: np.ndarray, length: int) -> dict[str, Any]:
    """Return a stable, compact semantic summary for CLI diagnostics."""
    rows = used_rows(factors, length)
    counts = Counter((int(row[0]), int(row[1]), int(row[2])) for row in rows)
    return {
        "length": int(length),
        "by_segment_kind_field": {
            f"{segment}/{kind}/{field}": count
            for (segment, kind, field), count in sorted(counts.items())
        },
        "red_five_tokens": int(np.count_nonzero(rows[:, 6] == 1)),
    }
