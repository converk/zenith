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
    FIELD_PUBLIC_MELD_TILE,
    FIELD_PUBLIC_RIVER,
    SEGMENT_CRITIC_FUTURE_WALL,
    SEGMENT_CRITIC_PRIVATE,
    SEGMENT_PUBLIC_SUMMARY,
    TOKEN_KIND_FUTURE_WALL,
    TOKEN_KIND_MELD,
    TOKEN_KIND_TILE_COUNT,
)

SEGMENT_HISTORY = 1
SEGMENT_STATE = 2

_PUBLIC_FIELDS = frozenset({FIELD_PUBLIC_RIVER, FIELD_PUBLIC_MELD_TILE})


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


def assert_actor_token_semantics(factors: np.ndarray, numeric: np.ndarray, lengths: np.ndarray) -> None:
    """Validate actor visibility and public-token shape invariants.

    An actor row may include public history/state and its own state suffix;
    no concealed-opponent representation belongs in this input.
    """
    factors = np.asarray(factors)
    numeric = np.asarray(numeric)
    lengths = np.asarray(lengths)
    if factors.ndim != 3 or factors.shape[-1] != TOKEN_WIDTH:
        raise AssertionError(f"actor factors must be [batch, tokens, {TOKEN_WIDTH}], got {factors.shape}")
    if numeric.shape != (*factors.shape[:2], 8) or lengths.shape != (factors.shape[0],):
        raise AssertionError("actor numeric or length shape is malformed")
    for index, length in enumerate(lengths):
        rows = used_rows(factors[index], int(length))
        _assert_factor_ranges(rows, label=f"actor[{index}]")
        if np.any(~np.isfinite(numeric[index, :int(length)])):
            raise AssertionError(f"actor[{index}] contains non-finite numeric features")
        contract_rows = np.isin(rows[:, 0], (6, 7))
        if np.any(np.abs(numeric[index, :int(length)][contract_rows]) > 1.0):
            maximum = float(np.abs(numeric[index, :int(length)][contract_rows]).max())
            raise AssertionError(
                f"actor[{index}] schema-13 numeric feature exceeds frozen range: {maximum}"
            )
        if np.any(~np.isin(rows[:, 0], (SEGMENT_HISTORY, SEGMENT_STATE, SEGMENT_PUBLIC_SUMMARY, 6, 7))):
            raise AssertionError(f"actor[{index}] contains a critic-only or unknown segment")
        if np.any((rows[:, 0] != 7) & (rows[:, 9] == 2)):
            raise AssertionError(f"actor[{index}] has hidden information")
        queries = rows[:, 0] == 7
        if np.any(queries):
            first = int(np.flatnonzero(queries)[0])
            if np.any(queries[:first]) or np.any(~queries[first:]):
                raise AssertionError(f"actor[{index}] action queries are not a suffix")
            query_rows = rows[first:]
            if len(query_rows) % 2 or np.any(query_rows[0::2, 9] != 1) or np.any(query_rows[1::2, 9] != 2):
                raise AssertionError(f"actor[{index}] action queries are not offense/defense pairs")
            if np.any(query_rows[0::2, 2] != query_rows[1::2, 2]):
                raise AssertionError(f"actor[{index}] paired action ids differ")
        public = rows[rows[:, 0] == SEGMENT_PUBLIC_SUMMARY]
        if public.size:
            if np.any(public[:, 9] != 1) or np.any(~np.isin(public[:, 3], (1, 2, 3, 4))):
                raise AssertionError(f"actor[{index}] has malformed public summary visibility or seats")
            counts = public[public[:, 1] == TOKEN_KIND_TILE_COUNT]
            if np.any(~np.isin(counts[:, 2], tuple(_PUBLIC_FIELDS))):
                raise AssertionError(f"actor[{index}] has malformed public summary fields")
            melds = public[public[:, 1] == TOKEN_KIND_MELD]
            if melds.size and np.any(~np.isin(melds[:, 2], (1, 2, 3, 4, 5))):
                raise AssertionError(f"actor[{index}] has malformed public meld headers")


def assert_critic_token_semantics(factors: np.ndarray, lengths: np.ndarray, *, include_public_state: bool = False) -> None:
    """Validate that centralized critic tokens contain opponent hands and the
    critic-only ordered future live-wall snapshot (V14)."""
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
