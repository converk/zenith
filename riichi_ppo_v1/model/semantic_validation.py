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
    FIELD_OPPONENT_HAND,
    FIELD_OPPONENT_MELD_TILE,
    FIELD_OPPONENT_RIVER,
    SEGMENT_CRITIC_PRIVATE,
    TOKEN_KIND_MELD,
    TOKEN_KIND_TILE_COUNT,
)

SEGMENT_HISTORY = 1
SEGMENT_STATE = 2
KIND_OPPONENT_HIDDEN_MASK = 7

_CRITIC_FIELDS = frozenset({FIELD_OPPONENT_HAND, FIELD_OPPONENT_RIVER, FIELD_OPPONENT_MELD_TILE})


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

    An actor row may include public history/state and its own state suffix.  A
    hidden visibility value is permitted only for the three opponent closed-
    hand *mask* rows; it must never carry a concealed tile identity.
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
        if np.any(~np.isin(rows[:, 0], (SEGMENT_HISTORY, SEGMENT_STATE))):
            raise AssertionError(f"actor[{index}] contains a critic-only or unknown segment")
        hidden = rows[:, 9] == 2
        if np.any(hidden & ~((rows[:, 0] == SEGMENT_STATE) & (rows[:, 1] == KIND_OPPONENT_HIDDEN_MASK))):
            raise AssertionError(f"actor[{index}] has hidden tile-bearing information")
        masks = rows[(rows[:, 0] == SEGMENT_STATE) & (rows[:, 1] == KIND_OPPONENT_HIDDEN_MASK)]
        if masks.shape[0] != 3 or masks[:, 3].tolist() != [2, 3, 4] or not np.all(masks[:, 9] == 2):
            raise AssertionError(f"actor[{index}] must contain exactly three opaque opponent masks")


def assert_critic_token_semantics(factors: np.ndarray, lengths: np.ndarray, *, include_public_state: bool) -> None:
    """Validate centralized critic token schema and opt-in public projection."""
    factors = np.asarray(factors)
    lengths = np.asarray(lengths)
    if factors.ndim != 3 or factors.shape[-1] != TOKEN_WIDTH or lengths.shape != (factors.shape[0],):
        raise AssertionError("critic factors or lengths are malformed")
    for index, length in enumerate(lengths):
        rows = used_rows(factors[index], int(length))
        _assert_factor_ranges(rows, label=f"critic[{index}]")
        if np.any(rows[:, 0] != SEGMENT_CRITIC_PRIVATE) or np.any(rows[:, 9] != 1):
            raise AssertionError(f"critic[{index}] contains non-private or non-visible tokens")
        if np.any(~np.isin(rows[:, 1], (TOKEN_KIND_TILE_COUNT, TOKEN_KIND_MELD))):
            raise AssertionError(f"critic[{index}] contains an unknown token kind")
        count_rows = rows[rows[:, 1] == TOKEN_KIND_TILE_COUNT]
        if np.any(~np.isin(count_rows[:, 2], tuple(_CRITIC_FIELDS))):
            raise AssertionError(f"critic[{index}] contains an unknown tile-count field")
        hands = count_rows[count_rows[:, 2] == FIELD_OPPONENT_HAND]
        if np.any(hands[:, 1] != TOKEN_KIND_TILE_COUNT) or np.any(~np.isin(hands[:, 3], (2, 3, 4))):
            raise AssertionError(f"critic[{index}] has malformed opponent hands")
        public = count_rows[np.isin(count_rows[:, 2], (FIELD_OPPONENT_RIVER, FIELD_OPPONENT_MELD_TILE))]
        if not include_public_state and public.size:
            raise AssertionError(f"critic[{index}] emitted opt-in public state while disabled")
        if public.size and np.any(~np.isin(public[:, 3], (2, 3, 4))):
            raise AssertionError(f"critic[{index}] has malformed opponent public state")
        meld_rows = rows[rows[:, 1] == TOKEN_KIND_MELD]
        if meld_rows.size and (np.any(~np.isin(meld_rows[:, 2], (1, 2, 3, 4, 5))) or np.any(~np.isin(meld_rows[:, 3], (2, 3, 4)))):
            raise AssertionError(f"critic[{index}] has malformed meld header")


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
