import numpy as np
import pytest

from riichi_ppo_v1.model.critic_features import (
    FIELD_FUTURE_WALL,
    SEGMENT_CRITIC_FUTURE_WALL,
    TOKEN_KIND_FUTURE_WALL,
    TOKEN_KIND_TILE_COUNT,
)
from riichi_ppo_v1.model.semantic_validation import (
    assert_actor_token_semantics,
    assert_critic_token_semantics,
    summarize_tokens,
)


def actor_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factors = np.zeros((1, 1, 10), dtype=np.uint8)
    factors[0, 0] = [1, 1, 4, 2, 1, 5, 1, 0, 2, 1]  # public red tsumogiri
    return factors, np.zeros((1, 1, 8), dtype=np.float32), np.asarray([1], dtype=np.int64)


def test_actor_validator_accepts_public_actor_tokens_without_opponent_masks() -> None:
    factors, numeric, lengths = actor_rows()
    assert_actor_token_semantics(factors, numeric, lengths)
    assert summarize_tokens(factors[0], 1)["red_five_tokens"] == 1


def test_actor_validator_rejects_hidden_information() -> None:
    factors, numeric, lengths = actor_rows()
    factors[0, 0, 9] = 2
    with pytest.raises(AssertionError, match="hidden information"):
        assert_actor_token_semantics(factors, numeric, lengths)


def test_critic_validator_rejects_public_state() -> None:
    factors = np.asarray([[
        [4, 4, 2, 2, 1, 5, 1, 1, 0, 1],
        [4, 4, 3, 2, 1, 5, 0, 1, 0, 1],
    ]], dtype=np.uint8)
    lengths = np.asarray([2], dtype=np.int64)
    with pytest.raises(AssertionError, match="unknown tile-count field"):
        assert_critic_token_semantics(factors, lengths)


def future_rows() -> np.ndarray:
    rows = []
    for position in range(1, 6):
        rows.append([
            SEGMENT_CRITIC_FUTURE_WALL,
            TOKEN_KIND_FUTURE_WALL,
            FIELD_FUTURE_WALL,
            position,
            1,
            5,
            0,
            1,
            0,
            1,
        ])
    return np.asarray(rows, dtype=np.uint8)


def test_critic_validator_accepts_opponent_hands_and_ordered_future_wall() -> None:
    rows = [
        [4, 4, 2, 2, 1, 5, 1, 1, 0, 1],
        [4, 4, 2, 3, 2, 1, 0, 1, 0, 1],
        *[row.tolist() for row in future_rows()],
    ]
    factors = np.asarray([rows], dtype=np.uint8)
    lengths = np.asarray([len(rows)], dtype=np.int64)
    assert_critic_token_semantics(factors, lengths)


def test_critic_validator_rejects_duplicate_future_wall_positions() -> None:
    rows = future_rows()
    rows[1, 3] = 1
    factors = rows[None, :]
    lengths = np.asarray([5], dtype=np.int64)
    with pytest.raises(AssertionError, match="repeats a future-wall position"):
        assert_critic_token_semantics(factors, lengths)


def test_critic_validator_rejects_out_of_range_future_wall_position() -> None:
    rows = future_rows()
    rows[0, 3] = 6
    factors = rows[None, :]
    lengths = np.asarray([5], dtype=np.int64)
    with pytest.raises(AssertionError, match="out-of-range future-wall position"):
        assert_critic_token_semantics(factors, lengths)


def test_critic_validator_rejects_wrong_future_wall_kind() -> None:
    rows = future_rows()
    rows[0, 1] = TOKEN_KIND_TILE_COUNT
    factors = rows[None, :]
    lengths = np.asarray([5], dtype=np.int64)
    with pytest.raises(AssertionError, match="unknown future-wall token kind"):
        assert_critic_token_semantics(factors, lengths)


def test_critic_validator_rejects_invalid_red_five_flag() -> None:
    rows = future_rows()
    rows[0, 4] = 4  # honor tile marked red
    rows[0, 6] = 1
    factors = rows[None, :]
    lengths = np.asarray([5], dtype=np.int64)
    with pytest.raises(AssertionError, match="non-red-five future-wall tile"):
        assert_critic_token_semantics(factors, lengths)


def test_actor_validator_rejects_critic_only_future_wall_segment() -> None:
    factors, numeric, lengths = actor_rows()
    factors[0, 0, 0] = SEGMENT_CRITIC_FUTURE_WALL
    factors[0, 0, 1] = TOKEN_KIND_FUTURE_WALL
    factors[0, 0, 2] = FIELD_FUTURE_WALL
    with pytest.raises(AssertionError, match="critic-only or unknown segment"):
        assert_actor_token_semantics(factors, numeric, lengths)
