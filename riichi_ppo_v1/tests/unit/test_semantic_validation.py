import numpy as np
import pytest

from riichi_ppo_v1.model.critic_features import (
    FIELD_FUTURE_WALL,
    SEGMENT_CRITIC_FUTURE_WALL,
    TOKEN_KIND_FUTURE_WALL,
    TOKEN_KIND_TILE_COUNT,
)
from riichi_ppo_v1.model.semantic_validation import (
    assert_critic_token_semantics,
    assert_v16_actor_input_semantics,
    summarize_tokens,
)


def v16_inputs() -> dict[str, np.ndarray]:
    history_factors = np.zeros((1, 2, 10), dtype=np.uint8)
    history_factors[0, 0] = [1, 1, 4, 2, 1, 5, 1, 0, 2, 1]
    history_factors[0, 1] = [2, 2, 1, 1, 0, 0, 0, 0, 0, 1]
    history_numeric = np.zeros((1, 2, 8), dtype=np.float32)
    history_lengths = np.asarray([2], dtype=np.int64)
    snapshot_kinds = np.asarray([[0, 1, 2, 3]], dtype=np.uint8)
    snapshot_cat = np.zeros((1, 4, 4), dtype=np.uint8)
    snapshot_num = np.zeros((1, 4, 7), dtype=np.float32)
    query_rows = np.zeros((1, 2, 15), dtype=np.int32)
    query_rows[0, 0, 0] = 1
    query_rows[0, 1, 0] = 2
    query_action_ids = np.asarray([[0]], dtype=np.int32)
    query_pair_counts = np.asarray([1], dtype=np.int64)
    legal_mask = np.zeros((1, 241), dtype=np.bool_)
    legal_mask[0, 0] = True
    return {
        "history_factors": history_factors,
        "history_numeric": history_numeric,
        "history_lengths": history_lengths,
        "snapshot_kinds": snapshot_kinds,
        "snapshot_cat": snapshot_cat,
        "snapshot_num": snapshot_num,
        "snapshot_lengths": np.asarray([4], dtype=np.int64),
        "query_rows": query_rows,
        "query_action_ids": query_action_ids,
        "query_pair_counts": query_pair_counts,
        "legal_mask": legal_mask,
    }


def test_v16_actor_validator_accepts_valid_segments_and_summarizes_reds() -> None:
    data = v16_inputs()
    assert_v16_actor_input_semantics(**data)
    assert summarize_tokens(data["history_factors"][0], 2)["red_five_tokens"] == 1


def test_v16_actor_validator_rejects_hidden_history_information() -> None:
    data = v16_inputs()
    data["history_factors"][0, 0, 9] = 2
    with pytest.raises(AssertionError, match="hidden information"):
        assert_v16_actor_input_semantics(**data)


def test_v16_actor_validator_rejects_critic_only_history_segment() -> None:
    data = v16_inputs()
    data["history_factors"][0, 0, 0] = SEGMENT_CRITIC_FUTURE_WALL
    data["history_factors"][0, 0, 1] = TOKEN_KIND_FUTURE_WALL
    data["history_factors"][0, 0, 2] = FIELD_FUTURE_WALL
    with pytest.raises(AssertionError, match="critic-only or unknown segment"):
        assert_v16_actor_input_semantics(**data)


def test_v16_actor_validator_requires_query_ids_to_match_legal_mask() -> None:
    data = v16_inputs()
    data["query_action_ids"][0, 0] = 5
    data["query_rows"][0, :, 1] = 5
    with pytest.raises(AssertionError, match="legal mask"):
        assert_v16_actor_input_semantics(**data)


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
    rows[0, 4] = 4
    rows[0, 6] = 1
    factors = rows[None, :]
    lengths = np.asarray([5], dtype=np.int64)
    with pytest.raises(AssertionError, match="non-red-five future-wall tile"):
        assert_critic_token_semantics(factors, lengths)
