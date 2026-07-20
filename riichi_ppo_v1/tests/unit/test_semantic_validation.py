import numpy as np
import pytest

from riichi_ppo_v1.model.semantic_validation import (
    assert_actor_token_semantics,
    assert_critic_token_semantics,
    summarize_tokens,
)


def actor_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factors = np.zeros((1, 4, 10), dtype=np.uint8)
    factors[0, 0] = [1, 1, 4, 2, 1, 5, 1, 0, 2, 1]  # public red tsumogiri
    for index, seat in enumerate((2, 3, 4), start=1):
        factors[0, index] = [2, 7, 1, seat, 0, 0, 0, 0, 0, 2]
    return factors, np.zeros((1, 4, 8), dtype=np.float32), np.asarray([4], dtype=np.int64)


def test_actor_validator_accepts_only_opaque_opponent_masks() -> None:
    factors, numeric, lengths = actor_rows()
    assert_actor_token_semantics(factors, numeric, lengths)
    assert summarize_tokens(factors[0], 4)["red_five_tokens"] == 1


def test_actor_validator_rejects_hidden_tile_information() -> None:
    factors, numeric, lengths = actor_rows()
    factors[0, 0, 9] = 2
    with pytest.raises(AssertionError, match="hidden tile-bearing"):
        assert_actor_token_semantics(factors, numeric, lengths)


def test_critic_validator_enforces_opt_in_public_state() -> None:
    factors = np.asarray([[
        [4, 4, 2, 2, 1, 5, 1, 1, 0, 1],
        [4, 4, 3, 2, 1, 5, 0, 1, 0, 1],
    ]], dtype=np.uint8)
    lengths = np.asarray([2], dtype=np.int64)
    assert_critic_token_semantics(factors, lengths, include_public_state=True)
    with pytest.raises(AssertionError, match="while disabled"):
        assert_critic_token_semantics(factors, lengths, include_public_state=False)
