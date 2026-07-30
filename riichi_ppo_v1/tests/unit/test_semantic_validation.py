import numpy as np
import pytest

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
