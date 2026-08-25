from types import SimpleNamespace

import numpy as np

from riichi_ppo_v1.model.critic_features import (
    FIELD_FUTURE_WALL,
    FIELD_OPPONENT_HAND,
    SEGMENT_CRITIC_FUTURE_WALL,
    SEGMENT_CRITIC_PRIVATE,
    TOKEN_KIND_FUTURE_WALL,
    collect_visible_table_state,
    encode_future_wall_tokens,
    encode_critic_features,
)
import pytest


def observation(seat: int, hand: list[int]):
    hands = [[], [], [], []]
    hands[seat] = hand
    return SimpleNamespace(hands=hands)


def test_collects_true_hands_from_four_self_observations_and_preserves_red_fives() -> None:
    observations = {
        0: observation(0, [0, 4, 8]),
        1: observation(1, [16, 17]),
        2: observation(2, [52]),
        3: observation(3, [108]),
    }

    table = collect_visible_table_state(observations)
    features = encode_critic_features(table, observer=0)

    hand_rows = features.factors[features.factors[:, 2] == FIELD_OPPONENT_HAND]
    assert hand_rows[:, 3].tolist() == [2, 2, 3, 4]
    assert hand_rows[:, 6].tolist() == [0, 1, 1, 0]
    assert any(row.tolist() == [4, 4, 2, 2, 1, 5, 1, 1, 0, 1] for row in hand_rows)
    assert any(row.tolist() == [4, 4, 2, 2, 1, 5, 0, 1, 0, 1] for row in hand_rows)

    assert features.length == 4
    assert features.factors.shape == (4, 10)


def test_future_wall_encodes_five_ordered_tokens() -> None:
    wall = [16, 52, 88, 108, 0]
    rows = encode_future_wall_tokens(wall)

    assert rows == [
        (SEGMENT_CRITIC_FUTURE_WALL, TOKEN_KIND_FUTURE_WALL, FIELD_FUTURE_WALL, 1, 1, 5, 1, 1, 0, 1),
        (SEGMENT_CRITIC_FUTURE_WALL, TOKEN_KIND_FUTURE_WALL, FIELD_FUTURE_WALL, 2, 2, 5, 1, 1, 0, 1),
        (SEGMENT_CRITIC_FUTURE_WALL, TOKEN_KIND_FUTURE_WALL, FIELD_FUTURE_WALL, 3, 3, 5, 1, 1, 0, 1),
        (SEGMENT_CRITIC_FUTURE_WALL, TOKEN_KIND_FUTURE_WALL, FIELD_FUTURE_WALL, 4, 4, 1, 0, 1, 0, 1),
        (SEGMENT_CRITIC_FUTURE_WALL, TOKEN_KIND_FUTURE_WALL, FIELD_FUTURE_WALL, 5, 1, 1, 0, 1, 0, 1),
    ]
    assert len(rows) == 5
    assert [row[3] for row in rows] == [1, 2, 3, 4, 5]


def test_future_wall_rejects_missing_tiles() -> None:
    with pytest.raises(ValueError, match="exactly five"):
        encode_future_wall_tokens([16, 52])


def test_future_wall_rejects_invalid_tile_ids() -> None:
    with pytest.raises(ValueError, match="invalid RiichiEnv tile id"):
        encode_future_wall_tokens([136, 0, 4, 8, 12])
    with pytest.raises(ValueError, match="invalid RiichiEnv tile id"):
        encode_future_wall_tokens([-1, 0, 4, 8, 12])
    with pytest.raises(ValueError, match="missing tile id"):
        encode_future_wall_tokens([None, 0, 4, 8, 12])


def test_critic_features_append_future_wall_after_opponent_hands() -> None:
    observations = {
        0: observation(0, [0, 1, 2]),
        1: observation(1, [16, 17]),
        2: observation(2, [52]),
        3: observation(3, [108]),
    }
    table = collect_visible_table_state(observations)
    features = encode_critic_features(
        table,
        observer=0,
        future_wall_tiles=[134, 41, 119, 67, 90],
    )

    assert features.length == 4 + 5
    segments = features.factors[:, 0].tolist()
    assert segments[:4] == [SEGMENT_CRITIC_PRIVATE] * 4
    assert segments[4:] == [SEGMENT_CRITIC_FUTURE_WALL] * 5
    future = features.factors[4:]
    assert np.all(future[:, 1] == TOKEN_KIND_FUTURE_WALL)
    assert np.all(future[:, 2] == FIELD_FUTURE_WALL)
    assert future[:, 3].tolist() == [1, 2, 3, 4, 5]
    assert future[:, 9].tolist() == [1] * 5
