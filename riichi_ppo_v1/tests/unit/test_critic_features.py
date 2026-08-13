from types import SimpleNamespace

import numpy as np

from riichi_ppo_v1.model.critic_features import (
    FIELD_FUTURE_WALL,
    FIELD_PUBLIC_MELD_TILE,
    FIELD_PUBLIC_RIVER,
    FIELD_OPPONENT_HAND,
    SEGMENT_CRITIC_FUTURE_WALL,
    SEGMENT_CRITIC_PRIVATE,
    SEGMENT_PUBLIC_SUMMARY,
    TOKEN_KIND_FUTURE_WALL,
    collect_visible_table_state,
    encode_future_wall_tokens,
    encode_critic_features,
    encode_public_summary,
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


def test_compact_four_seat_rivers_and_melds_are_actor_visible_only() -> None:
    pon = SimpleNamespace(
        meld_type="pon",
        tiles=[0, 1, 2],
        called_tile=0,
        from_who=2,
    )
    observations = {
        0: SimpleNamespace(
            hands=[[0], [], [], []],
            discards=[[], [16, 17, 20], [], []],
            melds=[[], [pon], [], []],
        ),
        1: observation(1, [4]),
        2: observation(2, [8]),
        3: observation(3, [12]),
    }

    table = collect_visible_table_state(observations, include_public_state=True)
    critic = encode_critic_features(table, observer=0)
    features = encode_public_summary(table, observer=0)

    assert critic.length == 3
    assert np.all(critic.factors[:, 2] == FIELD_OPPONENT_HAND)
    assert features.length == 5
    assert np.all(features.factors[:, 0] == SEGMENT_PUBLIC_SUMMARY)
    river_rows = features.factors[features.factors[:, 2] == FIELD_PUBLIC_RIVER]
    assert {tuple(row) for row in river_rows.tolist()} == {
        (3, 4, FIELD_PUBLIC_RIVER, 2, 1, 5, 0, 1, 0, 1),
        (3, 4, FIELD_PUBLIC_RIVER, 2, 1, 5, 1, 1, 0, 1),
        (3, 4, FIELD_PUBLIC_RIVER, 2, 1, 6, 0, 1, 0, 1),
    }
    meld_tile_rows = features.factors[features.factors[:, 2] == FIELD_PUBLIC_MELD_TILE]
    assert meld_tile_rows.tolist() == [[3, 4, FIELD_PUBLIC_MELD_TILE, 2, 1, 1, 0, 3, 1, 1]]
    assert any(row.tolist() == [3, 5, 2, 2, 1, 1, 0, 3, 1, 1] for row in features.factors)


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


def test_future_wall_skips_missing_tiles_without_losing_order() -> None:
    rows = encode_future_wall_tokens([16, 52])
    assert len(rows) == 2
    assert [row[3] for row in rows] == [1, 2]
    assert [row[4:6] for row in rows] == [(1, 5), (2, 5)]


def test_future_wall_rejects_invalid_tile_ids() -> None:
    with pytest.raises(ValueError, match="invalid RiichiEnv tile id"):
        encode_future_wall_tokens([136])
    with pytest.raises(ValueError, match="invalid RiichiEnv tile id"):
        encode_future_wall_tokens([-1])
    with pytest.raises(ValueError, match="missing tile id"):
        encode_future_wall_tokens([None])


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


def test_actor_public_summary_has_no_critic_only_segments() -> None:
    observations = {
        0: observation(0, [0]),
        1: observation(1, [16]),
        2: observation(2, [52]),
        3: observation(3, [108]),
    }
    table = collect_visible_table_state(observations, include_public_state=True)
    features = encode_public_summary(table, observer=0)

    assert np.all(np.isin(
        features.factors[:, 0],
        (SEGMENT_PUBLIC_SUMMARY,),
    ))
    assert not np.any(features.factors[:, 0] == SEGMENT_CRITIC_PRIVATE)
    assert not np.any(features.factors[:, 0] == SEGMENT_CRITIC_FUTURE_WALL)
