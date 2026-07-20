from types import SimpleNamespace

import numpy as np

from riichi_ppo_v1.model.critic_features import (
    FIELD_OPPONENT_MELD_TILE,
    FIELD_OPPONENT_RIVER,
    FIELD_OPPONENT_HAND,
    collect_visible_table_state,
    encode_critic_features,
)


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


def test_compact_opponent_rivers_and_melds_are_critic_only_opt_in() -> None:
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
    baseline = encode_critic_features(table, observer=0)
    features = encode_critic_features(table, observer=0, include_public_state=True)

    assert features.length == baseline.length + 5
    river_rows = features.factors[features.factors[:, 2] == FIELD_OPPONENT_RIVER]
    assert {tuple(row) for row in river_rows.tolist()} == {
        (4, 4, FIELD_OPPONENT_RIVER, 2, 1, 5, 0, 1, 0, 1),
        (4, 4, FIELD_OPPONENT_RIVER, 2, 1, 5, 1, 1, 0, 1),
        (4, 4, FIELD_OPPONENT_RIVER, 2, 1, 6, 0, 1, 0, 1),
    }
    meld_tile_rows = features.factors[features.factors[:, 2] == FIELD_OPPONENT_MELD_TILE]
    assert meld_tile_rows.tolist() == [[4, 4, FIELD_OPPONENT_MELD_TILE, 2, 1, 1, 0, 3, 1, 1]]
    assert any(row.tolist() == [4, 5, 2, 2, 1, 1, 0, 3, 1, 1] for row in features.factors)
