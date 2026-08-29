"""BatchedRiichiEnv.walls(): live+dead wall snapshot contract."""

from __future__ import annotations

from riichienv import Action, ActionType, BatchedRiichiEnv


def _discard_action(observation) -> Action:
    for action in observation.legal_actions():
        if action.action_type == ActionType.DISCARD:
            return action
    raise AssertionError("expected a discard action in the dealer's initial window")


def test_walls_returns_one_row_per_env_with_live_and_dead_tiles() -> None:
    env = BatchedRiichiEnv(2, seed=42)
    observations = list(env.reset())
    walls = env.walls()

    assert len(walls) == 2
    for env_index in range(2):
        assert len(walls[env_index]) == 83  # 69 live + 14 dead
        assert len(walls[env_index][:5]) == 5
        tiles_left = min(
            observations[env_index][seat].tiles_left
            for seat in range(4)
        )
        assert len(walls[env_index]) == tiles_left + 14
        assert all(0 <= tile < 136 for tile in walls[env_index])


def test_walls_head_advances_after_a_step() -> None:
    env = BatchedRiichiEnv(1, seed=42)
    observations = list(env.reset())
    before = env.walls()[0]
    dealer_observation = observations[0][0]
    action = _discard_action(dealer_observation)

    observations = list(env.step_batch([
        {dealer_observation.player_id: action},
    ]))
    after = env.walls()[0]

    # The dealer's discard is followed by the next player's draw, so the wall
    # head moves exactly one tile forward and the wall loses one tile.
    assert len(after) == len(before) - 1
    assert after == before[1:]
    assert after[:5] == before[1:6]
    tiles_left = min(
        observations[0][seat].tiles_left
        for seat in range(4)
    )
    assert len(after) == tiles_left + 14


def test_walls_refresh_after_reset_indices() -> None:
    env = BatchedRiichiEnv(2, seed=42)
    observations = list(env.reset())
    action = _discard_action(observations[0][0])
    list(env.step_batch([
        {0: action},
        {},
    ]))
    assert len(env.walls()[0]) == 82
    assert len(env.walls()[1]) == 83  # untouched table keeps its full wall

    list(env.reset_indices([0]))
    after = env.walls()

    assert len(after[0]) == 83
    assert len(after[1]) == 83  # untouched table is unchanged
