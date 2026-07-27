from riichi_ppo_v1.training.curriculum import rollout_lineups


def test_history_curriculum_is_deterministic_and_collects_two_current_seats() -> None:
    assert rollout_lineups(
        2, update=1500, worker_id=3, history_ids=("a", "b"),
    ) == [("current",) * 4] * 2
    first = rollout_lineups(
        8, update=1501, worker_id=3, history_ids=("a", "b", "c"),
    )
    second = rollout_lineups(
        8, update=1501, worker_id=3, history_ids=("a", "b", "c"),
    )
    assert first == second
    assert all(lineup.count("current") == 2 for lineup in first)
    assert all(len({policy for policy in lineup if policy != "current"}) == 2 for lineup in first)
