from riichi_ppo_v1.training.opponents.history import rollout_cohort


def test_rollout_cohort_is_deterministic_bounded_and_without_replacement() -> None:
    pool = tuple(f"checkpoint-{index}" for index in range(8))
    first = rollout_cohort(pool, seed=7, update=41)
    assert first == rollout_cohort(pool, seed=7, update=41)
    assert len(first) == 2 and len(set(first)) == 2
    assert first != rollout_cohort(pool, seed=7, update=42)
