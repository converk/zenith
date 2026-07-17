import random

from riichi_ppo_v1.training.curriculum import Curriculum
from riichi_ppo_v1.training.opponents.lineup import CURRENT, DEFENSE, EFFICIENCY, LineupSampler


def test_bootstrap_records_all_seats_and_heuristic_stage_records_two() -> None:
    sampler = LineupSampler(random.Random(3))
    curriculum = Curriculum(1000)
    bootstrap = sampler.sample(curriculum.snapshot(0), (), 0, 0)
    assert bootstrap.policies == (CURRENT,) * 4 and bootstrap.learner_mask == 0b1111
    heuristic = sampler.sample(curriculum.snapshot(200), (), 0, 0)
    assert heuristic.policies.count(CURRENT) == 2
    assert set(heuristic.policies) == {CURRENT, EFFICIENCY, DEFENSE}


def test_rank_lineup_uses_two_distinct_histories_when_available() -> None:
    sampler = LineupSampler(random.Random(4))
    lineup = sampler.sample(Curriculum(1000).snapshot(900), ("a", "b"), 0, 0)
    opponents = [policy for policy in lineup.policies if policy != CURRENT]
    assert len(opponents) == 2
