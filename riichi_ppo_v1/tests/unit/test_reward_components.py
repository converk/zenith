import numpy as np
import pytest

from riichi_ppo_v1.training.rewards.efficiency import efficiency_reward, remaining_ukeire
from riichi_ppo_v1.training.trajectory import Transition, finish_kyoku
from riichi_ppo_v1.training.worker import rank_reward


def test_efficiency_reward_prioritizes_shanten_then_ukeire() -> None:
    assert efficiency_reward(3, 80, 2, 100) == -3.0
    assert efficiency_reward(2, 80, 2, 100) == -0.4
    assert efficiency_reward(2, 100, 2, 100) == 0.0
    assert efficiency_reward(5, 0, 2, 100) == -6.0
    remaining = np.zeros(34, dtype=np.int16); remaining[1] = 3; remaining[3] = 2
    assert remaining_ukeire((1 << 1) | (1 << 3), remaining) == 5


def test_rank_is_only_added_to_last_kyoku_tail() -> None:
    transition = Transition(np.zeros((1, 10), np.uint8), np.zeros((1, 8), np.float32), 1,
                            np.ones(241, np.bool_), 0, 0.0, 0.0)
    transition.kyoku_reward = 8.0
    transition.rank_reward = rank_reward(1)
    transition.reward_weights = (0.0, 0.4, 0.6)
    transition.refresh_reward()
    assert transition.reward == pytest.approx(10.4)
    completed = finish_kyoku([transition], 0.0, .99, .95)
    assert completed[-1].reward == pytest.approx(10.4)
