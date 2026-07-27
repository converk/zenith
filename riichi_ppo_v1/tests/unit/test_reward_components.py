import numpy as np
import pytest

from riichi_ppo_v1.training.rewards.efficiency import efficiency_reward, remaining_ukeire
from riichi_ppo_v1.training.trajectory import Transition, finish_kyoku


def test_efficiency_reward_prioritizes_shanten_then_ukeire() -> None:
    assert efficiency_reward(3, 80, 2, 100) == -1.0
    assert efficiency_reward(2, 80, 2, 100) == -0.05
    assert efficiency_reward(2, 100, 2, 100) == 0.0
    assert efficiency_reward(5, 0, 2, 100) == -1.0
    remaining = np.zeros(34, dtype=np.int16); remaining[1] = 3; remaining[3] = 2
    assert remaining_ukeire((1 << 1) | (1 << 3), remaining) == 5


def test_three_component_reward_keeps_kyoku_score_primary() -> None:
    transition = Transition(
        np.zeros((1, 10), np.uint8), np.zeros((1, 8), np.float32), 1,
        np.ones(241, np.bool_), 0, 0.0, 0.0,
    )
    transition.discard_regret = -1.0
    transition.discard_weight = 0.10
    transition.call_regret = -0.5
    transition.call_weight = 0.10
    transition.kyoku_reward = 2.0
    transition.refresh_reward()

    assert transition.reward == pytest.approx(1.85)
    finish_kyoku([transition], gamma=.995, gae_lambda=.97)
    assert transition.reward == pytest.approx(1.85)


def test_weighted_efficiency_penalty_is_bounded() -> None:
    assert 0.10 * efficiency_reward(5, 0, 2, 100) == pytest.approx(-0.10)
