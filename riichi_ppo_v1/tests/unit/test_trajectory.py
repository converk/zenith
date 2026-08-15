import numpy as np
import pytest

from riichi_ppo_v1.training.rewards import terminal_hanchan_rank_rewards, terminal_kyoku_reward

from riichi_ppo_v1.training.learner import length_bucketed_minibatches, transition_length_metrics
from riichi_ppo_v1.training.trajectory import Transition, finish_kyoku_qboost


def transition(q_taken: float, expected_q: float | None = None) -> Transition:
    return Transition(
        np.zeros((1, 10), dtype=np.uint8), np.zeros((1, 8), dtype=np.float32),
        1, np.ones(241, dtype=bool), 0, 0.0, q_taken,
        expected_q=q_taken if expected_q is None else expected_q,
    )


def test_terminal_kyoku_reward_uses_configured_symmetric_clip() -> None:
    assert terminal_kyoku_reward(12_000, 32_000) == 12.0
    assert terminal_kyoku_reward(30_000, 32_000) == 30.0
    assert terminal_kyoku_reward(-30_000, 32_000) == -30.0
    assert terminal_kyoku_reward(40_000, 32_000) == 32.0
    with pytest.raises(ValueError, match="must be positive"):
        terminal_kyoku_reward(1_000, 0)


def test_qboost_does_not_cross_kyoku_boundaries() -> None:
    first = [transition(0.4), transition(0.2)]
    second = [transition(-0.1)]
    first[-1].reward = 1.0
    second[-1].reward = -1.0
    finish_kyoku_qboost(first, gamma=1.0, qboost_lambda=1.0)
    finish_kyoku_qboost(second, gamma=1.0, qboost_lambda=1.0)
    assert first[-1].done and second[-1].done
    assert first[-1].reward == 1.0 and second[-1].reward == -1.0
    assert first[0].q_target > 0
    assert second[0].q_target < 0


def test_terminal_kyoku_score_reaches_each_prior_learner_decision() -> None:
    kyoku = [transition(0.0), transition(0.0), transition(0.0)]
    kyoku[-1].reward = 1.0

    finish_kyoku_qboost(kyoku, gamma=0.995, qboost_lambda=0.97)

    decay = 0.995 * 0.97
    assert kyoku[-1].advantage == 1.0
    assert kyoku[-2].advantage == np.float32(decay)
    assert kyoku[-3].advantage == np.float32(decay ** 2)


def test_qboost_uses_expected_next_q_and_current_policy_baseline() -> None:
    rows = [transition(0.4, 0.25), transition(0.2, 0.1)]
    rows[-1].reward = 1.0
    finish_kyoku_qboost(rows, gamma=0.9, qboost_lambda=0.5)
    terminal_trace = 1.0 - 0.2
    first_trace = (0.9 * 0.1 - 0.4) + 0.9 * 0.5 * terminal_trace
    assert rows[-1].q_target == np.float32(1.0)
    assert rows[-1].advantage == np.float32(0.9)
    assert rows[0].q_target == np.float32(0.4 + first_trace)
    assert rows[0].advantage == np.float32(rows[0].q_target - 0.25)


def test_hanchan_rank_reward_is_zero_sum_and_breaks_ties_by_seat() -> None:
    assert terminal_hanchan_rank_rewards([40_000, 30_000, 20_000, 10_000]) == (
        16.0, 8.0, -8.0, -16.0,
    )
    tied = terminal_hanchan_rank_rewards([25_000, 25_000, 25_000, 25_000])
    assert tied == (16.0, 8.0, -8.0, -16.0)
    assert sum(tied) == 0.0


def test_transition_length_metrics_report_tokens_and_query() -> None:
    first = Transition(np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 8), dtype=np.float32), 10, np.ones(241, dtype=bool), 0, 0.0, 0.0)
    second = Transition(np.zeros((6, 10), dtype=np.uint8), np.zeros((6, 8), dtype=np.float32), 6, np.ones(241, dtype=bool), 0, 0.0, 0.0)

    metrics = transition_length_metrics([first, second])

    assert metrics["update/buffer_transition_tokens_mean"] == 8.0
    assert metrics["update/buffer_transition_input_tokens_mean"] == 9.0
    assert metrics["update/buffer_transition_input_tokens_max"] == 11.0
    assert metrics["update/buffer_effective_input_tokens"] == 18.0
    assert metrics["update/buffer_global_padded_input_tokens"] == 22.0
    assert metrics["update/buffer_global_padding_input_tokens"] == 4.0
    assert metrics["update/buffer_global_padding_fraction_of_padded_input_tokens"] == 4.0 / 22.0


def test_length_bucketing_covers_all_rows_with_homogeneous_batches() -> None:
    transitions = [
        Transition(np.zeros((length, 10), dtype=np.uint8), np.zeros((length, 8), dtype=np.float32),
                   length, np.ones(241, dtype=bool), 0, 0.0, 0.0)
        for length in (1, 1, 1, 1, 9, 9, 9, 9)
    ]
    np.random.seed(7)
    batches = length_bucketed_minibatches(transitions, minibatch_size=2)

    assert sorted(index for batch in batches for index in batch) == list(range(len(transitions)))
    assert all(len(batch) <= 2 for batch in batches)
    assert all(len({transitions[int(index)].token_length for index in batch}) == 1 for batch in batches)
