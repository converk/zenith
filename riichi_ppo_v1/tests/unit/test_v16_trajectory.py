"""V16 Transition 与 Expected-SARSA Q-boost 的小局结算测试。"""

from __future__ import annotations

import numpy as np

from riichi_ppo_v1.training.trajectory import (
    Transition,
    finish_kyoku_qboost,
    transition_sequence_length,
)


def transition(
    q_taken: float,
    expected_q: float | None = None,
    *,
    history_length: int = 1,
    snapshot_length: int = 1,
    pair_counts: int = 1,
) -> Transition:
    return Transition(
        np.zeros((history_length, 10), dtype=np.uint8),
        np.zeros((history_length, 8), dtype=np.float32),
        history_length,
        np.zeros(snapshot_length, dtype=np.uint8),
        np.zeros((snapshot_length, 4), dtype=np.uint8),
        np.zeros((snapshot_length, 7), dtype=np.float32),
        snapshot_length,
        np.zeros((2 * pair_counts, 15), dtype=np.int32),
        np.zeros(pair_counts, dtype=np.int32),
        pair_counts,
        np.ones(241, dtype=bool),
        0,
        0.0,
        q_taken,
        0.0,
        expected_q=q_taken if expected_q is None else expected_q,
    )


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
    assert kyoku[-3].advantage == np.float32(decay**2)


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


def test_sequence_length_counts_history_snapshot_and_query_pairs() -> None:
    item = transition(0.0, history_length=7, snapshot_length=3, pair_counts=5)
    assert transition_sequence_length(item) == 7 + 3 + 2 * 5
