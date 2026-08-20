"""V16 Transition 与小局内 GAE value advantage 的结算测试。"""

from __future__ import annotations

import numpy as np

from riichi_ppo_v1.training.trajectory import (
    Transition,
    finish_kyoku_gae,
    transition_sequence_length,
)


def transition(
    value: float,
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
        value,
    )


def test_gae_does_not_cross_kyoku_boundaries() -> None:
    first = [transition(0.2), transition(0.4)]
    second = [transition(-0.1)]
    first[-1].reward = 1.0
    second[-1].reward = -1.0
    finish_kyoku_gae(first, gamma=1.0, gae_lambda=1.0)
    finish_kyoku_gae(second, gamma=1.0, gae_lambda=1.0)
    assert first[-1].done and second[-1].done
    assert first[-1].reward == 1.0 and second[-1].reward == -1.0
    assert first[0].advantage > 0
    assert second[0].advantage < 0


def test_terminal_kyoku_score_reaches_each_prior_learner_decision() -> None:
    kyoku = [transition(0.0), transition(0.0), transition(0.0)]
    kyoku[-1].reward = 1.0

    finish_kyoku_gae(kyoku, gamma=0.995, gae_lambda=0.97)

    decay = 0.995 * 0.97
    assert kyoku[-1].advantage == np.float32(1.0)  # r_T - V_T
    assert kyoku[-2].advantage == np.float32(-0.0 + decay * 1.0)
    assert kyoku[-3].advantage == np.float32(decay * kyoku[-2].advantage)


def test_gae_uses_value_bootstrap_and_lamdba() -> None:
    rows = [transition(0.5), transition(1.0), transition(2.0)]
    rows[0].reward = 0.1
    finish_kyoku_gae(rows, gamma=0.9, gae_lambda=0.5)
    # δ_2 = 0 + γ·0 − V_2 = -2.0;A_2 = δ_2。
    # δ_1 = 0 + γ·V_2 − V_1 = 0.8;A_1 = δ_1 + γλ·A_2 = -0.1。
    # δ_0 = 0.1 + γ·V_1 − V_0 = 0.5;A_0 = δ_0 + γλ·A_1 = 0.455。
    assert rows[2].advantage == np.float32(-2.0)
    assert rows[1].advantage == np.float32(-0.1)
    assert rows[0].advantage == np.float32(0.5 + 0.9 * 0.5 * -0.1)


def test_sequence_length_counts_history_snapshot_and_query_pairs() -> None:
    item = transition(0.0, history_length=7, snapshot_length=3, pair_counts=5)
    assert transition_sequence_length(item) == 7 + 3 + 2 * 5
