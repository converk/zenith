import numpy as np

from riichi_ppo_v1.learner import transition_length_metrics
from riichi_ppo_v1.trajectory import Transition, finish_kyoku


def transition(value: float) -> Transition:
    return Transition(np.zeros((1, 8), dtype=np.int64), np.ones(1, dtype=bool), 1, np.ones(241, dtype=bool), 0, 0.0, value)


def test_gae_does_not_cross_kyoku_boundaries() -> None:
    first = [transition(0.4), transition(0.2)]
    second = [transition(-0.1)]
    finish_kyoku(first, 1.0, gamma=1.0, gae_lambda=1.0)
    finish_kyoku(second, -1.0, gamma=1.0, gae_lambda=1.0)
    assert first[-1].done and second[-1].done
    assert first[-1].reward == 1.0 and second[-1].reward == -1.0
    assert first[0].return_ > 0
    assert second[0].return_ < 0


def test_transition_length_metrics_separate_history_and_snapshot() -> None:
    first = Transition(np.zeros((10, 8), dtype=np.int64), np.ones(10, dtype=bool), 10, np.ones(241, dtype=bool), 0, 0.0, 0.0, history_length=7)
    second = Transition(np.zeros((6, 8), dtype=np.int64), np.ones(6, dtype=bool), 6, np.ones(241, dtype=bool), 0, 0.0, 0.0, history_length=2)

    metrics = transition_length_metrics([first, second])

    assert metrics["update/buffer_transition_sequence_tokens_mean"] == 8.0
    assert metrics["update/buffer_transition_history_tokens_mean"] == 4.5
    assert metrics["update/buffer_transition_snapshot_tokens_mean"] == 3.5
