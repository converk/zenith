import numpy as np

from riichi_ppo_v1.learner import transition_length_metrics
from riichi_ppo_v1.trajectory import Transition, finish_kyoku


def transition(value: float) -> Transition:
    return Transition(np.zeros(1, dtype=np.uint8), np.zeros((1, 4, 4), dtype=np.uint8), np.zeros((1, 8), dtype=np.uint8), np.zeros((12, 160), dtype=np.uint8), 1, np.ones(241, dtype=bool), 0, 0.0, value)


def test_gae_does_not_cross_kyoku_boundaries() -> None:
    first = [transition(0.4), transition(0.2)]
    second = [transition(-0.1)]
    finish_kyoku(first, 1.0, gamma=1.0, gae_lambda=1.0)
    finish_kyoku(second, -1.0, gamma=1.0, gae_lambda=1.0)
    assert first[-1].done and second[-1].done
    assert first[-1].reward == 1.0 and second[-1].reward == -1.0
    assert first[0].return_ > 0
    assert second[0].return_ < 0


def test_transition_length_metrics_report_v4_blocks_and_board_tokens() -> None:
    first = Transition(np.zeros(10, dtype=np.uint8), np.zeros((10, 4, 4), dtype=np.uint8), np.zeros((10, 8), dtype=np.uint8), np.zeros((12, 160), dtype=np.uint8), 10, np.ones(241, dtype=bool), 0, 0.0, 0.0)
    second = Transition(np.zeros(6, dtype=np.uint8), np.zeros((6, 4, 4), dtype=np.uint8), np.zeros((6, 8), dtype=np.uint8), np.zeros((12, 160), dtype=np.uint8), 6, np.ones(241, dtype=bool), 0, 0.0, 0.0)

    metrics = transition_length_metrics([first, second])

    assert metrics["update/buffer_transition_event_blocks_mean"] == 8.0
    assert metrics["update/buffer_transition_input_tokens_mean"] == 20.0
    assert metrics["update/buffer_transition_board_tokens_mean"] == 12.0
