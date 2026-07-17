import numpy as np

from riichi_ppo_v1.training.learner import length_bucketed_minibatches, transition_length_metrics
from riichi_ppo_v1.training.trajectory import Transition, finish_kyoku


def transition(value: float) -> Transition:
    return Transition(np.zeros((1, 10), dtype=np.uint8), np.zeros((1, 8), dtype=np.float32), 1, np.ones(241, dtype=bool), 0, 0.0, value)


def test_gae_does_not_cross_kyoku_boundaries() -> None:
    first = [transition(0.4), transition(0.2)]
    second = [transition(-0.1)]
    finish_kyoku(first, 1.0, gamma=1.0, gae_lambda=1.0)
    finish_kyoku(second, -1.0, gamma=1.0, gae_lambda=1.0)
    assert first[-1].done and second[-1].done
    assert first[-1].reward == 1.0 and second[-1].reward == -1.0
    assert first[0].return_ > 0
    assert second[0].return_ < 0


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
