import numpy as np

from riichi_ppo_v1.learner import PPOLearner
from riichi_ppo_v1.trajectory import Transition


def transition(value: float) -> Transition:
    item = Transition(
        np.array([1], dtype=np.uint8),
        np.array([[[1, 1, 12, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=np.uint8),
        np.zeros((1, 8), dtype=np.uint8), np.zeros((12, 160), dtype=np.uint8), 1,
        np.eye(1, 241, 0, dtype=np.bool_)[0], 0, 0.0, value,
    )
    item.advantage = value
    item.return_ = value
    return item


def test_target_kl_zero_completes_all_ppo_epochs() -> None:
    learner = PPOLearner(
        "mid", "cpu", learning_rate=1e-4, profile_enabled=False, profile_cuda_sync=False,
        update_epochs=2, minibatch_size=2, ppo_clip=0.2, value_coef=0.5,
        entropy_coef=0.01, max_grad_norm=0.5, target_kl=0.0,
    )

    metrics = learner.update([transition(0.2), transition(-0.1), transition(0.3)])

    assert metrics["update/early_stop"] == 0.0
    assert metrics["update/configured_epochs"] == 2.0
    assert metrics["update/epochs_completed"] == 2.0
    assert metrics["update/planned_minibatches"] == 4.0
    assert metrics["update/executed_minibatches"] == 4.0
    assert metrics["update/executed_transition_samples"] == 6.0
