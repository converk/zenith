"""Gradient accumulation 语义测试:多步 backward 累积后单次 step。"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch

from riichi_ppo_v1.training.learner import PPOLearner


@dataclass
class _Item:
    history_factors: np.ndarray
    history_numeric: np.ndarray
    history_length: int
    snapshot_kinds: np.ndarray
    snapshot_cat: np.ndarray
    snapshot_num: np.ndarray
    snapshot_length: int
    query_rows: np.ndarray
    query_action_ids: np.ndarray
    query_pair_counts: int
    legal_mask: np.ndarray
    action: int
    logprob: float
    q_taken: float
    value: float
    expected_q: float = 0.0
    reward: float = 0.0
    kyoku_reward: float = 0.0
    done: bool = False
    advantage: float = 0.0
    q_target: float = 0.0
    critic_factors: np.ndarray | None = None
    critic_length: int = 0
    top3_ids: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int32))


def _transition(reward: float, action: int = 0) -> _Item:
    legal = np.zeros(241, dtype=np.bool_)
    legal[action] = True
    return _Item(
        history_factors=np.zeros((2, 10), dtype=np.uint8),
        history_numeric=np.zeros((2, 8), dtype=np.float32),
        history_length=2,
        snapshot_kinds=np.zeros(2, dtype=np.uint8),
        snapshot_cat=np.zeros((2, 4), dtype=np.uint8),
        snapshot_num=np.zeros((2, 7), dtype=np.float32),
        snapshot_length=2,
        query_rows=np.zeros((4, 15), dtype=np.int32),
        query_action_ids=np.zeros(2, dtype=np.int32),
        query_pair_counts=2,
        legal_mask=legal,
        action=action,
        logprob=np.log(1.0),
        q_taken=0.0,
        value=0.0,
        expected_q=0.0,
        reward=float(reward),
        advantage=float(reward),
    )


def learner_kwargs(**overrides) -> dict:
    base = dict(
        learning_rate=2e-4,
        actor_learning_rate=2e-4,
        shared_learning_rate=2e-4,
        critic_learning_rate=2e-4,
        update_epochs=2,
        minibatch_size=2,
        target_kl=0.0,
        ppo_clip=0.2,
        max_grad_norm=5.0,
        warmup_fraction=0.0,
        value_coef=0.5,
        q_coef=1.0,
        q_boost_coef=0.0,
        q_boost_lambda=1.0,
        q_temperature=1.0,
        entropy_start=0.0,
        entropy_end=0.0,
        sft_kl_coef_start=0.0,
        critic_bootstrap_learning_rate=2e-4,
        critic_bootstrap_updates=0,
        value_target_normalization="none",
        value_target_std_floor=0.01,
        gradient_accumulation_steps=1,
    )
    base.update(overrides)
    return base


def test_gradient_accumulation_steps_affect_step_count() -> None:
    """4 minibatches(2 epochs × 2 batch)在 accumulation=2 下只 step 2 次。"""
    learner = PPOLearner(
        "v16", "cpu",
        **learner_kwargs(gradient_accumulation_steps=2),
    )
    transitions = [
        _transition(0.2, action=0), _transition(-0.1, action=1),
        _transition(0.3, action=2),
    ]
    metrics = learner.update(transitions, shuffle_seed=7)
    # 4 个 minibatch(3 样本/2 → 2 每 epoch × 2 epochs);accumulation=2。
    assert metrics["update/executed_minibatches"] == 4.0


def test_gradient_accumulation_single_step_does_not_break() -> None:
    learner = PPOLearner(
        "v16", "cpu",
        **learner_kwargs(gradient_accumulation_steps=1),
    )
    transitions = [
        _transition(0.2, action=0), _transition(-0.1, action=1),
        _transition(0.3, action=2),
    ]
    metrics = learner.update(transitions, shuffle_seed=7)
    assert metrics["update/executed_minibatches"] == 4.0
    assert np.isfinite(metrics["loss"])