"""Rollout records and kyoku-local GAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Transition:
    token_factors: np.ndarray
    token_numeric: np.ndarray
    token_length: int
    legal_mask: np.ndarray
    action: int
    logprob: float
    value: float
    reward: float = 0.0
    efficiency_reward: float = 0.0
    kyoku_reward: float = 0.0
    rank_reward: float = 0.0
    reward_weights: tuple[float, float, float] = (0.0, 1.0, 0.0)
    done: bool = False
    advantage: float = 0.0
    return_: float = 0.0
    critic_factors: np.ndarray | None = None
    critic_length: int = 0

    def refresh_reward(self) -> None:
        self.reward = float(sum(weight * value for weight, value in zip(
            self.reward_weights,
            (self.efficiency_reward, self.kyoku_reward, self.rank_reward),
            strict=True,
        )))


def finish_kyoku(
    transitions: list[Transition], reward: float, gamma: float, gae_lambda: float
) -> list[Transition]:
    """Assign a terminal kyoku reward and calculate GAE without cross-kyoku leakage."""
    if not transitions:
        return []
    # Efficiency shaping is attached to individual decisions before a kyoku
    # ends.  Terminal score/rank terms must therefore be added, not overwrite
    # the tail's existing shaped reward.
    transitions[-1].reward += float(reward)
    transitions[-1].done = True
    gae = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        current = transitions[index]
        next_value = 0.0 if current.done else transitions[index + 1].value
        delta = current.reward + gamma * next_value - current.value
        gae = delta + gamma * gae_lambda * (0.0 if current.done else gae)
        # Match exp/training's target buffers: retain each recursive GAE value
        # as FP32 before it becomes the next step's trace or a PPO minibatch.
        current.advantage = float(np.float32(gae))
        current.return_ = float(np.float32(current.advantage + current.value))
    return transitions


def flatten(kyokus: Iterable[list[Transition]]) -> list[Transition]:
    return [transition for kyoku in kyokus for transition in kyoku]
