"""Rollout records and kyoku-local Expected-SARSA(lambda) Q-boosting."""

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
    q_taken: float
    expected_q: float = 0.0
    reward: float = 0.0
    kyoku_reward: float = 0.0
    hanchan_rank_reward: float = 0.0
    done: bool = False
    advantage: float = 0.0
    q_target: float = 0.0
    critic_factors: np.ndarray | None = None
    critic_length: int = 0


def finish_kyoku_qboost(
    transitions: list[Transition], gamma: float, qboost_lambda: float,
) -> list[Transition]:
    """Mark one kyoku terminal and calculate a seat-local Expected-SARSA trace."""
    if not transitions:
        return []
    transitions[-1].done = True
    trace = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        current = transitions[index]
        next_expected_q = 0.0 if current.done else transitions[index + 1].expected_q
        delta = current.reward + gamma * next_expected_q - current.q_taken
        trace = delta + gamma * qboost_lambda * (0.0 if current.done else trace)
        current.q_target = float(np.float32(current.q_taken + trace))
        current.advantage = float(np.float32(current.q_target - current.expected_q))
    return transitions


def flatten(kyokus: Iterable[list[Transition]]) -> list[Transition]:
    return [transition for kyoku in kyokus for transition in kyoku]
