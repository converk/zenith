"""Rollout records and kyoku-local GAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Transition:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    length: int
    legal_mask: np.ndarray
    action: int
    logprob: float
    value: float
    reward: float = 0.0
    done: bool = False
    advantage: float = 0.0
    return_: float = 0.0
    # Confirmed event-history prefix length.  The remaining input tokens are
    # the per-decision snapshot; the model adds its learned decision token.
    history_length: int = 0


def finish_kyoku(
    transitions: list[Transition], reward: float, gamma: float, gae_lambda: float
) -> list[Transition]:
    """Assign a terminal kyoku reward and calculate GAE without cross-kyoku leakage."""
    if not transitions:
        return []
    transitions[-1].reward = float(reward)
    transitions[-1].done = True
    gae = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        current = transitions[index]
        next_value = 0.0 if current.done else transitions[index + 1].value
        delta = current.reward + gamma * next_value - current.value
        gae = delta + gamma * gae_lambda * (0.0 if current.done else gae)
        current.advantage = float(gae)
        current.return_ = float(gae + current.value)
    return transitions


def flatten(kyokus: Iterable[list[Transition]]) -> list[Transition]:
    return [transition for kyoku in kyokus for transition in kyoku]
