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
    discard_regret: float = 0.0
    call_regret: float = 0.0
    discard_weight: float = 0.25
    call_weight: float = 0.10
    kyoku_reward: float = 0.0
    done: bool = False
    advantage: float = 0.0
    return_: float = 0.0
    critic_factors: np.ndarray | None = None
    critic_length: int = 0
    teacher_mask: np.ndarray | None = None
    teacher_supervised: bool = False

    def refresh_reward(self) -> None:
        """Compose scale-controlled local regrets and terminal point reward."""
        self.reward = float(
            self.kyoku_reward
            + self.discard_weight * self.discard_regret
            + self.call_weight * self.call_regret
        )


def component_trace_statistics(
    transitions: list[Transition], gamma: float, gae_lambda: float,
) -> dict[str, float]:
    """Return reward-only GAE traces for driver-owned scale calibration."""
    rho = float(gamma) * float(gae_lambda)
    sums = {"kyoku": 0.0, "discard": 0.0, "call": 0.0}
    absolute = {"kyoku": 0.0, "discard": 0.0, "call": 0.0}
    tails = {"kyoku": 0.0, "discard": 0.0, "call": 0.0}
    for item in reversed(transitions):
        components = {
            "kyoku": item.kyoku_reward,
            "discard": item.discard_regret,
            "call": item.call_regret,
        }
        for name, reward in components.items():
            tails[name] = float(reward) + rho * tails[name]
            sums[name] += tails[name] ** 2
            absolute[name] += abs(tails[name])
    result = {"reward_scale/trace_count": float(len(transitions))}
    for name in sums:
        result[f"reward_scale/{name}_trace_sum_squares"] = sums[name]
        result[f"reward_scale/{name}_trace_abs_sum"] = absolute[name]
    return result


def finish_kyoku(transitions: list[Transition], gamma: float, gae_lambda: float) -> list[Transition]:
    """Mark one kyoku terminal and calculate GAE without cross-kyoku leakage."""
    if not transitions:
        return []
    transitions[-1].done = True
    gae = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        current = transitions[index]
        next_value = 0.0 if current.done else transitions[index + 1].value
        delta = current.reward + gamma * next_value - current.value
        gae = delta + gamma * gae_lambda * (0.0 if current.done else gae)
        current.advantage = float(np.float32(gae))
        current.return_ = float(np.float32(current.advantage + current.value))
    return transitions


def flatten(kyokus: Iterable[list[Transition]]) -> list[Transition]:
    return [transition for kyoku in kyokus for transition in kyoku]
