"""V18 rollout 记录与小局内 value-based GAE advantage 结算。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Transition:
    """一个 V18 决策点:完整 Actor 当前局面序列 + 每动作 Query 元数据。

    Actor 输入在 rollout 时由 ``BatchedStateBridge.prepare`` 一次性编码并随
    决策保留;训练更新时只按完整序列 padding,不再恢复旧 history/snapshot split。
    """

    actor_factors: np.ndarray
    actor_numeric: np.ndarray
    actor_length: int
    query_action_ids: np.ndarray
    query_pair_counts: int
    legal_mask: np.ndarray
    action: int
    logprob: float
    value: float
    reward: float = 0.0
    kyoku_reward: float = 0.0
    done: bool = False
    advantage: float = 0.0
    critic_factors: np.ndarray | None = None
    critic_length: int = 0


def transition_sequence_length(transition: Transition) -> int:
    """Actor 序列长度 = V18 当前局面完整 token 数。"""
    return int(transition.actor_length)


def finish_kyoku_gae(
    transitions: list[Transition], gamma: float, gae_lambda: float,
) -> list[Transition]:
    """标记小局终局并按 GAE(γ, λ) 计算 value-based advantage。

    以 critic 值预测为基线:δ_t = r_t + γV_{t+1} − V_t,小局终局的
    V_{t+1}=0;A_t = δ_t + γλ·A_{t+1},小局之间互不跨越。
    """
    if not transitions:
        return []
    transitions[-1].done = True
    gae = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        current = transitions[index]
        next_value = 0.0 if current.done else transitions[index + 1].value
        delta = current.reward + float(gamma) * float(next_value) - current.value
        gae = delta + float(gamma) * float(gae_lambda) * (0.0 if current.done else gae)
        current.advantage = float(np.float32(gae))
    return transitions


def flatten(kyokus: Iterable[list[Transition]]) -> list[Transition]:
    return [transition for kyoku in kyokus for transition in kyoku]
