"""V16 rollout 记录与小局内 Expected-SARSA(lambda) Top-3 Q-boosting。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class Transition:
    """一个 V16 决策点:Objective Facts + Compact Snapshot + 每动作 Query 对。

    历史段(Objective Facts)、快照段与 query 段在 rollout 时一次性编码并随决策
    一起保留;训练更新时按同段 padding 重组回 ``forward_v16`` 的输入。
    """

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


def transition_sequence_length(transition: Transition) -> int:
    """Actor 序列长度 = Objective Facts + Snapshot + 每动作 offense/defense 对。"""
    return (
        int(transition.history_length)
        + int(transition.snapshot_length)
        + 2 * int(transition.query_pair_counts)
    )


def finish_kyoku_qboost(
    transitions: list[Transition], gamma: float, qboost_lambda: float,
) -> list[Transition]:
    """标记小局终局并按 Expected-SARSA 回溯 Q 目标与优势。"""
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
