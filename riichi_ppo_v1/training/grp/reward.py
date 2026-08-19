"""V17 PPO 奖励:纯 GRP reward(Mortal/Suphx 每小局 credit assignment)。

排名 utility 为 [1, 1/3, -1/3, -1];每小局 reward = 下一小局开始时 expected
rank utility - 本小局开始时 expected rank utility;最后一局 = 真实最终排名
utility - 最后一局开始时 GRP expected utility。不再包含任何小局点差分量,
也不做 σ 归一化(δ 幅度天然在 ±2 以内)。
"""

from __future__ import annotations

import torch
from torch import Tensor

from ...model.grp import GRP_UTILITY

RANK_UTILITY = tuple(float(value) for value in GRP_UTILITY)


def rank_utility(rank: int) -> float:
    """排名 utility(rank 0..3 → 1/1/3/-1/3/-1)。"""
    if not 0 <= int(rank) < 4:
        raise ValueError("rank must be 0..3")
    return RANK_UTILITY[int(rank)]


def grp_expected_value(rank_logits: Tensor) -> Tensor:
    """把 4 类排名概率 logits 转为期望 utility:Σ P(rank)·U(rank)。

    24 类全排列 logits 需先经 ``GRPModel.calc_matrix`` 聚合为 [..., 4, 4]
    玩家排名概率后,再逐玩家取期望(见 ``grp_expected_values_from_matrix``)。
    """
    probabilities = torch.softmax(rank_logits.float(), dim=-1)
    utility = rank_logits.new_tensor(RANK_UTILITY)
    return probabilities @ utility


def grp_expected_values_from_matrix(matrix: Tensor) -> Tensor:
    """把 ``calc_matrix`` 的 [..., 4, 4] 玩家排名概率转为每玩家期望 utility。"""
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"GRP matrix must end with [4, 4], got {tuple(matrix.shape)}")
    utility = matrix.new_tensor(RANK_UTILITY)
    return matrix @ utility


def grp_delta(values: Tensor) -> Tensor:
    """小局边界 delta:R^GRP_k = V_{k+1} - V_k(每小局 credit assignment)。"""
    if values.shape[-1] < 2:
        return values.new_zeros(())
    return values[..., 1:] - values[..., :-1]
