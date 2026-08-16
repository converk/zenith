"""V16 PPO 奖励:70% 归一化 GRP delta + 30% 归一化小局分差。

utility [24,8,-12,-20];σ_GRP/σ_Score 离线固化后训练期只读;终局(半庄结束)使用
真实最终排名的 utility,不再叠加独立半庄排名奖励分量。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ...model.grp import GRP_UTILITY

RANK_UTILITY = tuple(float(value) for value in GRP_UTILITY)
GRP_REWARD_WEIGHT = 0.7
SCORE_REWARD_WEIGHT = 0.3
# V16 奖励范围放大:utility ×2 后 GRP 分量翻倍,外层归一化 clip 放宽到 ±10,
# 内层小局分差截断放宽到 ±24 千点(σ_GRP/σ_Score 离线固化值不变)。
REWARD_CLIP = 10.0
SCORE_DELTA_CLIP = 24.0
SCORE_DELTA_SCALE = 1_000.0


def rank_utility(rank: int) -> float:
    """排名 utility(rank 0..3 → 24/8/-12/-20)。"""
    if not 0 <= int(rank) < 4:
        raise ValueError("rank must be 0..3")
    return RANK_UTILITY[int(rank)]


def grp_expected_value(rank_logits: Tensor) -> Tensor:
    """V_GRP = 12·P1 + 4·P2 - 6·P3 - 10·P4。"""
    probabilities = torch.softmax(rank_logits, dim=-1)
    utility = rank_logits.new_tensor(RANK_UTILITY)
    return probabilities @ utility


def grp_delta(rank_logits: Tensor) -> Tensor:
    """小局边界 delta:R^GRP_k = V_{k+1} - V_k。"""
    values = grp_expected_value(rank_logits)
    if values.shape[-1] < 2:
        return values.new_zeros(())
    return values[..., 1:] - values[..., :-1]


def normalized_score_reward(score_delta: float, sigma_score: float) -> float:
    """R̂_Score = clip(clip(Δscore/1000, ±24) / σ_Score, ±10)。"""
    if float(sigma_score) <= 0.0:
        raise ValueError("sigma_score must be positive")
    scaled = float(np.clip(float(score_delta) / SCORE_DELTA_SCALE, -SCORE_DELTA_CLIP, SCORE_DELTA_CLIP))
    return float(np.clip(scaled / float(sigma_score), -REWARD_CLIP, REWARD_CLIP))


def combined_reward(
    r_grp: float,
    score_delta: float,
    sigma_grp: float,
    sigma_score: float,
) -> float:
    """R = 0.7·clip(R_GRP/σ_GRP, ±10) + 0.3·clip(clip(Δscore/1000, ±24)/σ_Score, ±10)。"""
    if float(sigma_grp) <= 0.0:
        raise ValueError("sigma_grp must be positive")
    grp_hat = float(np.clip(float(r_grp) / float(sigma_grp), -REWARD_CLIP, REWARD_CLIP))
    score_hat = normalized_score_reward(score_delta, sigma_score)
    return GRP_REWARD_WEIGHT * grp_hat + SCORE_REWARD_WEIGHT * score_hat


def load_normalization(dataset: Path) -> tuple[float, float]:
    """从 GRP 数据集 JSON 读取离线固化的 σ_GRP/σ_Score(训练期只读)。"""
    path = dataset / "dataset.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    normalization = value["normalization"]
    sigma_grp = float(normalization["sigma_grp"])
    sigma_score = float(normalization["sigma_score"])
    if sigma_grp <= 0.0 or sigma_score <= 0.0:
        raise RuntimeError(f"GRP normalization stats are missing in {path}")
    return sigma_grp, sigma_score
