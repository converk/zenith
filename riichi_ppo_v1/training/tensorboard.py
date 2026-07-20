"""Curated real-time TensorBoard projection for Riichi PPO.

The JSONL outputs retain full diagnostics.  TensorBoard intentionally exposes
only the small set of signals useful for routine training decisions.
"""

from __future__ import annotations

import math
from typing import Mapping, Protocol


class ScalarWriter(Protocol):
    """Minimal SummaryWriter surface used by the projection."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...


CURATED_SCALAR_TAGS = frozenset({
    # PPO policy/value optimisation and policy-drift diagnostics.
    "ppo/policy_loss",
    "ppo/value_loss",
    "ppo/value_loss_raw",
    "ppo/value_target_std",
    "ppo/value_prediction_std",
    "ppo/entropy",
    "ppo/entropy_normalized",
    "ppo/approx_kl",
    "ppo/clipfrac",
    "ppo/ratio_p95",
    "ppo/explained_variance",
    "ppo/grad_norm",
    "ppo/grad_norm_post_clip",
    "ppo/grad_norm_actor",
    "ppo/grad_norm_critic",
    "ppo/grad_norm_shared",
    "ppo/system/learning_rate",
    "ppo/system/entropy_coef",
    "ppo/update/early_stop",
    # Kyoku-primary reward balance and early discard-bootstrap health.
    "reward_schedule/efficiency_weight",
    "reward_schedule/kyoku_weight",
    "reward_schedule/efficiency_abs_mean",
    "reward_schedule/kyoku_abs_mean",
    # Sampled self-play kyoku health: outcomes and game pace.
    "train/kyoku/draw_rate",
    "train/kyoku/deal_in_rate",
    "train/kyoku/discard_count_mean",
    "train/kyoku/open_melds_mean",
    "train/match/length_kyokus_mean",
    "train/action/riichi_rate",
    "train/action/call_opportunity_accept_rate",
    "train/efficiency/optimal_shanten_rate",
    "train/efficiency/optimal_ukeire_rate",
    # Deterministic baseline evaluation: results, game pace and tile efficiency.
    "eval/kyoku/point_delta_mean",
    "eval/kyoku/point_delta_mean_stderr",
    "eval/kyoku/win_rate",
    "eval/kyoku/deal_in_rate",
    "eval/kyoku/draw_rate",
    "eval/kyoku/discard_count_mean",
    "eval/match/first_place_rate",
    "eval/match/mean_rank",
    "eval/match/top2_rate",
    "eval/match/last_place_rate",
    "eval/match/length_kyokus_mean",
    "eval/match/discard_count_mean",
    "eval/efficiency/optimal_shanten_rate",
    "eval/efficiency/optimal_ukeire_rate",
    # Per-iteration runtime health.
    "iteration/sps",
    "iteration/algorithm_wall_s",
    "update/wall_s",
    "system/learner_gpu_peak_allocated_mb",
})

# Keep metric keys stable for JSONL and programmatic consumers, while making
# TensorBoard's sidebar immediately readable during training.
TENSORBOARD_DISPLAY_TAGS = {
    "ppo/policy_loss": "PPO/策略损失 (policy_loss)",
    "ppo/value_loss": "PPO/价值损失 (value_loss)",
    "ppo/value_loss_raw": "PPO/原始尺度价值损失 (value_loss_raw)",
    "ppo/value_target_std": "PPO/价值目标标准差 (value_target_std)",
    "ppo/value_prediction_std": "PPO/价值预测标准差 (value_prediction_std)",
    "ppo/entropy": "PPO/策略熵 (entropy)",
    "ppo/entropy_normalized": "PPO/归一化策略熵 (entropy_normalized)",
    "ppo/approx_kl": "PPO/近似 KL (approx_kl)",
    "ppo/clipfrac": "PPO/裁剪比例 (clipfrac)",
    "ppo/ratio_p95": "PPO/概率比 P95 (ratio_p95)",
    "ppo/explained_variance": "PPO/价值解释方差 (explained_variance)",
    "ppo/grad_norm": "PPO/梯度范数 (grad_norm)",
    "ppo/grad_norm_post_clip": "PPO/裁剪后梯度范数 (grad_norm_post_clip)",
    "ppo/grad_norm_actor": "PPO/Actor 分支梯度范数·裁剪前 (grad_norm_actor)",
    "ppo/grad_norm_critic": "PPO/Critic 分支梯度范数·裁剪前 (grad_norm_critic)",
    "ppo/grad_norm_shared": "PPO/共享分支梯度范数·裁剪前 (grad_norm_shared)",
    "ppo/system/learning_rate": "PPO/学习率 (learning_rate)",
    "ppo/system/entropy_coef": "PPO/熵系数 (entropy_coef)",
    "ppo/update/early_stop": "PPO/提前停止 (early_stop)",
    "reward_schedule/efficiency_weight": "奖励调度/效率奖励权重 (efficiency_weight)",
    "reward_schedule/kyoku_weight": "奖励调度/牌局奖励权重 (kyoku_weight)",
    "reward_schedule/efficiency_abs_mean": "奖励调度/效率奖励绝对均值 (efficiency_abs_mean)",
    "reward_schedule/kyoku_abs_mean": "奖励调度/牌局奖励绝对均值 (kyoku_abs_mean)",
    "train/kyoku/draw_rate": "采样牌局/流局率 (draw_rate)",
    "train/kyoku/deal_in_rate": "采样牌局/放铳率·四家汇总 (deal_in_rate)",
    "train/kyoku/discard_count_mean": "采样牌局/平均每局打牌次数 (discard_count)",
    "train/kyoku/open_melds_mean": "采样牌局/平均副露数·四家合计 (open_melds_mean)",
    "train/match/length_kyokus_mean": "采样对局/平均局数 (length_kyokus_mean)",
    "train/action/riichi_rate": "采样策略/立直率 (riichi_rate)",
    "train/action/call_opportunity_accept_rate": "采样策略/鸣牌机会接受率 (call_opportunity_accept_rate)",
    "train/efficiency/optimal_shanten_rate": "采样效率/最优向听率·四家汇总 (optimal_shanten_rate)",
    "train/efficiency/optimal_ukeire_rate": "采样效率/最优进张率·四家汇总 (optimal_ukeire_rate)",
    "eval/kyoku/point_delta_mean": "评测牌局/平均得分变化 (point_delta_mean)",
    "eval/kyoku/point_delta_mean_stderr": "评测牌局/平均得分变化标准误 (point_delta_mean_stderr)",
    "eval/kyoku/win_rate": "评测牌局/和牌率 (win_rate)",
    "eval/kyoku/deal_in_rate": "评测牌局/放铳率 (deal_in_rate)",
    "eval/kyoku/draw_rate": "评测牌局/流局率 (draw_rate)",
    "eval/kyoku/discard_count_mean": "评测牌局/平均每局打牌次数 (discard_count)",
    "eval/match/first_place_rate": "评测对局/一位率 (first_place_rate)",
    "eval/match/mean_rank": "评测对局/平均排名 (mean_rank)",
    "eval/match/top2_rate": "评测对局/前二率 (top2_rate)",
    "eval/match/last_place_rate": "评测对局/四位率 (last_place_rate)",
    "eval/match/length_kyokus_mean": "评测对局/平均局数 (length_kyokus_mean)",
    "eval/match/discard_count_mean": "评测对局/平均打牌次数 (discard_count)",
    "eval/efficiency/optimal_shanten_rate": "评测效率/最优向听率 (optimal_shanten_rate)",
    "eval/efficiency/optimal_ukeire_rate": "评测效率/最优进张率 (optimal_ukeire_rate)",
    "iteration/sps": "性能/每秒决策数 (sps)",
    "iteration/algorithm_wall_s": "性能/单迭代总耗时·秒 (algorithm_wall_s)",
    "update/wall_s": "性能/PPO 更新耗时·秒 (wall_s)",
    "system/learner_gpu_peak_allocated_mb": "系统/学习器 GPU 峰值显存·MB (learner_gpu_peak_allocated_mb)",
}


def curated_scalars(metrics: Mapping[str, float]) -> dict[str, float]:
    """Return finite values whose paths belong on the realtime dashboard."""
    return {
        name: float(value)
        for name, value in metrics.items()
        if name in CURATED_SCALAR_TAGS and math.isfinite(float(value))
    }


def write_curated_scalars(writer: ScalarWriter, metrics: Mapping[str, float], step: int) -> None:
    """Project selected metrics to TensorBoard at one training step."""
    for name, value in curated_scalars(metrics).items():
        writer.add_scalar(TENSORBOARD_DISPLAY_TAGS[name], value, int(step))


def learner_peak_allocated_mb(metrics_by_rank: list[Mapping[str, float]]) -> float | None:
    """Return the highest learner peak allocation reported by any DDP rank."""
    peaks = [
        float(metrics["gpu/torch_memory_peak_allocated_mb"])
        for metrics in metrics_by_rank
        if "gpu/torch_memory_peak_allocated_mb" in metrics
        and math.isfinite(float(metrics["gpu/torch_memory_peak_allocated_mb"]))
    ]
    return max(peaks) if peaks else None
