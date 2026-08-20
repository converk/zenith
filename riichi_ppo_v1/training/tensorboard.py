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
    # Rollout volume (V17: complete-hanchan based stopping).
    "rollout/games",
    "rollout/kyokus",
    "rollout/grp_calls",
    "rollout/reward_mean",
    "ppo/advantage_mean",
    "ppo/advantage_std",
    "ppo/value_explained_variance",
    "ppo/buffer/advantage_mean",
    "ppo/buffer/advantage_std",
    "ppo/buffer/value_mean",
    "ppo/buffer/value_std",
    "ppo/offense_projection_weight_norm",
    "ppo/offense_projection_grad_norm",
    "ppo/entropy",
    "ppo/entropy_normalized",
    "ppo/approx_kl",
    "ppo/sft_reference_kl",
    "ppo/clipfrac",
    "ppo/ratio_p95",
    "ppo/grad_norm",
    "ppo/grad_norm_post_clip",
    "ppo/grad_norm_actor",
    "ppo/grad_norm_critic",
    "ppo/grad_norm_shared",
    "ppo/system/learning_rate",
    "ppo/system/actor_learning_rate",
    "ppo/system/shared_learning_rate",
    "ppo/system/critic_learning_rate",
    "ppo/system/critic_public_grad_scale",
    "ppo/system/entropy_coef",
    "ppo/system/sft_kl_coef",
    "ppo/training/critic_bootstrap",
    "ppo/update/early_stop",
    "reward_schedule/kyoku_weight",
    "train/reward/hanchan_rank_mean",
    "train/reward/hanchan_rank_std",
    "train/reward/kyoku_mean",
    "train/reward/kyoku_std",
    "train/reward/total_mean",
    "train/reward/total_std",
    # Sampled self-play kyoku health: outcomes and game pace.
    "train/kyoku/draw_rate",
    "train/kyoku/deal_in_rate",
    "train/kyoku/discard_count_mean",
    "train/kyoku/open_melds_mean",
    "train/match/length_kyokus_mean",
    "train/action/riichi_opportunity_accept_rate",
    "train/action/call_opportunity_accept_rate",
    # Periodic fixed-baseline evaluation.
    "eval/match/first_place_rate",
    "eval/match/mean_rank",
    "eval/match/top2_rate",
    "eval/match/last_place_rate",
    "eval/match/point_delta_mean",
    "eval/match/positive_point_delta_rate",
    "eval/kyoku/win_rate",
    "eval/kyoku/deal_in_rate",
    "eval/kyoku/deal_in_points_mean",
    "eval/kyoku/tsumo_loss_rate",
    "eval/kyoku/draw_rate",
    "eval/kyoku/point_delta_mean",
    "eval/efficiency/optimal_shanten_rate",
    "eval/efficiency/optimal_ukeire_rate",
    "eval/action/riichi_opportunity_accept_rate",
    "eval/action/call_opportunity_accept_rate",
    "eval/performance/elapsed_s",
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
    "rollout/games": "Rollout/完整半庄数 (games)",
    "rollout/kyokus": "Rollout/小局数 (kyokus)",
    "rollout/grp_calls": "Rollout/GRP 调用次数 (grp_calls)",
    "rollout/reward_mean": "Rollout/GRP 平均奖励 (reward_mean)",
    "ppo/advantage_mean": "PPO/GAE 优势均值 (advantage_mean)",
    "ppo/advantage_std": "PPO/GAE 优势标准差 (advantage_std)",
    "ppo/value_explained_variance": "PPO/Value 解释方差 (value_explained_variance)",
    "ppo/buffer/advantage_mean": "PPO/Buffer 优势均值",
    "ppo/buffer/advantage_std": "PPO/Buffer 优势标准差",
    "ppo/buffer/value_mean": "PPO/Buffer Value 均值",
    "ppo/buffer/value_std": "PPO/Buffer Value 标准差",
    "ppo/offense_projection_weight_norm": "PPO/Offense Projection 权重范数",
    "ppo/offense_projection_grad_norm": "PPO/Offense Projection 梯度范数",
    "ppo/entropy": "PPO/策略熵 (entropy)",
    "ppo/entropy_normalized": "PPO/归一化策略熵 (entropy_normalized)",
    "ppo/approx_kl": "PPO/近似 KL (approx_kl)",
    "ppo/sft_reference_kl": "PPO/SFT 参考策略 KL (sft_reference_kl)",
    "ppo/clipfrac": "PPO/裁剪比例 (clipfrac)",
    "ppo/ratio_p95": "PPO/概率比 P95 (ratio_p95)",
    "ppo/grad_norm": "PPO/梯度范数 (grad_norm)",
    "ppo/grad_norm_post_clip": "PPO/裁剪后梯度范数 (grad_norm_post_clip)",
    "ppo/grad_norm_actor": "PPO/Actor 分支梯度范数·裁剪前 (grad_norm_actor)",
    "ppo/grad_norm_critic": "PPO/Critic 分支梯度范数·裁剪前 (grad_norm_critic)",
    "ppo/grad_norm_shared": "PPO/共享分支梯度范数·裁剪前 (grad_norm_shared)",
    "ppo/system/learning_rate": "PPO/学习率 (learning_rate)",
    "ppo/system/actor_learning_rate": "PPO/Actor 学习率 (actor_learning_rate)",
    "ppo/system/shared_learning_rate": "PPO/Shared 学习率 (shared_learning_rate)",
    "ppo/system/critic_learning_rate": "PPO/Critic 学习率 (critic_learning_rate)",
    "ppo/system/critic_public_grad_scale": "PPO/Critic→Shared 梯度倍率 (critic_public_grad_scale)",
    "ppo/system/entropy_coef": "PPO/熵系数 (entropy_coef)",
    "ppo/system/sft_kl_coef": "PPO/SFT KL 系数 (sft_kl_coef)",
    "ppo/training/critic_bootstrap": "PPO/Critic Bootstrap 阶段 (critic_bootstrap)",
    "ppo/update/early_stop": "PPO/提前停止 (early_stop)",
    "reward_schedule/kyoku_weight": "奖励调度/牌局奖励权重 (kyoku_weight)",
    "train/reward/hanchan_rank_mean": "采样奖励/终局半庄顺位奖励均值",
    "train/reward/hanchan_rank_std": "采样奖励/终局半庄顺位奖励标准差",
    "train/reward/kyoku_mean": "采样奖励/小局分差奖励均值",
    "train/reward/kyoku_std": "采样奖励/小局分差奖励标准差",
    "train/reward/total_mean": "采样奖励/总奖励均值",
    "train/reward/total_std": "采样奖励/总奖励标准差",
    "train/kyoku/draw_rate": "采样牌局/流局率 (draw_rate)",
    "train/kyoku/deal_in_rate": "采样牌局/放铳率·四家汇总 (deal_in_rate)",
    "train/kyoku/discard_count_mean": "采样牌局/平均每局打牌次数 (discard_count)",
    "train/kyoku/open_melds_mean": "采样牌局/平均副露数·四家合计 (open_melds_mean)",
    "train/match/length_kyokus_mean": "采样对局/平均局数 (length_kyokus_mean)",
    "train/action/riichi_opportunity_accept_rate": "采样策略/立直机会接受率 (riichi_opportunity_accept_rate)",
    "train/action/call_opportunity_accept_rate": "采样策略/鸣牌机会接受率 (call_opportunity_accept_rate)",
    "eval/match/first_place_rate": "固定评测/半庄一位率 (first_place_rate)",
    "eval/match/mean_rank": "固定评测/半庄平均顺位 (mean_rank)",
    "eval/match/top2_rate": "固定评测/半庄连对率 (top2_rate)",
    "eval/match/last_place_rate": "固定评测/半庄四位率 (last_place_rate)",
    "eval/match/point_delta_mean": "固定评测/半庄平均分差 (point_delta_mean)",
    "eval/match/positive_point_delta_rate": "固定评测/半庄正收益率 (positive_point_delta_rate)",
    "eval/kyoku/win_rate": "固定评测/小局和牌率 (win_rate)",
    "eval/kyoku/deal_in_rate": "固定评测/小局放铳率 (deal_in_rate)",
    "eval/kyoku/deal_in_points_mean": "固定评测/放铳平均损失·千点 (deal_in_points_mean)",
    "eval/kyoku/tsumo_loss_rate": "固定评测/小局被自摸率 (tsumo_loss_rate)",
    "eval/kyoku/draw_rate": "固定评测/小局流局率 (draw_rate)",
    "eval/kyoku/point_delta_mean": "固定评测/小局平均分差 (point_delta_mean)",
    "eval/efficiency/optimal_shanten_rate": "固定评测/最优向听率 (optimal_shanten_rate)",
    "eval/efficiency/optimal_ukeire_rate": "固定评测/最优进张率 (optimal_ukeire_rate)",
    "eval/action/riichi_opportunity_accept_rate": "固定评测/立直机会接受率 (riichi_opportunity_accept_rate)",
    "eval/action/call_opportunity_accept_rate": "固定评测/鸣牌机会接受率 (call_opportunity_accept_rate)",
    "eval/performance/elapsed_s": "固定评测/耗时·秒 (elapsed_s)",
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
