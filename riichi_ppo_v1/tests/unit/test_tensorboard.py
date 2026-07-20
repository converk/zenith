from __future__ import annotations

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from riichi_ppo_v1.training.tensorboard import (
    CURATED_SCALAR_TAGS,
    TENSORBOARD_DISPLAY_TAGS,
    curated_scalars,
    learner_peak_allocated_mb,
    write_curated_scalars,
)


def test_curated_scalars_keep_only_the_dashboard_allowlist() -> None:
    expected = {
        "ppo/policy_loss", "ppo/value_loss", "ppo/entropy", "ppo/entropy_normalized", "ppo/approx_kl", "ppo/clipfrac",
        "ppo/ratio_p95", "ppo/explained_variance", "ppo/grad_norm", "ppo/system/learning_rate",
        "ppo/system/entropy_coef", "ppo/update/early_stop", "reward_schedule/efficiency_weight",
        "reward_schedule/kyoku_weight", "reward_schedule/efficiency_abs_mean", "reward_schedule/kyoku_abs_mean",
        "eval/kyoku/point_delta_mean", "eval/kyoku/point_delta_mean_stderr", "eval/kyoku/win_rate",
        "eval/kyoku/deal_in_rate", "eval/kyoku/draw_rate", "eval/kyoku/discard_count_mean",
        "eval/match/first_place_rate", "eval/match/mean_rank",
        "eval/match/top2_rate", "eval/match/last_place_rate", "eval/efficiency/optimal_shanten_rate",
        "eval/efficiency/optimal_ukeire_rate", "eval/match/length_kyokus_mean",
        "eval/match/discard_count_mean", "train/kyoku/draw_rate", "train/kyoku/discard_count_mean",
        "train/kyoku/open_melds_mean",
        "train/action/riichi_rate",
        "train/action/call_opportunity_accept_rate", "iteration/sps",
        "iteration/algorithm_wall_s", "update/wall_s", "system/learner_gpu_peak_allocated_mb",
    }
    assert CURATED_SCALAR_TAGS == expected
    values = {name: float(index) for index, name in enumerate(expected, start=1)}
    values.update({
        "ppo/loss": 99.0,
        "ppo/buffer/advantage_std": 1.0,
        "rollout/reward_mean": 2.0,
        "train/kyoku/win_rate": 0.5,
        "gpu/memory.used/max": 100.0,
        "iteration/sps": float("nan"),
    })

    result = curated_scalars(values)

    assert set(result) == expected - {"iteration/sps"}
    assert "ppo/loss" not in result
    assert "rollout/reward_mean" not in result


def test_learner_peak_allocated_mb_uses_the_largest_rank_peak() -> None:
    assert learner_peak_allocated_mb([
        {"gpu/torch_memory_peak_allocated_mb": 320.0},
        {"gpu/torch_memory_peak_allocated_mb": 512.0},
        {"gpu/torch_memory_peak_allocated_mb": float("nan")},
    ]) == 512.0
    assert learner_peak_allocated_mb([{}, {"gpu/torch_memory_allocated_mb": 10.0}]) is None


def test_tensorboard_display_tags_are_complete_and_localized() -> None:
    assert set(TENSORBOARD_DISPLAY_TAGS) == CURATED_SCALAR_TAGS
    assert TENSORBOARD_DISPLAY_TAGS["train/kyoku/open_melds_mean"] == "采样牌局/平均副露数·四家合计 (open_melds_mean)"


def test_tensorboard_event_contains_only_curated_scalars_and_selected_histograms(tmp_path) -> None:
    writer = SummaryWriter(str(tmp_path))
    write_curated_scalars(writer, {
        "ppo/ratio_p95": 1.04,
        "ppo/system/entropy_coef": 0.004,
        "ppo/update/early_stop": 1.0,
        "eval/kyoku/point_delta_mean": 2.1,
        "eval/match/mean_rank": 2.3,
        "iteration/sps": 512.0,
        "ppo/loss": 0.5,
        "train/action/riichi_rate": 0.1,
    }, step=7)
    writer.add_histogram("diagnostics/advantage", np.asarray([0.0, 1.0], dtype=np.float32), 7)
    writer.flush()
    writer.close()

    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    assert set(accumulator.Tags()["scalars"]) == {
        "PPO/概率比 P95 (ratio_p95)", "PPO/熵系数 (entropy_coef)", "PPO/提前停止 (early_stop)",
        "评测牌局/平均得分变化 (point_delta_mean)", "评测对局/平均排名 (mean_rank)", "性能/每秒决策数 (sps)",
        "采样策略/立直率 (riichi_rate)",
    }
    assert set(accumulator.Tags()["histograms"]) == {"diagnostics/advantage"}
