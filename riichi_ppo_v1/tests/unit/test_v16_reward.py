"""V16 GRP+分差奖励公式、固定 σ 与边界调用次数(SC-011)的契约测试。"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from riichi_ppo_v1.training.grp.reward import (
    REWARD_CLIP,
    SCORE_DELTA_CLIP,
    combined_reward,
    grp_delta,
    grp_expected_value,
    load_normalization,
    normalized_score_reward,
    rank_utility,
)
from riichi_ppo_v1.training.worker import V16GrpBoundaryTracker


def test_rank_utility_is_zero_sum() -> None:
    assert [rank_utility(rank) for rank in range(4)] == [24.0, 8.0, -12.0, -20.0]
    assert sum(rank_utility(rank) for rank in range(4)) == 0.0


def test_grp_expected_value_and_boundary_delta() -> None:
    logits = torch.tensor([[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0]])
    assert float(grp_expected_value(logits)[0]) == 24.0
    assert float(grp_expected_value(logits)[1]) == 8.0
    assert float(grp_delta(logits)[0]) == -16.0


def test_reward_formula_uses_frozen_sigmas_and_clips() -> None:
    # R_GRP=20/σ=4→5(在 ±10 内);Δscore=30000→clip 24/σ=4→6;R=0.7·5+0.3·6=5.3。
    assert abs(combined_reward(20.0, 30_000.0, 4.0, 4.0) - 5.3) < 1e-6
    # 分差先截断到 ±24000 点(±24),再除以 σ 并截断到 ±10。
    assert normalized_score_reward(12_000, 1.0) == 10.0
    assert normalized_score_reward(50_000, 1.0) == 10.0
    assert normalized_score_reward(-50_000, 1.0) == -10.0
    assert REWARD_CLIP == 10.0 and SCORE_DELTA_CLIP == 24.0
    # σ 非正必须拒绝(训练期不得动态修改)。
    try:
        combined_reward(1.0, 1.0, 0.0, 1.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("sigma_grp=0 must be rejected")


def test_normalization_stats_load_from_dataset_json() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "dataset.json").write_text(json.dumps({
            "normalization": {"sigma_grp": 2.5, "sigma_score": 3.0},
        }))
        sigma_grp, sigma_score = load_normalization(root)
        assert sigma_grp == 2.5 and sigma_score == 3.0


def test_grp_boundary_calls_equal_boundary_count_not_action_count() -> None:
    tracker = V16GrpBoundaryTracker(sigma_grp=2.0, sigma_score=3.0)
    tracker.register_start(0, 0, grp_value=4.0, score=25_000)
    # 每小局(任意多动作)只调用一次 GRP;调用数 = 边界数。
    for _action in range(50):
        pass
    tracker.boundary_reward(0, 0, grp_value=6.0, score=26_000)
    tracker.boundary_reward(0, 0, grp_value=2.0, score=25_500)
    assert tracker.grp_calls == 2
    # 半庄结束用真实排名 utility 计算终局 delta。
    terminal = tracker.terminal_reward(0, 0, rank=0, score=31_000)
    expected_grp = np.clip((rank_utility(0) - 2.0) / 2.0, -REWARD_CLIP, REWARD_CLIP)
    expected_score = np.clip(
        np.clip((31_000 - 25_500) / 1000.0, -SCORE_DELTA_CLIP, SCORE_DELTA_CLIP) / 3.0,
        -REWARD_CLIP, REWARD_CLIP,
    )
    assert abs(terminal - (0.7 * expected_grp + 0.3 * expected_score)) < 1e-6
