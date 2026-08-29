"""GRP reward 公式、每小局 credit assignment 与边界调用次数的契约测试(Mortal 方案)。"""

from __future__ import annotations

import pytest
import torch

from riichi_ppo_v1.training.grp.prepare import Boundary
from riichi_ppo_v1.training.grp.reward import (
    grp_delta,
    grp_expected_value,
    grp_expected_values_from_matrix,
    rank_utility,
)
from riichi_ppo_v1.training.worker import GrpRollout


def test_rank_utility_matches_mortal_contract() -> None:
    assert [rank_utility(rank) for rank in range(4)] == pytest.approx(
        [1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0]
    )
    assert sum(rank_utility(rank) for rank in range(4)) == pytest.approx(0.0)


def test_grp_expected_value_and_boundary_delta() -> None:
    logits = torch.tensor([[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0]])
    assert float(grp_expected_value(logits)[0]) == pytest.approx(1.0)
    assert float(grp_expected_value(logits)[1]) == pytest.approx(1.0 / 3.0)
    assert float(grp_delta(grp_expected_value(logits))[0]) == pytest.approx(-2.0 / 3.0)


def test_expected_values_from_matrix() -> None:
    # 玩家0 铁定 rank0 → utility 1;玩家3 铁定 rank3 → utility -1。
    matrix = torch.zeros(2, 4, 4)
    matrix[0, 0, 0] = 1.0
    matrix[0, 1, 1] = 1.0
    matrix[0, 2, 2] = 1.0
    matrix[0, 3, 3] = 1.0
    matrix[1] = matrix[0].clone()
    values = grp_expected_values_from_matrix(matrix)
    assert torch.allclose(values[:, 0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(values[:, 3], torch.tensor([-1.0, -1.0]))


def test_grp_boundary_calls_equal_boundary_count() -> None:
    class StubGRP:
        """固定 logits 的 GRP 桩:24 类均匀分布。"""

        def __call__(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            del lengths
            batch = features.shape[0]
            return torch.zeros((batch, 24))

        def calc_matrix(self, logits: torch.Tensor) -> torch.Tensor:
            # logits (N,24) → (N,4,4);均匀 24 类 → 每玩家每排名 6/24=0.25。
            probs = torch.softmax(logits, dim=-1)
            matrix = torch.zeros((probs.shape[0], 4, 4))
            matrix.fill_(0.25)
            return matrix

    tracker = GrpRollout(StubGRP(), game_type=1)
    tracker.start_match(0, Boundary(0, 0, 0, 0, 0, (25_000, 25_000, 25_000, 25_000), None))
    # 首局边界 1 次 GRP 调用(全 4 玩家同一次前向)。
    assert tracker.calls == 1
    # 两个非终局小局边界各 1 次调用;任意多动作不增加调用。
    for _action in range(50):
        pass
    tracker.boundary_reward(
        0, Boundary(0, 0, 0, 0, 0, (26_000, 25_000, 24_000, 25_000), None),
    )
    tracker.boundary_reward(
        0, Boundary(0, 0, 0, 0, 0, (25_500, 25_000, 25_500, 24_000), None),
    )
    assert tracker.calls == 3
    # 均匀分布 → 每玩家 expected utility = mean([1, 1/3, -1/3, -1]) = 0;
    # 所以非终局边界 reward 全为 0。
    rewards = tracker.boundary_reward(
        0,
        Boundary(0, 0, 0, 0, 0, (22_000, 25_000, 26_000, 27_000), None),
    )
    for seat in range(4):
        assert abs(rewards[seat]) < 1e-6
    assert tracker.calls == 4
    # 终局使用真实排名 utility,不再执行 GRP 推理。
    terminal = tracker.boundary_reward(
        0,
        Boundary(0, 0, 0, 0, 0, (31_000, 25_000, 22_000, 22_000), None),
        terminal_ranks={0: 0, 1: 1, 2: 3, 3: 2},
    )
    assert terminal[0] == pytest.approx(1.0)
    assert terminal[1] == pytest.approx(1.0 / 3.0)
    assert terminal[2] == pytest.approx(-1.0)
    assert terminal[3] == pytest.approx(-1.0 / 3.0)
    assert tracker.calls == 4  # 终局不产生 GRP 调用
