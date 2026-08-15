"""V16 GRP 模型、视角旋转、prefix 标签与冻结的契约测试。"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from riichi_ppo_v1.model.grp import GRPModel, GRP_CATEGORIES, GRP_NUMERIC_FEATURES
from riichi_ppo_v1.training.grp.prepare import (
    Boundary,
    KyokuResult,
    encode_view,
    rank_among,
)


def _boundaries() -> list[Boundary]:
    start = KyokuResult(
        result_type=1, winner=0, deal_in=2, tenpai_mask=0b0101,
        deltas=(6000, -2000, -2000, -2000),
    )
    return [
        Boundary(0, 0, 0, 0, 0, (25000, 25000, 25000, 25000), None),
        Boundary(0, 1, 0, 1, 0, (31000, 23000, 23000, 23000), start),
    ]


def test_four_view_rotation_is_unique_and_relative() -> None:
    boundaries = _boundaries()
    views = [encode_view(boundaries, viewer) for viewer in range(4)]
    assert len(views) == 4
    for viewer, (categorical, numeric) in enumerate(views):
        assert categorical.shape == (2, len(GRP_CATEGORIES))
        assert numeric.shape == (2, GRP_NUMERIC_FEATURES)
        # 自身顺位恒为 1..4;庄家相对 0..3;上一小局在首局为 START 哨兵。
        assert bool((categorical[:, 0] < 4).all())
        assert bool((categorical[:, 3] < 4).all())
        assert int(categorical[0, 4]) == 0  # START
        assert int(categorical[0, 6]) == 0  # deal-in N/A
        assert int(categorical[0, 8]) == 0  # 首局无连庄
        # 第二局:庄家 0 相对 viewer 的编码各不相同。
        assert int(categorical[1, 3]) == (0 - viewer) % 4
        # 胜者 0 / 放铳 2 的相对位置随视角旋转。
        assert int(categorical[1, 5]) == (0 - viewer) % 4 + 1
        assert int(categorical[1, 6]) == (2 - viewer) % 4 + 1
        assert int(categorical[1, 8]) == 1  # 胜者 0 继续做庄
    # 不同视角的自身分差符号互斥。
    pressures = [view[1][1, 4:7].copy() for view in views]
    assert any(not (pressures[0] == value).all() for value in pressures[1:])


def test_prefix_labels_supervise_final_rank() -> None:
    labels = torch.tensor([3, 0, 2, 1])
    rank = rank_among(0, scores=(31000, 23000, 23000, 23000))
    assert rank == 0
    # 训练损失对所有 prefix 求 CE,标签固定为视角玩家的最终排名。
    model = GRPModel()
    categorical = torch.zeros(1, 4, len(GRP_CATEGORIES), dtype=torch.long)
    numeric = torch.zeros(1, 4, GRP_NUMERIC_FEATURES)
    lengths = torch.tensor([4])
    logits = model(categorical, numeric, lengths)
    assert logits.shape == (1, 4, 4)
    loss = F.cross_entropy(logits.transpose(1, 2), labels[:1].expand(1, 4))
    assert loss.ndim == 0
    loss.backward()
    assert model.gru.weight_ih_l0.grad is not None


def test_grp_parameter_budget() -> None:
    model = GRPModel()
    total = sum(parameter.numel() for parameter in model.parameters())
    assert 50_000 <= total <= 70_000, f"total={total}"


def test_frozen_grp_weights_do_not_change() -> None:
    model = GRPModel()
    model.freeze()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    assert all(not parameter.requires_grad for parameter in model.parameters())
    categorical = torch.zeros(1, 3, len(GRP_CATEGORIES), dtype=torch.long)
    numeric = torch.zeros(1, 3, GRP_NUMERIC_FEATURES)
    with torch.no_grad():
        model(categorical, numeric, torch.tensor([3]))
    probe = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([probe], lr=0.1)
    optimizer.zero_grad()
    probe.backward()
    optimizer.step()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
