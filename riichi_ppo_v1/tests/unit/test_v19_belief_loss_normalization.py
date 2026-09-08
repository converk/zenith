"""V19 信念归一化基线修正单测（2026-09-08 实测修正）。

背景：wait/danger/loss_bucket 的模糊标签 99.9%+ 为阴性类，标签熵可低至
~1e-7，直接作分母把归一化损失放大 6~8 个数量级（实测 loss_bucket_norm
1.4e8），λ_k=1.0 的四头均衡完全失效。修正：①全部基线下限
``NORM_BASELINE_FLOOR=0.05``；②loss_bucket 基线在其损失实际所在的危险
正例子集上计算（分母与分子测同一分布）。

PPO（``training/belief.belief_losses``）与 SFT（``sft/trainer._belief_losses``）
为同构实现，必须逐项一致（parity 测试锁定）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from riichi_ppo_v1.sft.trainer import _belief_losses as sft_belief_losses
from riichi_ppo_v1.training.belief import belief_losses
from riichi_ppo_v1.training.belief import NORM_BASELINE_FLOOR


def _imbalanced_output_and_batch(
    batch_size: int = 8, positives: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """构造极不平衡的合成批：danger/wait/loss_bucket 标签几乎全为阴性。"""
    output = {
        "belief_hand_logits": torch.zeros(batch_size, 3, 16, 3),
        "belief_wait_logits": torch.zeros(batch_size, 3, 5),
        "belief_danger_logits": torch.zeros(batch_size, 3, 34),
        "belief_loss_bucket_logits": torch.zeros(batch_size, 3, 34, 6),
    }
    batch = {
        # hand 全 0（病态确定分布,熵≈0 → 也必须被下限兜住）。
        "belief_hand": torch.zeros(batch_size, 48, dtype=torch.long),
        # wait 全 0（全非听）。
        "belief_wait": torch.zeros(batch_size, 3, dtype=torch.long),
        "belief_danger": torch.zeros(batch_size, 102, dtype=torch.float32),
        "belief_loss": torch.zeros(batch_size, 102, dtype=torch.float32),
    }
    if positives:
        # 扁平 [B,102](=3 家×34 牌):第 0 行前两格为危险正例(玩家 0 的前两种牌)。
        batch["belief_danger"][0, 0] = 1.0
        batch["belief_danger"][0, 1] = 1.0
        batch["belief_loss"][0, 0] = 1000.0
        batch["belief_loss"][0, 1] = 6000.0
    return output, batch


def test_floor_bounds_all_normalized_components() -> None:
    """全阴性批：基线全部触底 0.05,归一化分量有界且可解析预报。"""
    output, batch = _imbalanced_output_and_batch(positives=False)
    parts = belief_losses(output, batch)
    # hand:均匀 logits 对全 0 标签 CE=ln3,基线触底 → ln3/0.05。
    assert parts["belief/hand_loss_norm"] == pytest.approx(
        math.log(3) / NORM_BASELINE_FLOOR,
    )
    # wait:均匀 logits 对全 0 标签 CE=ln5 → ln5/0.05(修正前为 raw/~1e-7 量级)。
    assert parts["belief/wait_loss_norm"] == pytest.approx(
        math.log(5) / NORM_BASELINE_FLOOR,
    )
    # danger:无正例→加权 BCE 全为 ln2,常数基线≈0 触底 → ln2/0.05。
    assert parts["belief/danger_loss_norm"] == pytest.approx(
        math.log(2) / NORM_BASELINE_FLOOR,
    )
    # 空正例批:loss_bucket 回退全量熵(触底)→ ln6/0.05。
    assert parts["belief/loss_bucket_loss_norm"] == pytest.approx(
        math.log(6) / NORM_BASELINE_FLOOR,
    )
    for key in (
        "belief/hand_loss_norm", "belief/wait_loss_norm",
        "belief/danger_loss_norm", "belief/loss_bucket_loss_norm",
    ):
        assert torch.isfinite(parts[key]), key


def test_loss_bucket_baseline_uses_positive_subset() -> None:
    """loss_bucket 基线 = 危险正例子集的标签熵(分母与分子测同一分布)。

    2 个正例、桶标签 {0,1} → 子集熵 = ln2;均匀 logits 每格 CE=ln6,
    正例 raw>0 → 逐格加权 (1+20)=21,子集均值 = 21·ln6;
    归一化 = 21·ln6/ln2(修正前分母为全量标签熵≈0,数值 1e8 量级)。
    """
    output, batch = _imbalanced_output_and_batch()
    parts = belief_losses(output, batch)
    expected = 21.0 * math.log(6) / math.log(2)
    assert parts["belief/loss_bucket_loss_norm"] == pytest.approx(
        expected, rel=1e-4,
    )
    # 子集基线语义下,loss_bucket 归一化与 hand 同量级(而非 1e8:1)。
    assert parts["belief/loss_bucket_loss_norm"] < 100.0


def test_ppo_sft_belief_loss_parity() -> None:
    """PPO 与 SFT 同构实现必须逐项一致(同一批、同一配置)。"""
    torch.manual_seed(3)
    batch_size = 16
    output = {
        "belief_hand_logits": torch.randn(batch_size, 3, 16, 3),
        "belief_wait_logits": torch.randn(batch_size, 3, 5),
        "belief_danger_logits": torch.randn(batch_size, 3, 34),
        "belief_loss_bucket_logits": torch.randn(batch_size, 3, 34, 6),
    }
    # 真实分布形态:阴性占绝对多数 + 少量正例。
    batch = {
        "belief_hand": torch.randint(0, 3, (batch_size, 48), dtype=torch.long),
        "belief_wait": torch.zeros(batch_size, 3, dtype=torch.long),
        "belief_danger": torch.zeros(batch_size, 102, dtype=torch.float32),
        "belief_loss": torch.zeros(batch_size, 102, dtype=torch.float32),
    }
    batch["belief_wait"] = torch.randint(0, 3, (batch_size, 3))  # 少量非听
    for _ in range(6):
        b = int(torch.randint(0, batch_size, (1,)))
        t = int(torch.randint(0, 102, (1,)))
        batch["belief_danger"][b, t] = 1.0
        batch["belief_loss"][b, t] = float(np.random.default_rng(0).integers(1000, 24000))
    config = {"belief_sft_coef": 1.0}

    ppo = belief_losses(output, batch)
    sft = sft_belief_losses(output, batch, config)

    pairs = (
        ("belief/hand_loss_norm", "belief_hand_loss_norm"),
        ("belief/wait_loss_norm", "belief_wait_loss_norm"),
        ("belief/danger_loss_norm", "belief_danger_loss_norm"),
        ("belief/loss_bucket_loss_norm", "belief_loss_bucket_loss_norm"),
        ("belief_loss_total", "belief_loss_total"),
    )
    for ppo_key, sft_key in pairs:
        assert torch.allclose(
            ppo[ppo_key].detach(), sft[sft_key].detach(), rtol=1e-6,
        ), f"{ppo_key} 与 {sft_key} 不同构"
