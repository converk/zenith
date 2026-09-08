"""V19 信念网络模块单测：模糊化四头形状、摘要拼接、token 形状与共享矩阵、Loss 范围。

V19 60% 方案起，``BeliefNetwork`` 接收 backbone 读出的
``player_query_hidden [B,3,3,256]``；四个头共享逐家小头、对每个查询分别
应用后按查询平均 logits。2026-09-07 模糊化：Hand = 16 组 × 3 桶，
Wait = 听牌+宽度桶 5 类。2026-09-08 信念头精简：删除 shanten 头，
Loss 由逐牌回归改为铳点分桶分类（6 桶）+ 桶期望 belief_loss_expected。
"""

from __future__ import annotations

import torch

from riichi_ppo_v1.model.belief_network import (
    LOSS_BUCKET_CENTERS,
    LOSS_NORM_MAX,
    BeliefNetwork,
)


def _belief_output() -> dict[str, torch.Tensor]:
    """构造随机 player_query_hidden 并前向信念网络（B=2，玩家×3 查询×d=256）。"""
    torch.manual_seed(2026)
    network = BeliefNetwork()
    player_query_hidden = torch.randn(2, 3, 3, 256)
    return network(player_query_hidden)


def test_belief_head_shapes() -> None:
    """四个头、桶期望与 token 的批量形状必须符合模糊化协议。"""
    output = _belief_output()
    assert output["belief_hand_logits"].shape == (2, 3, 16, 3)
    assert output["belief_wait_logits"].shape == (2, 3, 5)
    assert output["belief_danger_logits"].shape == (2, 3, 34)
    assert output["belief_loss_bucket_logits"].shape == (2, 3, 34, 6)
    assert output["belief_loss_expected"].shape == (2, 3, 34)
    # 三家 × 8 token × 256 维（2026-09-08 起 10/家 → 8/家）。
    assert output["belief_tokens"].shape == (2, 24, 256)


def test_belief_summary_width_and_composition() -> None:
    """摘要为 121 维且逐段拼接顺序与设计一致。"""
    output = _belief_output()
    summary = output["belief_summary"]
    assert summary.shape == (2, 3, 121)
    # softmax(hand)48 + softmax(wait)5 + sigmoid(danger)34 + loss_expected 34。
    hand = torch.softmax(output["belief_hand_logits"], dim=-1).reshape(2, 3, -1)
    wait = torch.softmax(output["belief_wait_logits"], dim=-1)
    danger = torch.sigmoid(output["belief_danger_logits"])
    expected = torch.cat(
        [hand, wait, danger, output["belief_loss_expected"]], dim=-1,
    )
    torch.testing.assert_close(summary, expected)
    assert summary.shape[-1] == 16 * 3 + 5 + 34 + 34


def test_loss_expected_in_unit_interval() -> None:
    """Loss 期望为桶概率对桶中心的凸组合，预测必须落在 [0,1]。"""
    output = _belief_output()
    loss = output["belief_loss_expected"]
    assert torch.isfinite(loss).all()
    assert float(loss.min()) >= 0.0
    assert float(loss.max()) <= 1.0
    # 凸组合语义：Σ softmax(bucket)·center（center 已除以 LOSS_NORM_MAX）。
    prob = torch.softmax(output["belief_loss_bucket_logits"], dim=-1)
    centers = torch.tensor(
        [center / LOSS_NORM_MAX for center in LOSS_BUCKET_CENTERS],
        dtype=torch.float32,
    )
    expected = (prob * centers).sum(dim=-1)
    torch.testing.assert_close(loss, expected)


def test_token_matrix_shared_between_players() -> None:
    """三家共用同一个转换矩阵：矩阵为单个 121→8×d_model 的 Linear。"""
    network = BeliefNetwork()
    matrix = network.token_matrix
    assert isinstance(matrix, torch.nn.Linear)
    assert matrix.in_features == 121
    assert matrix.out_features == 8 * network.d_model
    # 没有按玩家拆分的多份转换矩阵。
    matrix_weights = [name for name, _ in network.named_parameters() if "token_matrix" in name]
    assert matrix_weights == ["token_matrix.weight", "token_matrix.bias"]


def test_token_matrix_zero_init_noop_start() -> None:
    """token_matrix 零初始化：初始 token 全零（残差式 no-op 起步）。

    零初始化只影响起点、不影响梯度：策略/BC 损失沿 token 回传时，
    dL/dW = 上游梯度 ⊗ summary（summary 非零），W 必须能从零长出。
    """
    torch.manual_seed(11)
    network = BeliefNetwork()
    player_query_hidden = torch.randn(2, 3, 3, 256)
    output = network(player_query_hidden)
    assert torch.count_nonzero(output["belief_tokens"]) == 0

    # 模拟策略损失沿 token 回传：token 逐元素求和的非零上游梯度。
    output["belief_tokens"].sum().backward()
    assert network.token_matrix.weight.grad is not None
    assert float(network.token_matrix.weight.grad.abs().sum()) > 0.0
    assert network.token_matrix.bias.grad is not None


def test_player_query_order_and_per_query_average() -> None:
    """九个查询按「玩家主序 × 查询序」排列，四头对每查询应用后按查询平均。"""
    torch.manual_seed(7)
    network = BeliefNetwork()
    batch = 3
    hidden = torch.randn(batch, 3, 3, 256)
    output = network(hidden)

    flat = hidden.reshape(batch * 9, 256)
    expected_hand = network.hand_head(flat).view(
        batch, 3, 3, 16, 3,
    ).mean(dim=2)
    expected_wait = network.wait_head(flat).view(
        batch, 3, 3, 5,
    ).mean(dim=2)
    expected_danger = network.danger_head(flat).view(
        batch, 3, 3, 34,
    ).mean(dim=2)
    expected_loss = network.loss_bucket_head(flat).view(
        batch, 3, 3, 34, 6,
    ).mean(dim=2)
    torch.testing.assert_close(output["belief_hand_logits"], expected_hand)
    torch.testing.assert_close(output["belief_wait_logits"], expected_wait)
    torch.testing.assert_close(output["belief_danger_logits"], expected_danger)
    torch.testing.assert_close(output["belief_loss_bucket_logits"], expected_loss)
    # p0 的 hidden 与 p1 不同：输出也必须逐家不同（顺序锁定的可观察性）。
    assert not torch.equal(
        output["belief_danger_logits"][0, 0], output["belief_danger_logits"][0, 1],
    )
