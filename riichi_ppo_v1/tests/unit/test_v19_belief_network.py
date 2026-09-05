"""V19 信念网络模块单测：五头形状、摘要拼接、token 形状与共享转换矩阵、Loss 范围。"""

from __future__ import annotations

import torch

from riichi_ppo_v1.model.belief_network import BeliefNetwork


def _belief_output() -> dict[str, torch.Tensor]:
    """构造一个随机 shared_hidden 并前向信念网络（B=2,T=17,d=256）。"""
    torch.manual_seed(2026)
    network = BeliefNetwork()
    shared_hidden = torch.randn(2, 17, 256)
    return network(shared_hidden)


def test_belief_head_shapes() -> None:
    """五个头与 token 的批量形状必须符合定稿协议。"""
    output = _belief_output()
    assert output["belief_hand_logits"].shape == (2, 3, 34, 5)
    assert output["belief_shanten_logits"].shape == (2, 3, 9)
    assert output["belief_wait_logits"].shape == (2, 3, 35)
    assert output["belief_danger_logits"].shape == (2, 3, 34)
    assert output["belief_loss_pred"].shape == (2, 3, 34)
    # 三家 × 10 token × 256 维。
    assert output["belief_tokens"].shape == (2, 30, 256)


def test_belief_summary_width_and_composition() -> None:
    """摘要为 282 维且逐段拼接顺序与设计一致。"""
    output = _belief_output()
    summary = output["belief_summary"]
    assert summary.shape == (2, 3, 282)
    # 用五个头的输出按规范组合重建摘要：softmax(hand)170 + softmax(shanten)9 +
    # sigmoid(wait)35 + sigmoid(danger)34 + loss_pred(已 sigmoid)34。
    hand = torch.softmax(output["belief_hand_logits"], dim=-1).reshape(2, 3, -1)
    shanten = torch.softmax(output["belief_shanten_logits"], dim=-1)
    wait = torch.sigmoid(output["belief_wait_logits"])
    danger = torch.sigmoid(output["belief_danger_logits"])
    loss = output["belief_loss_pred"]
    expected = torch.cat([hand, shanten, wait, danger, loss], dim=-1)
    torch.testing.assert_close(summary, expected)
    assert summary.shape[-1] == 34 * 5 + 9 + 35 + 34 + 34


def test_loss_pred_in_unit_interval() -> None:
    """Loss 为 sigmoid 归一化回归，预测必须落在 [0,1]。"""
    output = _belief_output()
    loss = output["belief_loss_pred"]
    assert torch.isfinite(loss).all()
    assert float(loss.min()) >= 0.0
    assert float(loss.max()) <= 1.0


def test_token_matrix_shared_between_players() -> None:
    """三家共用同一个转换矩阵：矩阵为单个 282→10×d_model 的 Linear。"""
    network = BeliefNetwork()
    matrix = network.token_matrix
    assert isinstance(matrix, torch.nn.Linear)
    assert matrix.in_features == 282
    assert matrix.out_features == 10 * network.d_model
    # 没有按玩家拆分的多份转换矩阵。
    matrix_weights = [name for name, _ in network.named_parameters() if "token_matrix" in name]
    assert matrix_weights == ["token_matrix.weight", "token_matrix.bias"]
