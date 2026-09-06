"""V19 信念网络模块单测：五头形状、摘要拼接、token 形状与共享转换矩阵、Loss 范围。

V19 60% 方案起，``BeliefNetwork`` 不再接收 mean-pool 的 ``shared_hidden``，
而是接收 backbone 读出的 ``player_query_hidden [B,3,3,256]``；五个头共享逐家
小头、对每个查询分别应用后按查询平均 logits。标签/摘要/token 输出形状不变。
"""

from __future__ import annotations

import torch

from riichi_ppo_v1.model.belief_network import BeliefNetwork


def _belief_output() -> dict[str, torch.Tensor]:
    """构造随机 player_query_hidden 并前向信念网络（B=2，玩家×3 查询×d=256）。"""
    torch.manual_seed(2026)
    network = BeliefNetwork()
    player_query_hidden = torch.randn(2, 3, 3, 256)
    return network(player_query_hidden)


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


def test_player_query_order_and_per_query_average() -> None:
    """九个查询按「玩家主序 × 查询序」排列，五头对每查询应用后按查询平均。

    玩家主序 p0 q0/q1/q2、p1 q0/q1/q2、p2 q0/q1/q2。手动重算前向使用的展平
    顺序与平均，与网络输出逐位一致，锁定实现与 ``belief_labels``/``belief_tokens``
    的玩家顺序（p=0 下家 / p=1 对面 / p=2 上家）一致。
    """
    torch.manual_seed(7)
    network = BeliefNetwork()
    batch = 3
    # 每个玩家/查询使用不同的 hidden（数值可区分），验证平均与玩家切片。
    hidden = torch.randn(batch, 3, 3, 256)
    output = network(hidden)

    flat = hidden.reshape(batch * 9, 256)
    expected_hand = network.hand_head(flat).view(
        batch, 3, 3, 34, 5,
    ).mean(dim=2)
    expected_shanten = network.shanten_head(flat).view(
        batch, 3, 3, 9,
    ).mean(dim=2)
    expected_wait = network.wait_head(flat).view(
        batch, 3, 3, 35,
    ).mean(dim=2)
    expected_danger = network.danger_head(flat).view(
        batch, 3, 3, 34,
    ).mean(dim=2)
    expected_loss = torch.sigmoid(network.loss_head(flat).view(
        batch, 3, 3, 34,
    ).mean(dim=2))
    torch.testing.assert_close(output["belief_hand_logits"], expected_hand)
    torch.testing.assert_close(output["belief_shanten_logits"], expected_shanten)
    torch.testing.assert_close(output["belief_wait_logits"], expected_wait)
    torch.testing.assert_close(output["belief_danger_logits"], expected_danger)
    torch.testing.assert_close(output["belief_loss_pred"], expected_loss)
    # p0 的 hidden 与 p1 不同：输出也必须逐家不同（顺序锁定的可观察性）。
    assert not torch.equal(output["belief_shanten_logits"][0, 0], output["belief_shanten_logits"][0, 1])
