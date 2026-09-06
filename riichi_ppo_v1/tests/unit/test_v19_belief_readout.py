"""V19 逐动作信念读出（BeliefActionReadout）单测。

覆盖：特征维度/形状、detach 语义（SFT 下信念头不回传、读出投影仍更新）、
tile_code=0 回退全局特征、零初始化后与关闭 readout 的 logits 一致、训练后
读出可影响 logits。
"""

from __future__ import annotations

import torch
from torch import nn

from riichi_ppo_v1.model.belief_network import BeliefNetwork
from riichi_ppo_v1.model.belief_readout import BELIEF_PLAYERS, FEATURE_DIM, BeliefActionReadout


def _fake_belief(batch: int = 2, queries: int = 4) -> dict[str, torch.Tensor]:
    """构造数值可控的信念输出（不需要真实前向）。"""
    device = torch.device("cpu")
    return {
        "belief_hand_logits": torch.zeros(batch, BELIEF_PLAYERS, 34, 5, device=device),
        "belief_shanten_logits": torch.zeros(batch, BELIEF_PLAYERS, 9, device=device),
        "belief_wait_logits": torch.zeros(batch, BELIEF_PLAYERS, 35, device=device),
        "belief_danger_logits": torch.zeros(batch, BELIEF_PLAYERS, 34, device=device),
        "belief_loss_pred": torch.zeros(batch, BELIEF_PLAYERS, 34, device=device),
    }


def test_feature_and_output_shapes() -> None:
    """输出必须为 [B,Q,d_model]；投影为 21→d_model 的 Linear。"""
    readout = BeliefActionReadout(256)
    belief = _fake_belief(batch=2, queries=4)
    tile_codes = torch.tensor([[0, 1, 5, 34], [7, 0, 12, 3]])
    output = readout(belief, tile_codes, detach=True)
    assert output.shape == (2, 4, 256)
    assert isinstance(readout.proj, nn.Linear)
    assert readout.proj.in_features == FEATURE_DIM
    assert readout.proj.out_features == 256
    assert torch.isfinite(output).all()


def test_zero_init_projection() -> None:
    """投影权重/bias 零初始化：未训练时读出输出恒为零。"""
    readout = BeliefActionReadout(256)
    assert float(readout.proj.weight.abs().max()) == 0.0
    assert float(readout.proj.bias.abs().max()) == 0.0
    output = readout(_fake_belief(batch=1, queries=4), torch.tensor([[0, 1, 5, 34]]), detach=False)
    assert float(output.abs().max()) == 0.0


def test_detach_true_blocks_belief_gradients_but_updates_projection() -> None:
    """detach=True：只用读出损失时信念网络参数无梯度，读出投影有梯度。"""
    torch.manual_seed(0)
    network = BeliefNetwork()
    hidden = torch.randn(2, 3, 3, 256)
    belief = network(hidden)
    readout = BeliefActionReadout(256)
    tile_codes = torch.tensor([[0, 1, 5], [7, 0, 12]])
    output = readout(belief, tile_codes, detach=True)
    output.sum().backward()
    assert readout.proj.weight.grad is not None
    assert float(readout.proj.weight.grad.abs().sum()) > 0.0
    # 信念头与共享层不出现在读出回传图中。
    finding = False
    for name, parameter in network.named_parameters():
        if parameter.grad is not None:
            finding = True
            break
    assert not finding


def test_detach_false_allows_belief_gradients() -> None:
    """detach=False：读出损失可回传信念头（模块级能力；当前 SFT/PPO 训练均恒 detach）。"""
    torch.manual_seed(0)
    network = BeliefNetwork()
    hidden = torch.randn(2, 3, 3, 256)
    belief = network(hidden)
    readout = BeliefActionReadout(256)
    tile_codes = torch.tensor([[0, 1, 5], [7, 0, 12]])
    output = readout(belief, tile_codes, detach=False)
    output.sum().backward()
    assert readout.proj.weight.grad is not None
    # wait/danger/loss/shanten 头参与特征构造，应获得梯度。
    grads = [
        name for name, parameter in network.named_parameters()
        if parameter.grad is not None
    ]
    assert grads
    assert any("wait_head" in name or "danger_head" in name for name in grads)


def test_tile_code_zero_keeps_global_and_zeroes_tile_features() -> None:
    """tile_code=0 时逐牌特征置零、全局项保留。"""
    readout = BeliefActionReadout(256)
    batch, queries = 1, 2
    belief = _fake_belief(batch, queries)
    # 让 danger/wait 的逐牌 sigmoid 都为 1（logits 很大），loss 也为 1。
    belief["belief_danger_logits"] = torch.full((batch, BELIEF_PLAYERS, 34), 20.0)
    belief["belief_wait_logits"] = torch.full((batch, BELIEF_PLAYERS, 35), 20.0)
    belief["belief_loss_pred"] = torch.ones(batch, BELIEF_PLAYERS, 34)
    tile_codes = torch.tensor([[0, 1]])
    with torch.no_grad():
        # 只读取逐牌特征列（前 3+次 3+最后 3），全局列权重为零。
        mask = torch.cat([
            torch.ones(3), torch.ones(3), torch.zeros(3),
            torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.ones(3),
        ])[None, :]
        readout.proj.weight.zero_()
        readout.proj.weight[0] = mask
        readout.proj.bias.zero_()
        output = readout(belief, tile_codes, detach=True)
    assert float(output[0, 0, 0]) == 0.0
    assert abs(float(output[0, 1, 0]) - 9.0) < 1e-6


def test_readout_changes_logits_after_training() -> None:
    """零初始化下与关闭 readout 的 logits 一致；投影非零后读出会改变 logits。

    直接使用真实模型前向（默认 detach=True 与关闭 readout 对照）。
    """
    from riichi_ppo_v1.model import KyokuTransformerActorCritic
    from riichi_ppo_v1.tests.v19_fixtures import actor_inputs

    torch.manual_seed(2026)
    model = KyokuTransformerActorCritic()
    inputs = actor_inputs(batch=1, action_ids=(1, 7, 12))
    with torch.no_grad():
        off = model(
            actor_factors=inputs["actor_factors"],
            actor_numeric=inputs["actor_numeric"],
            actor_lengths=inputs["actor_lengths"],
            query_action_ids=inputs["action_ids"],
            query_pair_counts=inputs["query_pair_counts"],
            legal_mask=inputs["legal_mask"],
            policy_only=True,
            belief_readout_enabled=False,
        )["policy_logits"]
        on = model(
            actor_factors=inputs["actor_factors"],
            actor_numeric=inputs["actor_numeric"],
            actor_lengths=inputs["actor_lengths"],
            query_action_ids=inputs["action_ids"],
            query_pair_counts=inputs["query_pair_counts"],
            legal_mask=inputs["legal_mask"],
            policy_only=True,
            belief_readout_enabled=True,
            belief_readout_detach=True,
        )["policy_logits"]
    assert torch.equal(on, off)
    # 手动置非零投影（模拟训练一步后），读出必须进入策略 logits 的梯度图/数值。
    with torch.no_grad():
        model.belief_readout.proj.weight.fill_(0.01)
        model.belief_readout.proj.bias.fill_(0.01)
        trained_on = model(
            actor_factors=inputs["actor_factors"],
            actor_numeric=inputs["actor_numeric"],
            actor_lengths=inputs["actor_lengths"],
            query_action_ids=inputs["action_ids"],
            query_pair_counts=inputs["query_pair_counts"],
            legal_mask=inputs["legal_mask"],
            policy_only=True,
            belief_readout_enabled=True,
            belief_readout_detach=True,
        )["policy_logits"]
    assert not torch.equal(trained_on, off)
