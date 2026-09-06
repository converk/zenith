"""V19 信念网络策略梯度隔离单测：监督单源。

硬性要求（见 `audit/reports/v19/design/V19_信念网络策略梯度隔离_实施方案.md`）：
1. 策略/BC 损失只能更新 ``token_matrix``，不得回传信念五头、belief backbone 与
   belief_query；
2. 信念网络（五头 + 1 层 backbone + belief_query）的梯度完全来自五头监督标签；
3. SFT 与 PPO 均使用 ``belief_readout_detach=True``（配置与默认值在
   ``v19_ppo.yaml`` / ``learner.py`` / ``test_v19_ppo_config.py`` 锁定）。
"""

from __future__ import annotations

import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.tests.v19_fixtures import actor_inputs
from riichi_ppo_v1.training.belief import belief_losses


def _model_and_inputs(batch: int = 2) -> tuple[KyokuTransformerActorCritic, dict[str, torch.Tensor]]:
    """构造默认 V19 模型与合成输入（CPU，测试规模）。"""
    torch.manual_seed(2026)
    model = KyokuTransformerActorCritic()
    inputs = actor_inputs(batch=batch, action_ids=(1, 7, 12))
    return model, inputs


def _forward_policy_only(
    model: KyokuTransformerActorCritic,
    inputs: dict[str, torch.Tensor],
    *,
    belief_readout_detach: bool = True,
) -> dict[str, torch.Tensor]:
    """走与训练一致的 policy-only 前向，返回模型输出字典。"""
    return model(
        actor_factors=inputs["actor_factors"],
        actor_numeric=inputs["actor_numeric"],
        actor_lengths=inputs["actor_lengths"],
        query_action_ids=inputs["action_ids"],
        query_pair_counts=inputs["query_pair_counts"],
        legal_mask=inputs["legal_mask"],
        policy_only=True,
        belief_public_grad_scale=0.25,
        belief_readout_enabled=True,
        belief_readout_detach=belief_readout_detach,
        validate_structure=False,
    )


def _belief_heads_have_no_grad(model: KyokuTransformerActorCritic) -> bool:
    """信念五头、belief backbone 与 belief_query 均无梯度（token_matrix 可除外）。"""
    for name, parameter in model.named_parameters():
        is_belief_private = (
            name.startswith("belief_backbone.")
            or name == "belief_query"
            or (
                name.startswith("belief_network.")
                and "token_matrix" not in name
            )
        )
        if is_belief_private and parameter.grad is not None:
            return False
    return True


def test_policy_loss_updates_token_matrix_but_not_belief_network() -> None:
    """BC/策略损失只更新 token_matrix，不进入私有信念网络。"""
    model, inputs = _model_and_inputs()
    output = _forward_policy_only(model, inputs, belief_readout_detach=True)
    log_probs = torch.log_softmax(output["policy_logits"], dim=-1)
    actions = inputs["action_ids"][:, 0]
    policy_loss = -log_probs.gather(1, actions[:, None]).mean()
    policy_loss.backward()

    # token_matrix 是「信念 → 策略 token」接口，必须仍由策略梯度更新。
    assert model.belief_network.token_matrix.weight.grad is not None
    assert model.belief_network.token_matrix.bias.grad is not None
    # 读出投影只由 actor 损失训练，detach 语义下依旧有梯度。
    assert model.belief_readout.proj.weight.grad is not None
    # 私有信念网络（五头 + backbone + belief_query）不接收策略梯度。
    for name in (
        "belief_network.hand_head.weight",
        "belief_network.shanten_head.weight",
        "belief_network.wait_head.weight",
        "belief_network.danger_head.weight",
        "belief_network.loss_head.weight",
    ):
        param = model
        for part in name.split("."):
            param = getattr(param, part)
        assert param.grad is None, f"{name} 不应有策略梯度"
    assert _belief_heads_have_no_grad(model)


def test_supervised_loss_updates_belief_network_only_path() -> None:
    """五头监督损失更新信念网络，且不经过 token_matrix。"""
    model, inputs = _model_and_inputs()
    output = _forward_policy_only(model, inputs, belief_readout_detach=True)
    batch = {
        "belief_hand": torch.randint(0, 5, (2, 102), dtype=torch.long),
        "belief_shanten": torch.randint(0, 9, (2, 3), dtype=torch.long),
        "belief_wait": torch.randint(0, 2, (2, 105), dtype=torch.float32),
        "belief_danger": torch.randint(0, 2, (2, 102), dtype=torch.float32),
        "belief_loss": torch.rand(2, 102, dtype=torch.float32) * 24000.0,
    }
    parts = belief_losses(
        output,
        batch,
        head_weights={
            "hand": 1.0,
            "shanten": 1.0,
            "wait": 1.0,
            "danger": 1.0,
            "loss": 1.0,
        },
        wait_danger_weight=0.0,
    )
    parts["belief_loss_total"].backward()

    # 五头、backbone 与 belief_query 都从监督路径获得梯度。
    for name in (
        "belief_network.hand_head.weight",
        "belief_network.shanten_head.weight",
        "belief_network.wait_head.weight",
        "belief_network.danger_head.weight",
        "belief_network.loss_head.weight",
        "belief_query",
    ):
        param = model
        for part in name.split("."):
            param = getattr(param, part)
        assert param.grad is not None, f"{name} 应有监督梯度"
    backbone_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("belief_backbone.") and parameter.grad is not None
    ]
    assert backbone_grads, "belief_backbone 应由监督损失更新"
    # 监督损失不经过 token 路径：token_matrix 无梯度。
    assert model.belief_network.token_matrix.weight.grad is None
    assert model.belief_network.token_matrix.bias.grad is None
