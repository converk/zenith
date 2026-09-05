"""V19 架构测试：信念 token 注入、mask 可达性、梯度缩放与参数量。

复用 ``riichi_ppo_v1.tests.v19_fixtures`` 的合法合成输入（夹具本身已是 V19
协议；文件名仅历史遗留，后续阶段统一清理）。
"""

from __future__ import annotations

import pytest
import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
from riichi_ppo_v1.model.architecture import _actor_structured_layout
from riichi_ppo_v1.model.encoding_protocol import (
    KIND_BELIEF,
    KIND_SEP_ACTIONS,
    SEGMENT_ACTIONS,
    SEGMENT_ANALYSIS,
    SEGMENT_BELIEF,
    SEGMENT_SHARED,
)
from riichi_ppo_v1.model.parameter_count import assert_v19_parameter_contract
from riichi_ppo_v1.tests.v19_fixtures import actor_inputs, critic_inputs


def _tiny_config() -> ModelConfig:
    """测试用小拓扑（belief 网络仍用默认 512 隐藏/10 token）。"""
    return ModelConfig(
        layers=2,
        shared_layers=1,
        critic_layers=1,
        d_model=32,
        query_heads=4,
        kv_heads=1,
        head_dim=8,
        ffn_dim=64,
        dense_slot_dim=8,
        dense_fusion_dim=64,
        context_tokens=320,
    )


def test_forward_runs_and_belief_tokens_stay_internal() -> None:
    """前向可跑；信念 token 只出现在模型输出，不混入 actor factors。"""
    model = KyokuTransformerActorCritic()
    inputs = actor_inputs(batch=2, action_ids=(1, 7, 12))
    critic = critic_inputs(batch=2)
    with torch.no_grad():
        output = model(
            actor_factors=inputs["actor_factors"],
            actor_numeric=inputs["actor_numeric"],
            actor_lengths=inputs["actor_lengths"],
            query_action_ids=inputs["action_ids"],
            query_pair_counts=inputs["query_pair_counts"],
            legal_mask=inputs["legal_mask"],
            critic_factors=critic["critic_factors"],
            critic_lengths=critic["critic_lengths"],
        )
        policy_only = model(
            actor_factors=inputs["actor_factors"],
            actor_numeric=inputs["actor_numeric"],
            actor_lengths=inputs["actor_lengths"],
            query_action_ids=inputs["action_ids"],
            query_pair_counts=inputs["query_pair_counts"],
            legal_mask=inputs["legal_mask"],
            policy_only=True,
        )
    # 信念输出始终存在（policy_only 也不例外）。
    for key in (
        "belief_hand_logits", "belief_shanten_logits", "belief_wait_logits",
        "belief_danger_logits", "belief_loss_pred", "belief_tokens",
    ):
        assert key in output and key in policy_only
    assert output["belief_hand_logits"].shape == (2, 3, 34, 5)
    assert output["belief_tokens"].shape == (2, 30, 256)
    # 信念是模型内部产物：传入 factors 不含 BELIEF 段/类，输出也没有 factor 张量。
    assert not (inputs["actor_factors"][..., 0] == SEGMENT_BELIEF).any()
    assert not (inputs["actor_factors"][..., 1] == KIND_BELIEF).any()
    assert not any("factor" in key for key in output)
    # 策略读出保持合法掩码契约。
    assert torch.isfinite(output["policy_logits"][inputs["legal_mask"]]).all()
    assert output["policy_logits"][~inputs["legal_mask"]].eq(float("-inf")).all()
    assert output["value"].shape == (2,)


def test_actor_mask_belief_reachability() -> None:
    """mask 可达性：query 可读 belief，belief 不读 query/analysis，belief 互见。"""
    segments = torch.tensor([[
        SEGMENT_SHARED, SEGMENT_SHARED, SEGMENT_SHARED,
        SEGMENT_ANALYSIS, SEGMENT_ANALYSIS, SEGMENT_ANALYSIS,
        SEGMENT_ACTIONS,  # SEP_ACTIONS 行
        SEGMENT_BELIEF, SEGMENT_BELIEF, SEGMENT_BELIEF,
        SEGMENT_ACTIONS, SEGMENT_ACTIONS,  # 一对 O/D query
    ]])
    kinds = torch.tensor([[
        1, 2, 3,
        10, 10, 10,
        KIND_SEP_ACTIONS,
        KIND_BELIEF, KIND_BELIEF, KIND_BELIEF,
        11, 12,
    ]])
    lengths = torch.tensor([12])
    mask, valid = _actor_structured_layout(segments, kinds, lengths, 12)
    assert mask.shape == (1, 12, 12)
    assert bool(valid.all())
    query_idx, belief_idx, analysis_idx = 10, 7, 3
    # query 可读信念；信念不读 query。
    assert mask[0, query_idx, belief_idx]
    assert not mask[0, belief_idx, query_idx]
    # 信念不读 analysis；analysis 不读信念。
    assert not mask[0, belief_idx, analysis_idx]
    assert not mask[0, analysis_idx, belief_idx]
    # 信念互见、查询可读共享与分析。
    assert mask[0, belief_idx, belief_idx + 1]
    assert mask[0, query_idx, 0]
    assert mask[0, query_idx, analysis_idx]


def test_belief_public_grad_scale_scales_encoder_gradient() -> None:
    """belief_public_grad_scale=0.25 时信念分支回传公共 backbone 的梯度约为 1.0 的 1/4。"""
    torch.manual_seed(2026)
    model = KyokuTransformerActorCritic(_tiny_config())
    inputs = actor_inputs(batch=1, action_ids=(1, 7))

    def encoder_grad_norm(scale: float) -> float:
        model.zero_grad(set_to_none=True)
        output = model(
            actor_factors=inputs["actor_factors"],
            actor_numeric=inputs["actor_numeric"],
            actor_lengths=inputs["actor_lengths"],
            query_action_ids=inputs["action_ids"],
            query_pair_counts=inputs["query_pair_counts"],
            legal_mask=inputs["legal_mask"],
            policy_only=True,
            belief_public_grad_scale=scale,
        )
        # 只通过信念分支回传（不反传策略/价值路径），检验缩放边界。
        loss = output["belief_loss_pred"].sum() + output["belief_hand_logits"].sum()
        loss.backward()
        squares = [
            parameter.grad.detach().float().square().sum()
            for parameter in model.public_backbone.parameters()
            if parameter.grad is not None
        ]
        if not squares:
            return 0.0
        return float(torch.stack(squares).sum().sqrt())

    norm_full = encoder_grad_norm(1.0)
    norm_scaled = encoder_grad_norm(0.25)
    assert norm_full > 0.0
    # 线性 detach+重标度在纯信念损失下应给出精确 0.25 比例；放宽容差防浮点/编译误差。
    assert norm_scaled == pytest.approx(0.25 * norm_full, rel=0.10, abs=1e-8)


def test_v19_parameter_contract_range() -> None:
    """V19 全模型参数在 [7.0M, 7.2M]（设计估算 ~7.09M，实际嵌入增删浮动）。"""
    model = KyokuTransformerActorCritic()
    total = sum(parameter.numel() for parameter in model.parameters())
    assert 7_000_000 <= total <= 7_200_000
    report = assert_v19_parameter_contract(model)
    assert report["total"] == total
    assert not report["forbidden_q_keys"]
