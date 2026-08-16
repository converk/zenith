"""V16 网络结构的契约测试(US2 硬门槛)。

覆盖参数量预算、d_model=Q×head_dim、Offense/Defense 对称融合、无 zero-init、
无 241 维 Q head,以及 v16 前向的 smoke。
"""

from __future__ import annotations

import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig


def _v16_config(*, context_tokens: int = 128) -> ModelConfig:
    """真实 v16 preset 的小容量替身参数也必须满足结构不变量。"""
    return ModelConfig(
        layers=5,
        shared_layers=4,
        critic_layers=2,
        d_model=256,
        query_heads=16,
        kv_heads=4,
        head_dim=16,
        ffn_dim=1088,
        context_tokens=context_tokens,
        policy_head_type="symmetric_action_query",
    )


def _v16_inputs(batch: int = 1, action_ids: tuple[int, ...] = (0, 5)) -> dict:
    """构造合法的 v16 前向输入:历史 3 token + 快照 + 每动作一对 query。"""
    history_factors = torch.zeros(batch, 3, 10, dtype=torch.long)
    history_numeric = torch.zeros(batch, 3, 8)
    if batch and history_factors.shape[1]:
        history_factors[:, 0] = torch.tensor([1, 1, 4, 1, 1, 1, 0, 0, 2, 1])
        history_factors[:, 1] = torch.tensor([2, 2, 1, 1, 0, 0, 0, 0, 0, 0])
        history_numeric[:, 1, 0] = 1.0
    history_lengths = torch.full((batch,), 3, dtype=torch.long)

    # 快照:base(1)+ dora(1)+ score(1)+ summary(3)= 6 行。
    snapshot_kinds = torch.zeros(batch, 6, dtype=torch.long)
    snapshot_kinds[:, 1] = 1
    snapshot_kinds[:, 2] = 2
    snapshot_kinds[:, 3:] = 3
    snapshot_cat = torch.zeros(batch, 6, 4, dtype=torch.long)
    snapshot_num = torch.zeros(batch, 6, 7)
    snapshot_lengths = torch.full((batch,), 6, dtype=torch.long)

    queries = 2 * len(action_ids)
    query_rows = torch.zeros(batch, queries, 15, dtype=torch.long)
    for pair, action in enumerate(action_ids):
        offense, defense = 2 * pair, 2 * pair + 1
        query_rows[:, offense, 0] = 1
        query_rows[:, defense, 0] = 2
        query_rows[:, offense, 1] = action
        query_rows[:, defense, 1] = action
    query_action_ids = torch.tensor([list(action_ids)], dtype=torch.long)
    query_pair_counts = torch.full((batch,), len(action_ids), dtype=torch.long)
    legal = torch.zeros(batch, 241, dtype=torch.bool)
    for action in action_ids:
        legal[:, action] = True
    return {
        "history_factors": history_factors,
        "history_numeric": history_numeric,
        "history_lengths": history_lengths,
        "snapshot_kinds": snapshot_kinds,
        "snapshot_cat": snapshot_cat,
        "snapshot_num": snapshot_num,
        "snapshot_lengths": snapshot_lengths,
        "query_rows": query_rows,
        "query_action_ids": query_action_ids,
        "query_pair_counts": query_pair_counts,
        "legal_mask": legal,
    }


def test_v16_preset_matches_design_table() -> None:
    config = ModelConfig.preset("v16")
    assert config.d_model == 256
    assert config.query_heads == 16
    assert config.kv_heads == 4
    assert config.head_dim == 16
    assert config.ffn_dim == 1088
    assert config.shared_layers == 4
    assert config.layers - config.shared_layers == 1
    assert config.critic_layers == 2
    assert config.policy_head_type == "symmetric_action_query"
    assert config.critic_head_type == "state_value"


def test_d_model_must_equal_heads_times_head_dim() -> None:
    assert ModelConfig.preset("v16").d_model == 16 * 16
    try:
        ModelConfig(d_model=257, query_heads=16, kv_heads=4, head_dim=16)
    except ValueError as error:
        assert "d_model" in str(error)
    else:  # pragma: no cover
        raise AssertionError("d_model != Q×head_dim 必须被拒绝")


def test_v16_parameter_budget() -> None:
    model = KyokuTransformerActorCritic(ModelConfig.preset("v16"))
    total = sum(parameter.numel() for parameter in model.parameters())
    actor_modules = (
        model.token_embedding,
        model.snapshot_embeddings,
        model.query_embedding,
        model.public_backbone,
        model.actor_backbone,
        model.action_fusion,
        model.policy_mlp,
    )
    actor = sum(parameter.numel() for module in actor_modules for parameter in module.parameters())
    # 设计钦定 7.5–7.8M 总参数 / 约 5.3M Actor 推理,容差 ±0.3M。
    assert 7_200_000 <= total <= 8_100_000, f"total={total}"
    assert 5_000_000 <= actor <= 5_600_000, f"actor={actor}"


def test_offense_defense_symmetric_fusion_without_zero_init() -> None:
    model = KyokuTransformerActorCritic(ModelConfig.preset("v16"))
    # 对称融合:concat 512→256 的单一 Linear,两分支共用,不存在独立 offense 投影。
    assert not hasattr(model, "offense_projection") or model.offense_projection is None
    fusion_linear = model.action_fusion[0]
    assert isinstance(fusion_linear, torch.nn.Linear)
    assert fusion_linear.in_features == 2 * model.config.d_model
    assert fusion_linear.out_features == model.config.d_model
    # 普通初始化:策略融合路径不允许 zero-init。
    assert torch.count_nonzero(fusion_linear.weight) > 0
    assert torch.count_nonzero(fusion_linear.bias) > 0
    assert torch.count_nonzero(model.policy_mlp[1].weight) > 0


def test_v16_has_no_241_dim_q_head() -> None:
    model = KyokuTransformerActorCritic(ModelConfig.preset("v16"))
    assert getattr(model, "q_head", None) is None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert module.out_features != 241, f"{name} 仍是 241 维 Q head"
    assert model.value_head is not None


def test_v16_forward_masks_illegal_actions_and_backpropagates() -> None:
    model = KyokuTransformerActorCritic(ModelConfig.preset("v16"))
    output = model.forward_v16(**_v16_inputs(batch=1, action_ids=(0, 5)))
    assert output["policy_logits"].shape == (1, 241)
    assert output["policy_logits"].dtype is torch.float32
    assert output["value"].shape == (1,)
    assert torch.isneginf(output["policy_logits"][:, 1]).all()
    (output["value"].mean() + output["policy_logits"][:, 0].mean()).backward()
    assert model.action_fusion[0].weight.grad is not None
    assert model.value_head.weight.grad is not None


def test_v16_policy_only_forward_skips_critic() -> None:
    model = KyokuTransformerActorCritic(ModelConfig.preset("v16"))
    output = model.forward_v16(**_v16_inputs(batch=1, action_ids=(7,)), policy_only=True)
    assert "policy_logits" in output
    assert "value" not in output


def test_v16_critic_private_inputs_do_not_change_policy_logits() -> None:
    torch.manual_seed(5)
    model = KyokuTransformerActorCritic(ModelConfig.preset("v16")).eval()
    inputs = _v16_inputs(batch=1, action_ids=(0, 5))
    critic_a = torch.ones(1, 2, 10, dtype=torch.long)
    critic_b = torch.full((1, 2, 10), 2, dtype=torch.long)
    with torch.no_grad():
        first = model.forward_v16(
            **inputs,
            critic_factors=critic_a,
            critic_lengths=torch.tensor([2]),
        )
        second = model.forward_v16(
            **inputs,
            critic_factors=critic_b,
            critic_lengths=torch.tensor([2]),
        )
    torch.testing.assert_close(first["raw_policy_logits"], second["raw_policy_logits"])
    assert not torch.allclose(first["value"], second["value"])


def test_v16_forward_is_invariant_to_batchmates_and_padding() -> None:
    """同一行的策略 logits 不得随 batchmates 或 padding 宽度变化。

    回归测试:forward_v16 曾把三个整段 padding 的 tensor 直接 cat,导致短行的
    snapshot/query 落到 padding 空隙、真实内容被 valid mask 屏蔽,同一行的
    输出随 batch 组成改变,进而使 rollout 与 update 重算的 logprob 不一致。
    """
    torch.manual_seed(11)
    model = KyokuTransformerActorCritic(_v16_config(context_tokens=256)).eval()
    single = _v16_inputs(batch=1, action_ids=(0, 5))

    # 第二条是更长的历史 + 更多动作对,迫使第一条被 padding。
    paired = _v16_inputs(batch=2, action_ids=(0, 5))
    history_factors = torch.zeros(2, 10, 10, dtype=torch.long)
    history_numeric = torch.zeros(2, 10, 8)
    history_factors[:, :3] = paired["history_factors"]
    history_numeric[:, :3] = paired["history_numeric"]
    history_factors[1, 3:] = 1
    paired["history_factors"] = history_factors
    paired["history_numeric"] = history_numeric
    paired["history_lengths"][1] = 10
    with torch.no_grad():
        alone = model.forward_v16(**single)
        together = model.forward_v16(**paired)
    torch.testing.assert_close(
        alone["policy_logits"][0],
        together["policy_logits"][0],
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        alone["raw_policy_logits"][0],
        together["raw_policy_logits"][0],
        rtol=1e-4,
        atol=1e-4,
    )
