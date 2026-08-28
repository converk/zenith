"""V18 密集槽位嵌入的敏感性、内部顺序、padding、梯度与尺度测试。"""

from __future__ import annotations

import numpy as np
import torch

from riichi_ppo_v1.model.dense_embedding import StateTokenEmbedding
from riichi_ppo_v1.model.encoding_protocol import (
    CATEGORY_SCHEMAS,
    KIND_RIVER_SUMMARY,
    KIND_TABLE,
    KIND_TILE_STATE,
    TOKEN_ROW_WIDTH,
)


def _row(kind: int, fields: tuple[int, ...]) -> torch.Tensor:
    row = torch.zeros(1, 1, TOKEN_ROW_WIDTH, dtype=torch.long)
    row[0, 0, 0] = CATEGORY_SCHEMAS[kind].segment
    row[0, 0, 1] = kind
    row[0, 0, 2:2 + len(fields)] = torch.tensor(fields, dtype=torch.long)
    return row


def test_single_field_change_changes_embedding() -> None:
    embedding = StateTokenEmbedding(256)
    base = _row(KIND_TABLE, (1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0))
    changed = _row(KIND_TABLE, (1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0))
    e0 = embedding(base, torch.zeros(1, 1, 8))
    e1 = embedding(changed, torch.zeros(1, 1, 8))
    assert not torch.allclose(e0, e1)


def test_summary_internal_order_swap_changes_embedding() -> None:
    embedding = StateTokenEmbedding(256)
    fields_a = (2,) + (1, 0, 1, 0, 2, 0, 0, 1) + (0,) * 16
    fields_b = (2,) + (2, 0, 0, 1, 1, 0, 1, 0) + (0,) * 16
    e0 = embedding(_row(KIND_RIVER_SUMMARY, fields_a), torch.zeros(1, 1, 8))
    e1 = embedding(_row(KIND_RIVER_SUMMARY, fields_b), torch.zeros(1, 1, 8))
    assert not torch.allclose(e0, e1)


def test_padding_slot_zero_contribution() -> None:
    embedding = StateTokenEmbedding(256)
    valid = _row(KIND_RIVER_SUMMARY, (1,) + (1, 0, 1, 0) + (0,) * 20)
    empty = _row(KIND_RIVER_SUMMARY, (0,) + (0,) * 24)
    e0 = embedding(valid, torch.zeros(1, 1, 8))
    e1 = embedding(empty, torch.zeros(1, 1, 8))
    assert not torch.allclose(e0, e1)
    # 合法零值（cut=0）与 padding 不混淆：合法槽位 tile_type=1 使 embedding 非零。
    assert e0.abs().sum() > 0.0


def test_summary_padding_slots_strictly_zero_contribution() -> None:
    """padding 子槽严格零贡献：改写 padding 槽字段不改变输出。"""
    embedding = StateTokenEmbedding(256)
    # valid_length=1：slot2 及其后均为 padding，字段非零也必须是零贡献。
    base = _row(KIND_RIVER_SUMMARY, (1,) + (1, 0, 0, 0) + (9, 1, 1, 1) + (0,) * 16)
    zeroed = _row(KIND_RIVER_SUMMARY, (1,) + (1, 0, 0, 0) + (0,) * 20)
    e0 = embedding(base, torch.zeros(1, 1, 8))
    e1 = embedding(zeroed, torch.zeros(1, 1, 8))
    assert torch.allclose(e0, e1, atol=1e-6)


def test_summary_slot_internal_field_order_is_concatenated() -> None:
    """槽内字段按 concat 而非求和：交换槽内字段值必须改变 embedding。"""
    embedding = StateTokenEmbedding(256)
    a = _row(KIND_RIVER_SUMMARY, (2,) + (1, 0, 2, 0, 2, 0, 1, 0) + (0,) * 16)
    b = _row(KIND_RIVER_SUMMARY, (2,) + (2, 0, 1, 0, 1, 0, 2, 0) + (0,) * 16)
    e0 = embedding(a, torch.zeros(1, 1, 8))
    e1 = embedding(b, torch.zeros(1, 1, 8))
    assert not torch.allclose(e0, e1)


def test_summary_slot_single_field_change_changes_embedding() -> None:
    """槽内单字段改变必须改变 embedding（concat 语义）。"""
    embedding = StateTokenEmbedding(256)
    a = _row(KIND_RIVER_SUMMARY, (2,) + (1, 0, 0, 0, 2, 0, 0, 0) + (0,) * 16)
    b = _row(KIND_RIVER_SUMMARY, (2,) + (1, 0, 1, 0, 2, 0, 0, 0) + (0,) * 16)
    e0 = embedding(a, torch.zeros(1, 1, 8))
    e1 = embedding(b, torch.zeros(1, 1, 8))
    assert not torch.allclose(e0, e1)


def test_content_token_keeps_segment_and_kind_base() -> None:
    """B5：内容 token 的 segment 变化必须改变最终嵌入（基础向量保留）。"""
    embedding = StateTokenEmbedding(256)
    base = _row(KIND_TABLE, (0,) * 11)
    modified = _row(KIND_TABLE, (0,) * 11)
    modified[0, 0, 0] = 2  # 非法组合，但只测嵌入层的 segment 基础向量是否保留
    e0 = embedding(base, torch.zeros(1, 1, 8))
    e1 = embedding(modified, torch.zeros(1, 1, 8))
    assert not torch.allclose(e0, e1)


def test_gradients_reach_all_slot_tables() -> None:
    embedding = StateTokenEmbedding(256)
    row = _row(KIND_TILE_STATE, (5, 1, 2, 1, 3, 3, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 2, 1, 1, 1))
    factors = row
    numeric = torch.zeros(1, 1, 8, requires_grad=True)
    output = embedding(factors, numeric)
    output.sum().backward()
    # 至少 10 个独立 embedding/投影参数收到梯度（TILE_STATE 无数值槽位）。
    grads = [p.grad.abs().sum().item() for p in embedding.parameters() if p.grad is not None]
    assert sum(1 for value in grads if value > 0) >= 10


def test_activation_scale_bounded() -> None:
    embedding = StateTokenEmbedding(256)
    rows = []
    for i in range(16):
        rows.append(_row(KIND_RIVER_SUMMARY, (6,) + (1, 0, 0, 0) * 6))
    factors = torch.cat(rows, dim=0)
    output = embedding(factors, torch.zeros(16, 1, 8))
    assert torch.isfinite(output).all()
    assert output.norm(dim=-1).mean().item() < 20.0


def test_random_combinations_unique() -> None:
    embedding = StateTokenEmbedding(256)
    rng = np.random.default_rng(7)
    factors = []
    for _ in range(32):
        fields = tuple(int(value) for value in rng.integers(0, 9, size=21))
        factors.append(_row(KIND_TILE_STATE, fields))
    output = embedding(torch.cat(factors, dim=0), torch.zeros(32, 1, 8))
    rows = output.reshape(32, -1)
    # 简易去重检查：任意两行不完全相等（允许浮点误差用 round 比较）。
    rounded = torch.round(rows * 1e4)
    unique = torch.unique(rounded, dim=0)
    assert unique.shape[0] >= 30


def test_metadata_ablation_changes_action_embedding_and_logits() -> None:
    from riichi_ppo_v1.model import KyokuTransformerActorCritic
    from riichi_ppo_v1.tests.v18_fixtures import actor_inputs

    model = KyokuTransformerActorCritic()
    inputs = actor_inputs(batch=1, action_ids=(1, 7))
    base = model(
        actor_factors=inputs["actor_factors"], actor_numeric=inputs["actor_numeric"],
        actor_lengths=inputs["actor_lengths"], query_action_ids=inputs["action_ids"],
        query_pair_counts=inputs["query_pair_counts"], legal_mask=inputs["legal_mask"], policy_only=True,
    )
    # 改变一个 answer 槽（answer_0 从 0 改为 1）；action_id 位于 2+4，answer_0 位于 2+5。
    modified = inputs["actor_factors"].clone()
    first_action = torch.nonzero(modified[0, :, 1].eq(11))[0, 0]
    modified[0, first_action, 2 + 5] = 1
    out = model(
        actor_factors=modified, actor_numeric=inputs["actor_numeric"],
        actor_lengths=inputs["actor_lengths"], query_action_ids=inputs["action_ids"],
        query_pair_counts=inputs["query_pair_counts"], legal_mask=inputs["legal_mask"], policy_only=True,
    )
    assert not torch.allclose(base["raw_policy_logits"], out["raw_policy_logits"])
    # action_id 本身进入嵌入：改列 2+4 也必须改变 logits。
    modified2 = inputs["actor_factors"].clone()
    modified2[0, first_action, 2 + 4] = 2
    out2 = model(
        actor_factors=modified2, actor_numeric=inputs["actor_numeric"],
        actor_lengths=inputs["actor_lengths"], query_action_ids=inputs["action_ids"],
        query_pair_counts=inputs["query_pair_counts"], legal_mask=inputs["legal_mask"], policy_only=True,
    )
    assert not torch.allclose(base["raw_policy_logits"], out2["raw_policy_logits"])


def test_kind_row_plan_host_computation() -> None:
    """host 行表:升序、成员正确、未注册 kind 不入表。"""
    from riichi_ppo_v1.model.dense_embedding import compute_kind_row_plan

    registered = {int(kind) for kind in CATEGORY_SCHEMAS}
    kinds = [999, registered.pop(), 5, registered.pop(), 5, 999]
    factors = np.zeros((1, len(kinds), TOKEN_ROW_WIDTH), dtype=np.int64)
    for position, kind in enumerate(kinds):
        factors[0, position, 1] = kind
    plan = compute_kind_row_plan(factors)
    assert set(plan) == {k for k in kinds if k in registered or k in plan}
    for kind_value, indices in plan.items():
        assert list(indices) == [p for p, k in enumerate(kinds) if k == kind_value]
    assert 999 not in plan


def test_forward_plan_path_bitwise_equal_to_legacy_path() -> None:
    """C2:plan 路径与旧 argsort/tolist 路径输出 torch.equal 逐位一致。"""
    from riichi_ppo_v1.model.dense_embedding import compute_kind_row_plan

    torch.manual_seed(11)
    embedding = StateTokenEmbedding(256)
    rng = np.random.default_rng(4)
    batch, tokens = 6, 48
    factors = np.zeros((batch, tokens, TOKEN_ROW_WIDTH), dtype=np.int64)
    # 覆盖:未注册 kind(999)、分隔符(101-111)、BOS/各类别混合、padding 行 0。
    kind_pool = sorted(int(kind) for kind in CATEGORY_SCHEMAS)
    for b in range(batch):
        kinds = [0, *rng.choice(kind_pool, size=tokens - 3, replace=True).tolist(), 999, 0]
        factors[b, :, 0] = [CATEGORY_SCHEMAS[k].segment if k in CATEGORY_SCHEMAS else 1 for k in kinds]
        factors[b, :, 1] = kinds
        fields = rng.integers(0, 4, size=(batch, tokens, 10))
        factors[b, :, 2:12] = fields[b]
    factors_t = torch.from_numpy(factors)
    numeric = torch.from_numpy(rng.random((batch, tokens, 8)).astype(np.float32) * 0.1)
    valid = torch.ones(batch, tokens, dtype=torch.bool)
    valid[:, -6:] = False  # 触发 valid 掩蔽分支

    with torch.no_grad():
        legacy = embedding(factors_t, numeric, valid)
        plan = compute_kind_row_plan(factors)
        fast = embedding(factors_t, numeric, valid, kind_row_plan=plan)
    assert torch.equal(legacy, fast)
    # 空批(全部 padding 行 0 kind)也须一致。
    empty_factors = torch.zeros(2, 5, TOKEN_ROW_WIDTH, dtype=torch.long)
    empty_numeric = torch.zeros(2, 5, 8)
    with torch.no_grad():
        assert torch.equal(
            embedding(empty_factors, empty_numeric),
            embedding(empty_factors, empty_numeric, kind_row_plan=compute_kind_row_plan(empty_factors.numpy())),
        )
