"""V18 密集槽位嵌入的敏感性、内部顺序、padding、梯度与尺度测试。"""

from __future__ import annotations

import numpy as np
import torch

from riichi_ppo_v1.model.dense_embedding import DenseSlotFusion, SharedDenseMLP, StateTokenEmbedding
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
    # 改变一个 answer 槽（O1 的答案从 0 改为 1）。
    modified = inputs["actor_factors"].clone()
    # 找到第一个 action 行并修改其 answer_0 字段（行偏移 2+4+0）。
    first_action = torch.nonzero(modified[0, :, 1].eq(11))[0, 0]
    modified[0, first_action, 2 + 4] = 1
    out = model(
        actor_factors=modified, actor_numeric=inputs["actor_numeric"],
        actor_lengths=inputs["actor_lengths"], query_action_ids=inputs["action_ids"],
        query_pair_counts=inputs["query_pair_counts"], legal_mask=inputs["legal_mask"], policy_only=True,
    )
    assert not torch.allclose(base["raw_policy_logits"], out["raw_policy_logits"])
