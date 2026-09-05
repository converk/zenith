"""V19 密集/简单槽位嵌入的敏感性、梯度、scale 与 RIICHI_CARD 字段敏感性测试。"""

from __future__ import annotations

import numpy as np
import torch

from riichi_ppo_v1.model.dense_embedding import StateTokenEmbedding
from riichi_ppo_v1.model.encoding_protocol import (
    CATEGORY_SCHEMAS,
    KIND_RIICHI_CARD,
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


def test_riichi_card_field_change_changes_embedding() -> None:
    """RIICHI_CARD 是 DENSE 卡：单个字段变化必须改变嵌入（槽位独立表）。"""
    embedding = StateTokenEmbedding(256)
    base = _row(KIND_RIICHI_CARD, (1, 2, 3, 1, 4, 5, 1))
    changed = _row(KIND_RIICHI_CARD, (1, 2, 3, 1, 4, 6, 1))
    e0 = embedding(base, torch.zeros(1, 1, 8))
    e1 = embedding(changed, torch.zeros(1, 1, 8))
    assert not torch.allclose(e0, e1)


def test_content_token_keeps_segment_and_kind_base() -> None:
    embedding = StateTokenEmbedding(256)
    base = _row(KIND_TABLE, (0,) * 11)
    modified = _row(KIND_TABLE, (0,) * 11)
    modified[0, 0, 0] = 2
    e0 = embedding(base, torch.zeros(1, 1, 8))
    e1 = embedding(modified, torch.zeros(1, 1, 8))
    assert not torch.allclose(e0, e1)


def test_gradients_reach_all_slot_tables() -> None:
    embedding = StateTokenEmbedding(256)
    # TILE_STATE 21 个离散字段(0..20 合法域内)。
    fields = (5, 1, 2, 1, 3, 3, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 2, 1, 1, 1)
    factors = _row(KIND_TILE_STATE, fields)
    numeric = torch.zeros(1, 1, 8, requires_grad=True)
    output = embedding(factors, numeric)
    output.sum().backward()
    grads = [p.grad.abs().sum().item() for p in embedding.parameters() if p.grad is not None]
    assert sum(1 for value in grads if value > 0) >= 10


def test_activation_scale_bounded() -> None:
    embedding = StateTokenEmbedding(256)
    rows = []
    for _i in range(16):
        rows.append(_row(KIND_TILE_STATE, (6,) + (0,) * 20))
    factors = torch.cat(rows, dim=0)
    output = embedding(factors, torch.zeros(16, 1, 8))
    assert torch.isfinite(output).all()
    assert output.norm(dim=-1).mean().item() < 20.0


def test_random_combinations_unique() -> None:
    embedding = StateTokenEmbedding(256)
    rng = np.random.default_rng(7)
    factors = []
    for _ in range(32):
        fields = (9,) + tuple(int(value) for value in rng.integers(0, 4, size=20))
        factors.append(_row(KIND_TILE_STATE, fields))
    output = embedding(torch.cat(factors, dim=0), torch.zeros(32, 1, 8))
    rows = output.reshape(32, -1)
    rounded = torch.round(rows * 1e4)
    unique = torch.unique(rounded, dim=0)
    assert unique.shape[0] >= 30
