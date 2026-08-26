"""V18 当前局面输入的槽位感知嵌入层。

密集类别使用：
- 每个离散槽位一张独立 embedding 表（``dense_slot_dim=32``，code 0 零贡献）；
- 数值槽位先稳定归一化（x / scale，clip 到 [-1,1]）再经专属投影；
- 每个类别把槽位向量按规范顺序 concat 到统一的 ``max_dense_slots`` 宽度（缺失槽位为严格零），
  经**共享**输入投影到 ``dense_fusion_dim=512``，再经**共享** ``RMSNorm + gated/SiLU MLP``
  投影回 ``d_model=256``。

简单类别用轻量 concat（``simple_slot_dim=16``，统一宽度）+ 共享单层投影；
分隔符使用类别专属 learned embedding。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .encoding_protocol import (
    CATEGORY_SCHEMAS,
    CategorySchema,
    is_separator_kind,
    separator_id_of_kind,
)

SIMPLE_SLOT_DIM = 16
DENSE_SLOT_DIM = 32
DENSE_FUSION_DIM = 512

# 统一 concat 宽度：取各类别槽位向量数最大值（TABLE=29，RIVER_SUMMARY=31）。
# 每个 summary 槽按 4 字段 concat + slot_id 共 5 个槽位向量（B6 修复）。
_MAX_DENSE_SLOTS = max(
    len(schema.discrete) - 4 * schema.slot_count + (5 if schema.slot_count else 0) * schema.slot_count + len(schema.numeric)
    for schema in CATEGORY_SCHEMAS.values() if schema.cls == "DENSE"
)
_MAX_SIMPLE_SLOTS = max(
    (len(schema.discrete) + len(schema.numeric) for schema in CATEGORY_SCHEMAS.values() if schema.cls == "SIMPLE"),
    default=1,
)


def _clip_normalize(values: Tensor, scale: float) -> Tensor:
    return (values.float() / scale).clamp(-1.0, 1.0)


class SharedSimpleProjection(nn.Module):
    """简单类别共享 concat→单层投影。"""

    def __init__(self, d_model: int, slot_dim: int = SIMPLE_SLOT_DIM,
                 max_slots: int = _MAX_SIMPLE_SLOTS) -> None:
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.max_slots = int(max_slots)
        self.projection = nn.Linear(self.max_slots * self.slot_dim, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, parts: list[Tensor]) -> Tensor:
        padded = [torch.zeros_like(parts[0]) for _ in range(self.max_slots - len(parts))]
        fused = torch.cat([*parts, *padded], dim=-1)
        return self.norm(self.projection(fused))


class SharedDenseProjection(nn.Module):
    """密集类别共享输入投影（统一宽度 → dense_fusion_dim）。"""

    def __init__(self, fusion_dim: int = DENSE_FUSION_DIM, slot_dim: int = DENSE_SLOT_DIM,
                 max_slots: int = _MAX_DENSE_SLOTS) -> None:
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.max_slots = int(max_slots)
        self.projection = nn.Linear(self.max_slots * self.slot_dim, fusion_dim, bias=False)
        self.norm = nn.RMSNorm(fusion_dim)

    def forward(self, parts: list[Tensor]) -> Tensor:
        padded = [torch.zeros_like(parts[0]) for _ in range(self.max_slots - len(parts))]
        fused = torch.cat([*parts, *padded], dim=-1)
        return self.norm(self.projection(fused))


class SharedDenseMLP(nn.Module):
    """共享的 gated/SiLU 融合 MLP：512 → 2*512 → 256。"""

    def __init__(self, fusion_dim: int, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(fusion_dim, eps=eps)
        self.gate = nn.Linear(fusion_dim, 2 * fusion_dim, bias=False)
        self.down = nn.Linear(fusion_dim, d_model, bias=False)

    def forward(self, values: Tensor) -> Tensor:
        gate, value = self.gate(self.norm(values)).chunk(2, dim=-1)
        return self.down(torch.nn.functional.silu(gate) * value)


class StateTokenEmbedding(nn.Module):
    """V18 状态快照 token 嵌入：segment/kind/separator + 共享槽位融合。"""

    def __init__(self, d_model: int, *, dense_slot_dim: int = DENSE_SLOT_DIM,
                 dense_fusion_dim: int = DENSE_FUSION_DIM, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.dense_slot_dim = int(dense_slot_dim)
        self.dense_fusion_dim = int(dense_fusion_dim)
        self.segment = nn.Embedding(6, self.d_model)
        self.kind = nn.Embedding(128, self.d_model)
        self.separator = nn.Embedding(12, self.d_model)
        for table in (self.segment, self.kind, self.separator):
            nn.init.normal_(table.weight, std=1.0 / math.sqrt(self.d_model))
        with torch.no_grad():
            self.segment.weight[0].zero_()
            self.kind.weight[0].zero_()
            self.separator.weight[0].zero_()
        self.shared_mlp = SharedDenseMLP(dense_fusion_dim, self.d_model, eps=eps)
        self.dense_input = SharedDenseProjection(dense_fusion_dim, dense_slot_dim)
        self.simple_input = SharedSimpleProjection(self.d_model, SIMPLE_SLOT_DIM)
        # 每个类别独立的槽位 embedding 表（子模块注册以保证参数/state_dict 完整）。
        self.simple_tables = nn.ModuleDict({})
        self.dense_tables = nn.ModuleDict({})
        self.numeric_proj = nn.ModuleDict({})
        self.slot_ids = nn.ModuleDict({})
        for kind, schema in sorted(CATEGORY_SCHEMAS.items()):
            self._register_category(schema)

    def _register_category(self, schema: CategorySchema) -> None:
        key = str(schema.kind)
        tables = nn.ModuleList(
            nn.Embedding(field.cardinality, self.dense_slot_dim if schema.cls == "DENSE" else SIMPLE_SLOT_DIM, padding_idx=0)
            for field in schema.discrete
        )
        for table in tables:
            nn.init.normal_(table.weight, std=1.0 / math.sqrt(table.embedding_dim))
            if table.padding_idx is not None:
                with torch.no_grad():
                    table.weight[table.padding_idx].zero_()
        if schema.cls == "DENSE":
            self.dense_tables[key] = tables
            if schema.numeric:
                self.numeric_proj[key] = nn.ModuleList(
                    nn.Linear(1, self.dense_slot_dim, bias=False) for _field in schema.numeric
                )
            if schema.slot_count:
                self.slot_ids[key] = nn.Embedding(schema.slot_count + 1, self.dense_slot_dim, padding_idx=0)
                nn.init.normal_(self.slot_ids[key].weight, std=1.0 / math.sqrt(self.dense_slot_dim))
                with torch.no_grad():
                    self.slot_ids[key].weight[0].zero_()
        else:
            self.simple_tables[key] = tables

    def _dense_parts(self, key: str, schema: CategorySchema, factors: Tensor, numeric: Tensor) -> list[Tensor]:
        """``factors`` 为 [M, W] 行；每个槽位向量 [M, dense_slot_dim]。"""
        tables = self.dense_tables[key]
        parts = [
            table(factors[..., 2 + index].clamp(0, table.num_embeddings - 1))
            for index, table in enumerate(tables)
        ]
        if schema.numeric:
            proj = self.numeric_proj[key]
            parts.extend([
                proj[index](_clip_normalize(numeric[..., index], field.scale).unsqueeze(-1))
                for index, field in enumerate(schema.numeric)
            ])
        if schema.slot_count:
            slot_fields = parts[-4 * schema.slot_count:]
            scalar = parts[: len(parts) - 4 * schema.slot_count]
            valid_length = factors[..., 2].long()
            slot_ids_embedding = self.slot_ids[key]
            slot_parts = []
            slot_ids = torch.arange(1, schema.slot_count + 1, device=factors.device)
            valid_mask = (slot_ids[None, :] <= valid_length[:, None]).float()
            for slot_index in range(schema.slot_count):
                mask = valid_mask[:, slot_index:slot_index + 1]
                slot_parts.extend([
                    slot_fields[4 * slot_index + 0] * mask,
                    slot_fields[4 * slot_index + 1] * mask,
                    slot_fields[4 * slot_index + 2] * mask,
                    slot_fields[4 * slot_index + 3] * mask,
                    slot_ids_embedding(torch.full(
                        (factors.shape[0],), slot_index + 1, device=factors.device,
                    )) * mask,
                ])
            parts = [*scalar, *slot_parts]
        return parts

    def _simple_parts(self, key: str, factors: Tensor) -> list[Tensor]:
        tables = self.simple_tables[key]
        return [
            table(factors[..., 2 + index].clamp(0, table.num_embeddings - 1))
            for index, table in enumerate(tables)
        ]

    def forward(self, factors: Tensor, numeric: Tensor, valid: Tensor | None = None) -> Tensor:
        batch, tokens, _width = factors.shape
        segment = factors[..., 0].long()
        kind = factors[..., 1].long()
        base = self.segment(segment.clamp(0, 5)) + self.kind(kind.clamp(0, 127))
        # 分隔符：额外类别专属 embedding。
        flat_kind = kind.reshape(-1)
        sep_positions = torch.nonzero(
            torch.tensor([is_separator_kind(int(value)) for value in flat_kind.tolist()], device=kind.device),
            as_tuple=False,
        ).squeeze(-1)
        if sep_positions.numel():
            sep_ids = torch.tensor(
                [separator_id_of_kind(int(flat_kind[index].item())) for index in sep_positions.tolist()],
                device=kind.device,
            )
            base_flat = base.reshape(-1, self.d_model)
            base_flat[sep_positions] = base_flat[sep_positions] + self.separator(sep_ids)
            base = base_flat.reshape(batch, tokens, -1)
        content = base
        flat_factors = factors.reshape(batch * tokens, -1)
        flat_numeric = numeric.reshape(batch * tokens, -1)
        flat_kind = flat_factors[:, 1]
        content_flat = content.reshape(batch * tokens, -1)
        for kind_value, schema in CATEGORY_SCHEMAS.items():
            idx = torch.nonzero(flat_kind.eq(kind_value), as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            if schema.cls == "DENSE":
                parts = self._dense_parts(str(kind_value), schema, flat_factors[idx], flat_numeric[idx])
                embedded = self.shared_mlp(self.dense_input(parts))
            elif schema.cls == "SIMPLE":
                if schema.discrete:
                    embedded = self.simple_input(self._simple_parts(str(kind_value), flat_factors[idx]))
                else:
                    # BOS：无字段类别，直接使用 kind/segment 基础向量。
                    embedded = content_flat[idx]
            else:  # pragma: no cover
                continue
            content_flat[idx] = content_flat[idx] + embedded.to(content_flat.dtype)
        content = content_flat.reshape(batch, tokens, -1)
        if valid is not None:
            content = content * valid[..., None].to(content.dtype)
        return content
