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

import numpy as np
import torch
from torch import Tensor, nn

from .encoding_protocol import (
    CATEGORY_SCHEMAS,
    CategorySchema,
)

SIMPLE_SLOT_DIM = 16
DENSE_SLOT_DIM = 32
DENSE_FUSION_DIM = 512

# 统一 concat 宽度：取各类别槽位向量数最大值（TABLE=29，RIVER_SUMMARY=31）。
# 每个 summary 槽按 4 字段 concat + slot_id 共 5 个槽位向量（B6 修复）。
_MAX_DENSE_SLOTS = max(
    len(schema.discrete) - 4 * schema.slot_count
    + (5 if schema.slot_count else 0) * schema.slot_count + len(schema.numeric)
    for schema in CATEGORY_SCHEMAS.values() if schema.cls == "DENSE"
)
_MAX_SIMPLE_SLOTS = max(
    (len(schema.discrete) + len(schema.numeric) for schema in CATEGORY_SCHEMAS.values() if schema.cls == "SIMPLE"),
    default=1,
)


def _clip_normalize(values: Tensor, scale: float) -> Tensor:
    return (values.float() / scale).clamp(-1.0, 1.0)


def compute_kind_row_plan(actor_factors: np.ndarray) -> dict[int, np.ndarray]:
    """在 host 侧按静态类别键表计算各 kind 的平坦行下标(C2)。

    返回 ``{kind_value: 行下标数组(升序)}``,下标为 padded 批展平后的行号。
    升序与 forward 内 stable argsort 的段内顺序逐位一致;仅包含
    ``CATEGORY_SCHEMAS`` 注册的类别,未注册 kind 的行不进入任何段(与旧实现
    的 continue 语义一致)。纯 numpy 向量化,无任何 GPU 同步,供 collate/
    推理 host 装配时顺带计算后传入 forward 的 ``kind_row_plan``。
    """
    kinds = np.asarray(actor_factors)[..., 1].reshape(-1)
    plan: dict[int, np.ndarray] = {}
    for kind_value in CATEGORY_SCHEMAS:
        indices = np.flatnonzero(kinds == kind_value)
        if indices.size:
            plan[int(kind_value)] = indices
    return plan


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
        for _kind, schema in sorted(CATEGORY_SCHEMAS.items()):
            self._register_category(schema)
        # C2:静态类别元数据表(键序固定)。配合 host 侧 kind_row_plan,
        # forward 免去 argsort/边界计算/3 次 tolist 同步与按动态 kind 值的
        # Python 分派(键序与表引用在编译期已知,init 一次预计算)。
        self._static_categories = tuple(
            (int(kind), str(kind), schema.cls, schema)
            for kind, schema in sorted(CATEGORY_SCHEMAS.items())
        )

    def _register_category(self, schema: CategorySchema) -> None:
        key = str(schema.kind)
        tables = nn.ModuleList(
            nn.Embedding(
                field.cardinality,
                self.dense_slot_dim if schema.cls == "DENSE" else SIMPLE_SLOT_DIM,
                padding_idx=0,
            )
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

    def forward(
        self,
        factors: Tensor,
        numeric: Tensor,
        valid: Tensor | None = None,
        *,
        kind_row_plan: dict[int, np.ndarray] | None = None,
    ) -> Tensor:
        batch, tokens, _width = factors.shape
        segment = factors[..., 0].long()
        kind = factors[..., 1].long()
        base = self.segment(segment.clamp(0, 5)) + self.kind(kind.clamp(0, 127))
        base_flat = base.reshape(-1, self.d_model)
        # 分隔符：类别专属 embedding（纯向量化；kind=100+separator_id ∈ [101,111]）。
        # C2:算术掩蔽取代布尔索引读写(布尔变长索引必同步);非分隔行查到
        # separator 的 0 号 padding 行(严格零)再乘 0 掩蔽,x+0.0==x 逐位等价。
        flat_kind = kind.reshape(-1)
        sep_mask = ((flat_kind >= 101) & (flat_kind <= 111)).to(base_flat.dtype)
        sep_ids = (flat_kind - 100).clamp(0, self.separator.num_embeddings - 1)
        base_flat = base_flat + self.separator(sep_ids) * sep_mask[:, None]
        content_flat = base_flat
        flat_factors = factors.reshape(batch * tokens, -1)
        flat_numeric = numeric.reshape(batch * tokens, -1)
        if kind_row_plan is not None:
            # C2 host 计划路径:静态类别键序迭代 + host 行表(升序,与旧路径
            # 段内顺序一致),全程无 argsort/tolist/item 同步;行内容与累加
            # 语义逐位等价(单测 torch.equal 断言)。行表先合并为单次 pinned
            # 异步 H2D(逐类别 pageable 拷贝每次都是一个同步点),再按 host
            # 偏移交量切片取视图。
            arrays = [
                kind_row_plan[kind_value]
                for kind_value, _key, _cls, _schema in self._static_categories
                if kind_row_plan.get(kind_value) is not None
                and kind_row_plan[kind_value].size
            ]
            idx_by_kind: list[Tensor] = []
            if arrays:
                total = int(sum(arr.size for arr in arrays))
                pinned = torch.empty(
                    total, dtype=torch.long,
                    pin_memory=(kind.device.type == "cuda"),
                )
                pinned.copy_(torch.from_numpy(np.concatenate(arrays)))
                all_idx = pinned.to(kind.device, non_blocking=True)
                cursor = 0
                for arr in arrays:
                    size = int(arr.size)
                    idx_by_kind.append(all_idx[cursor:cursor + size])
                    cursor += size
            cursor = 0
            for kind_value, key, cls, schema in self._static_categories:
                indices = kind_row_plan.get(kind_value)
                if indices is None or indices.size == 0:
                    continue
                idx = idx_by_kind[cursor]
                cursor += 1
                row_factors = flat_factors[idx]
                row_numeric = flat_numeric[idx]
                if cls == "DENSE":
                    parts = self._dense_parts(key, schema, row_factors, row_numeric)
                    embedded = self.shared_mlp(self.dense_input(parts))
                elif cls == "SIMPLE":
                    if schema.discrete:
                        embedded = self.simple_input(self._simple_parts(key, row_factors))
                    else:
                        # BOS：无字段类别，直接使用 kind/segment 基础向量。
                        embedded = content_flat[idx]
                else:  # pragma: no cover
                    continue
                content_flat[idx] = content_flat[idx] + embedded.to(content_flat.dtype)
        else:
            # 一次稳定排序得到按 kind 连续区段，替代逐类别 torch.nonzero
            # （旧实现每类一次 GPU 操作 + 同步）。
            order = torch.argsort(flat_kind, stable=True)
            sorted_kind = flat_kind[order]
            change = sorted_kind[1:] != sorted_kind[:-1]
            starts = torch.cat([
                torch.zeros(1, dtype=torch.long, device=kind.device),
                torch.nonzero(change, as_tuple=False).squeeze(-1) + 1,
            ])
            ends = torch.cat([
                starts[1:],
                torch.tensor([sorted_kind.numel()], dtype=torch.long, device=kind.device),
            ])
            starts_list = starts.tolist()
            ends_list = ends.tolist()
            kind_values = sorted_kind[starts].tolist()
            for offset, kind_value in enumerate(kind_values):
                schema = CATEGORY_SCHEMAS.get(kind_value)
                if schema is None:
                    continue
                start, end = starts_list[offset], ends_list[offset]
                if start >= end:
                    continue
                idx = order[start:end]
                row_factors = flat_factors[idx]
                row_numeric = flat_numeric[idx]
                if schema.cls == "DENSE":
                    parts = self._dense_parts(str(kind_value), schema, row_factors, row_numeric)
                    embedded = self.shared_mlp(self.dense_input(parts))
                elif schema.cls == "SIMPLE":
                    if schema.discrete:
                        embedded = self.simple_input(self._simple_parts(str(kind_value), row_factors))
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
