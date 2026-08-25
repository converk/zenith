"""V18 单协议 GQA Actor-Critic。

Actor 输入固定为 Objective Facts + Atomic Snapshot + 每动作一对
Offense/Defense Query,动作对由结构化注意力隔离。Critic 仅在共享公开表示后
读取三家真实手牌与未来五张牌。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .encoding_protocol import (
    DEFENSE_SLOT_ORDER,
    OFFENSE_SLOT_ORDER,
    ACTION_TYPE_CARDINALITY,
    QUERY_ROW_WIDTH,
    SLOT_CARDINALITIES,
    SNAPSHOT_FACTOR_CARDINALITIES,
    SNAPSHOT_FACTOR_WIDTH,
    SNAPSHOT_FIELD_COUNT,
    SNAPSHOT_NUMERIC_WIDTH,
)
from .schema import NUM_ACTIONS

TOKEN_WIDTH = 10
NUMERIC_WIDTH = 8
TOKEN_CARDINALITIES = (8, 32, 256, 8, 8, 16, 4, 16, 256, 4)


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 4
    shared_layers: int = 3
    critic_layers: int = 2
    d_model: int = 256
    query_heads: int = 16
    kv_heads: int = 4
    head_dim: int = 16
    ffn_dim: int = 704
    context_tokens: int = 4096
    rope_base: float = 10_000.0
    eps: float = 1e-6
    policy_head_type: str = "isolated_action_query"

    def __post_init__(self) -> None:
        if self.d_model != self.query_heads * self.head_dim:
            raise ValueError("d_model must equal query_heads * head_dim")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.context_tokens < 2:
            raise ValueError("context_tokens must leave room for the query token")
        if not 0 < self.shared_layers < self.layers:
            raise ValueError("shared_layers must be positive and leave at least one actor-only layer")
        if self.critic_layers < 1:
            raise ValueError("critic_layers must be positive")
        if self.policy_head_type != "isolated_action_query":
            raise ValueError("policy_head_type must be isolated_action_query")

    @classmethod
    def preset(cls, size: str) -> "ModelConfig":
        if size != "v18":
            raise ValueError("model size must be 'v18'")
        return cls()

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "ModelConfig":
        """从纯 V18 checkpoint/config 映射恢复精确拓扑。"""
        return cls(**dict(values))


class FactorEmbedding(nn.Module):
    """Summed categorical factors with an optional continuous feature channel."""

    def __init__(self, cardinalities: tuple[int, ...], d_model: int, numeric_dim: int = 0) -> None:
        super().__init__()
        offsets, end = [], 0
        for size in cardinalities:
            if size < 1:
                raise ValueError("factor cardinalities must be positive")
            offsets.append(end)
            end += size - 1
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.table = nn.Embedding(end + 1, d_model, padding_idx=0)
        self.numeric = nn.Linear(numeric_dim, d_model, bias=False) if numeric_dim else None
        self.norm = nn.RMSNorm(d_model)
        nn.init.normal_(self.table.weight, std=1.0 / math.sqrt(d_model))
        with torch.no_grad():
            self.table.weight[0].zero_()

    def forward(self, factors: Tensor, numeric: Tensor | None = None) -> Tensor:
        if factors.ndim != 3 or factors.shape[-1] != self.offsets.numel():
            raise ValueError("token_factors must be [batch, tokens, 10]")
        factors = factors.long()
        indices = torch.where(factors.eq(0), 0, factors + self.offsets)
        active = factors.ne(0).sum(-1, keepdim=True).clamp_min(1)
        embedded = self.table(indices).sum(-2) / active.sqrt()
        if self.numeric is not None:
            if numeric is None or numeric.shape != (*factors.shape[:2], self.numeric.in_features):
                raise ValueError(f"numeric features must match token factors with width {self.numeric.in_features}")
            embedded = embedded + self.numeric(numeric.float())
        return self.norm(embedded)


class QueryEmbedding(nn.Module):
    """V18 动作 Query 的 metadata 与 answer slot 聚合嵌入。

    ``E_q = E_action + E_queryType + Σ_i E_{type,i}(answer_i)``,再经
    LayerNorm/Projection 到 d_model。Offense 与 Defense 使用各自独立、结构对称
    的 10 组 slot 嵌入;两个分支共用同一个 action/queryType 嵌入与投影层。
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.action = nn.Embedding(NUM_ACTIONS, d_model)
        self.query_type = nn.Embedding(3, d_model)
        self.action_type = nn.Embedding(ACTION_TYPE_CARDINALITY, d_model)
        self.primary_tile = nn.Embedding(35, d_model)
        self.source_seat = nn.Embedding(4, d_model)
        self.offense_slots = nn.ModuleList(
            nn.Embedding(SLOT_CARDINALITIES[slot], d_model) for slot in OFFENSE_SLOT_ORDER
        )
        self.defense_slots = nn.ModuleList(
            nn.Embedding(SLOT_CARDINALITIES[slot], d_model) for slot in DEFENSE_SLOT_ORDER
        )
        self.norm = nn.RMSNorm(d_model)
        self.projection = nn.Linear(d_model, d_model, bias=False)
        for embedding in (
            self.action,
            self.query_type,
            self.action_type,
            self.primary_tile,
            self.source_seat,
            *self.offense_slots,
            *self.defense_slots,
        ):
            nn.init.normal_(embedding.weight, std=1.0 / math.sqrt(d_model))

    def forward(self, rows: Tensor) -> Tensor:
        """``rows`` 为 [B, Q, QUERY_ROW_WIDTH] 的整型编码行。"""
        if rows.ndim != 3 or rows.shape[-1] != QUERY_ROW_WIDTH:
            raise ValueError(f"query rows must be [batch, queries, {QUERY_ROW_WIDTH}]")
        rows = rows.long()
        query_type = rows[..., 0]
        action_ids = rows[..., 1]
        action_types = rows[..., 2]
        primary_tiles = rows[..., 3]
        source_seats = rows[..., 4]
        answers = rows[..., 5:]
        if torch.any(action_ids < 0) or torch.any(action_ids >= NUM_ACTIONS):
            raise ValueError("query action_id is outside the fixed action space")
        if torch.any(query_type < 0) or torch.any(query_type > 2):
            raise ValueError("query_type must be 0..2")
        if torch.any(action_types < 0) or torch.any(action_types >= ACTION_TYPE_CARDINALITY):
            raise ValueError("query action_type is outside its domain")
        if torch.any(primary_tiles < 0) or torch.any(primary_tiles >= 35):
            raise ValueError("query primary_tile is outside its domain")
        if torch.any(source_seats < 0) or torch.any(source_seats >= 4):
            raise ValueError("query source_seat is outside its domain")
        embedded = (
            self.action(action_ids)
            + self.query_type(query_type)
            + self.action_type(action_types)
            + self.primary_tile(primary_tiles)
            + self.source_seat(source_seats)
        )
        offense = torch.zeros_like(embedded)
        defense = torch.zeros_like(embedded)
        for index, embedding in enumerate(self.offense_slots):
            maximum = SLOT_CARDINALITIES[OFFENSE_SLOT_ORDER[index]] - 1
            # 非 offense 行会在下方被 where 丢弃,先按本 slot 基数截断避免
            # 越界;真实越界数据由编码期审计负责拒绝。
            offense = offense + embedding(answers[..., index].clamp(0, maximum))
        for index, embedding in enumerate(self.defense_slots):
            maximum = SLOT_CARDINALITIES[DEFENSE_SLOT_ORDER[index]] - 1
            defense = defense + embedding(answers[..., index].clamp(0, maximum))
        is_offense = query_type.eq(1).unsqueeze(-1)
        embedded = embedded + torch.where(is_offense, offense, defense)
        return self.projection(self.norm(embedded))


class SnapshotEmbedding(nn.Module):
    """固定原子字段共用的 factor/numeric 嵌入。"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.embedding = FactorEmbedding(
            SNAPSHOT_FACTOR_CARDINALITIES,
            d_model,
            SNAPSHOT_NUMERIC_WIDTH,
        )

    def forward(self, factors: Tensor, numeric: Tensor) -> Tensor:
        if factors.ndim != 3 or factors.shape[1:] != (
            SNAPSHOT_FIELD_COUNT,
            SNAPSHOT_FACTOR_WIDTH,
        ):
            raise ValueError(
                f"snapshot_factors must be [batch,{SNAPSHOT_FIELD_COUNT},{SNAPSHOT_FACTOR_WIDTH}]"
            )
        if numeric.shape != (
            factors.shape[0],
            SNAPSHOT_FIELD_COUNT,
            SNAPSHOT_NUMERIC_WIDTH,
        ):
            raise ValueError(
                f"snapshot_numeric must be [batch,{SNAPSHOT_FIELD_COUNT},{SNAPSHOT_NUMERIC_WIDTH}]"
            )
        return self.embedding(factors, numeric)


def _rope_values(position_ids: Tensor, head_dim: int, dtype: torch.dtype, base: float) -> tuple[Tensor, Tensor]:
    positions = position_ids.to(dtype=torch.float32)
    frequencies = torch.exp(torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32) * (-math.log(base) / head_dim))
    angle = positions[..., None] * frequencies
    # q/k are [B,H,T,D].  The head axis must broadcast independently.
    return angle.cos().to(dtype).unsqueeze(1), angle.sin().to(dtype).unsqueeze(1)


def _rope(x: Tensor, values: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = values
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def _attention_layout(lengths: Tensor, tokens: int) -> tuple[Tensor, Tensor]:
    causal = torch.ones(tokens, tokens, dtype=torch.bool, device=lengths.device).tril()
    valid = torch.arange(tokens, device=lengths.device)[None] < lengths[:, None]
    valid_query = valid[:, None, :, None]
    mask = causal[None, None] & valid[:, None, None, :]
    # Padded rows need one finite key to avoid softmax NaNs. Blocks zero them
    # immediately afterwards, so this artificial key cannot affect real rows.
    first_key = torch.zeros_like(mask)
    first_key[..., 0] = True
    return (mask & valid_query) | (first_key & ~valid_query), valid


def _isolated_action_layout(
    public_lengths: Tensor,
    pair_counts: Tensor,
    tokens: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """构造公共 causal、动作对内双向、动作对之间隔离的注意力布局。"""
    device = public_lengths.device
    batch = public_lengths.shape[0]
    positions = torch.arange(tokens, device=device)[None, :]
    total_lengths = public_lengths + 2 * pair_counts
    valid = positions < total_lengths[:, None]
    query_pos = positions[:, :, None]
    key_pos = positions[:, None, :]
    public_query = query_pos < public_lengths[:, None, None]
    public_key = key_pos < public_lengths[:, None, None]
    public_causal = public_query & public_key & (key_pos <= query_pos)
    query_local = query_pos - public_lengths[:, None, None]
    key_local = key_pos - public_lengths[:, None, None]
    query_pair = torch.div(query_local.clamp_min(0), 2, rounding_mode="floor")
    key_pair = torch.div(key_local.clamp_min(0), 2, rounding_mode="floor")
    action_query = ~public_query
    own_pair = action_query & ~public_key & query_pair.eq(key_pair)
    action_to_public = action_query & public_key
    mask = (public_causal | action_to_public | own_pair)[:, None]
    valid_query = valid[:, None, :, None]
    valid_key = valid[:, None, None, :]
    mask = mask & valid_query & valid_key
    first_key = torch.zeros_like(mask)
    first_key[..., 0] = True
    mask = mask | (first_key & ~valid_query)

    position_ids = positions.expand(batch, -1).clone()
    query_slots = (positions - public_lengths[:, None]).clamp_min(0) % 2
    local_positions = public_lengths[:, None] + query_slots
    position_ids = torch.where(positions >= public_lengths[:, None], local_positions, position_ids)
    return mask, valid, position_ids


class CausalGQA(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.qh, self.kvh, self.head_dim = config.query_heads, config.kv_heads, config.head_dim
        self.qkv = nn.Linear(config.d_model, (self.qh + 2 * self.kvh) * self.head_dim, bias=False)
        self.out = nn.Linear(self.qh * self.head_dim, config.d_model, bias=False)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], attention_mask: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        shape = lambda value, heads: value.view(batch, tokens, heads, self.head_dim).transpose(1, 2)
        q_raw, k_raw, v_raw = self.qkv(x).split((self.qh * self.head_dim, self.kvh * self.head_dim, self.kvh * self.head_dim), dim=-1)
        q, k, v = _rope(shape(q_raw, self.qh), rope), _rope(shape(k_raw, self.kvh), rope), shape(v_raw, self.kvh)
        repeat = self.qh // self.kvh
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        return self.out(value.transpose(1, 2).reshape(batch, tokens, self.qh * self.head_dim))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(config.d_model, eps=config.eps), nn.RMSNorm(config.d_model, eps=config.eps)
        self.attention = CausalGQA(config)
        self.gate = nn.Linear(config.d_model, 2 * config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], attention_mask: Tensor, valid: Tensor) -> Tensor:
        x = x + self.attention(self.n1(x), rope, attention_mask)
        gate, value = self.gate(self.n2(x)).chunk(2, dim=-1)
        x = x + self.down(F.silu(gate) * value)
        return torch.where(valid[..., None], x, torch.zeros((), dtype=x.dtype, device=x.device))


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig, *, layers: int | None = None, final_norm: bool = True) -> None:
        super().__init__()
        self.context_tokens, self.rope_base = config.context_tokens, config.rope_base
        block_count = config.layers if layers is None else int(layers)
        if block_count < 1:
            raise ValueError("decoder must contain at least one block")
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(block_count))
        self.norm: nn.Module = nn.RMSNorm(config.d_model, eps=config.eps) if final_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        lengths: Tensor,
        *,
        attention_mask: Tensor | None = None,
        valid: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        tokens = x.shape[1]
        if tokens > self.context_tokens:
            raise ValueError(f"context overflow: {tokens} > {self.context_tokens}")
        if attention_mask is None or valid is None:
            attention_mask, valid = _attention_layout(lengths, tokens)
        if position_ids is None:
            position_ids = torch.arange(tokens, device=x.device)[None].expand(x.shape[0], -1)
        rope = _rope_values(
            position_ids, x.shape[-1] // self.blocks[0].attention.qh, x.dtype, self.rope_base
        )
        for block in self.blocks:
            x = block(x, rope, attention_mask, valid)
        return self.norm(x)


class KyokuTransformerActorCritic(nn.Module):
    """Shared-public actor with a centralized critic-only private branch."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig.preset("v18")
        self.token_embedding = FactorEmbedding(TOKEN_CARDINALITIES, self.config.d_model, NUMERIC_WIDTH)
        self.critic_embedding = FactorEmbedding(TOKEN_CARDINALITIES, self.config.d_model)
        self.value_query = nn.Parameter(torch.empty(self.config.d_model))
        # 三层 shared 只处理公共前缀,Actor-only 层再读取隔离的动作对。
        self.public_backbone = Decoder(self.config, layers=self.config.shared_layers, final_norm=False)
        self.actor_backbone = Decoder(self.config, layers=self.config.layers - self.config.shared_layers)
        self.critic_backbone = Decoder(self.config, layers=self.config.critic_layers)
        self.snapshot_embeddings = SnapshotEmbedding(self.config.d_model)
        self.query_embedding = QueryEmbedding(self.config.d_model)
        self.action_fusion = nn.Sequential(
            nn.Linear(2 * self.config.d_model, self.config.d_model),
            nn.SiLU(),
        )
        self.policy_mlp = nn.Sequential(
            nn.RMSNorm(self.config.d_model, eps=self.config.eps),
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.SiLU(),
            nn.Linear(self.config.d_model, 1),
        )
        self.value_head = nn.Linear(self.config.d_model, 1)
        nn.init.normal_(self.value_query, std=self.config.d_model ** -0.5)

    def forward_actor(
        self,
        *,
        history_factors: Tensor,
        history_numeric: Tensor,
        history_lengths: Tensor,
        snapshot_factors: Tensor,
        snapshot_numeric: Tensor,
        snapshot_lengths: Tensor,
        query_rows: Tensor,
        query_action_ids: Tensor,
        query_pair_counts: Tensor,
        legal_mask: Tensor,
    ) -> dict[str, Tensor]:
        """只接受 Actor 可见输入并返回合法动作 logits。"""
        return self.forward(
            history_factors=history_factors,
            history_numeric=history_numeric,
            history_lengths=history_lengths,
            snapshot_factors=snapshot_factors,
            snapshot_numeric=snapshot_numeric,
            snapshot_lengths=snapshot_lengths,
            query_rows=query_rows,
            query_action_ids=query_action_ids,
            query_pair_counts=query_pair_counts,
            legal_mask=legal_mask,
            policy_only=True,
        )

    def forward(
        self,
        history_factors: Tensor,
        history_numeric: Tensor,
        history_lengths: Tensor,
        snapshot_factors: Tensor,
        snapshot_numeric: Tensor,
        snapshot_lengths: Tensor,
        query_rows: Tensor,
        query_action_ids: Tensor,
        query_pair_counts: Tensor,
        legal_mask: Tensor,
        *,
        critic_factors: Tensor | None = None,
        critic_lengths: Tensor | None = None,
        detach_critic_public: bool = False,
        critic_public_grad_scale: float = 1.0,
        policy_only: bool = False,
    ) -> dict[str, Tensor]:
        """V18 前向:Objective Facts + Atomic Snapshot + 隔离动作对。

        Query token 加入主序列末尾(每动作 Offense/Defense 相邻),策略头对每一
        对做对称融合;Critic 公共前缀只含 Objective Facts + Snapshot(不追加
        Action Query Token),其后拼接特权输入(三家手牌 + 后 5 牌山)与 Value
        Query。
        """
        if self.config.policy_head_type != "isolated_action_query":
            raise ValueError("V18 forward requires isolated_action_query")
        if history_factors.ndim != 3 or history_factors.shape[-1] != TOKEN_WIDTH:
            raise ValueError(f"history_factors must be [batch, tokens, {TOKEN_WIDTH}]")
        if history_numeric.shape != (*history_factors.shape[:2], NUMERIC_WIDTH):
            raise ValueError(f"history_numeric must be [batch, tokens, {NUMERIC_WIDTH}]")
        if snapshot_factors.shape[1:] != (SNAPSHOT_FIELD_COUNT, SNAPSHOT_FACTOR_WIDTH):
            raise ValueError(
                f"snapshot_factors must be [batch,{SNAPSHOT_FIELD_COUNT},{SNAPSHOT_FACTOR_WIDTH}]"
            )
        if snapshot_numeric.shape != (
            snapshot_factors.shape[0], SNAPSHOT_FIELD_COUNT, SNAPSHOT_NUMERIC_WIDTH,
        ):
            raise ValueError(
                f"snapshot_numeric must be [batch,{SNAPSHOT_FIELD_COUNT},{SNAPSHOT_NUMERIC_WIDTH}]"
            )
        if query_rows.ndim != 3 or query_rows.shape[-1] != QUERY_ROW_WIDTH:
            raise ValueError(f"query_rows must be [batch, queries, {QUERY_ROW_WIDTH}]")
        if query_action_ids.ndim != 2 or query_rows.shape[1] != 2 * query_action_ids.shape[1]:
            raise ValueError("query rows must be exactly one offense/defense pair per action")
        batch, history_capacity, _ = history_factors.shape
        device = history_factors.device

        def lengths(value: Tensor, capacity: int, label: str) -> Tensor:
            result = value.to(device=device, dtype=torch.long)
            if result.shape != (batch,):
                raise ValueError(f"{label} must have one entry per batch row")
            if torch.any(result < 0) or torch.any(result > capacity):
                raise ValueError(f"{label} exceed supplied rows")
            return result

        history_lengths = lengths(history_lengths, history_capacity, "history_lengths")
        snapshot_lengths = lengths(
            snapshot_lengths, SNAPSHOT_FIELD_COUNT, "snapshot_lengths",
        )
        if torch.any(snapshot_lengths != SNAPSHOT_FIELD_COUNT):
            raise ValueError(f"snapshot_lengths must all equal {SNAPSHOT_FIELD_COUNT}")
        pair_counts = lengths(
            query_pair_counts, query_action_ids.shape[1], "query_pair_counts",
        )
        if legal_mask.shape != (batch, NUM_ACTIONS):
            raise ValueError(f"legal_mask must be [batch, {NUM_ACTIONS}]")
        if torch.any(query_action_ids < 0) or torch.any(query_action_ids >= NUM_ACTIONS):
            raise ValueError("query_action_ids are outside the fixed action space")
        action_capacity = query_action_ids.shape[1]
        pair_positions = torch.arange(action_capacity, device=device)[None].expand(batch, -1)
        pair_valid = pair_positions < pair_counts[:, None]
        pair_rows = query_rows.view(batch, action_capacity, 2, QUERY_ROW_WIDTH)
        if torch.any(pair_rows[:, :, 0, 0][pair_valid] != 1) or torch.any(
            pair_rows[:, :, 1, 0][pair_valid] != 2
        ):
            raise ValueError("each action must contain one Offense row followed by one Defense row")
        expected_ids = query_action_ids.to(device=device, dtype=pair_rows.dtype)
        if torch.any(pair_rows[:, :, 0, 1][pair_valid] != expected_ids[pair_valid]) or torch.any(
            pair_rows[:, :, 1, 1][pair_valid] != expected_ids[pair_valid]
        ):
            raise ValueError("query pair action ids disagree with query_action_ids")
        if torch.any(
            pair_rows[:, :, 0, 2:5][pair_valid]
            != pair_rows[:, :, 1, 2:5][pair_valid]
        ):
            raise ValueError("Offense/Defense metadata must match within each action pair")
        represented = torch.zeros_like(legal_mask, dtype=torch.bool, device=device)
        represented_rows = torch.arange(batch, device=device)[:, None].expand_as(query_action_ids)
        represented[
            represented_rows[pair_valid],
            query_action_ids.to(device=device, dtype=torch.long)[pair_valid],
        ] = True
        if torch.any(represented.sum(dim=1) != pair_counts):
            raise ValueError("query action ids must be unique within each sample")
        if torch.any(represented != legal_mask.to(device=device, dtype=torch.bool)):
            raise ValueError("query action ids must be unique and equal the legal action set")
        if torch.any(history_lengths + snapshot_lengths + 2 * pair_counts > self.config.context_tokens):
            raise ValueError("V18 context overflow: history + snapshot + queries exceed context_tokens")

        history = self.token_embedding(history_factors, history_numeric)
        snapshot = self.snapshot_embeddings(snapshot_factors, snapshot_numeric)
        queries = self.query_embedding(query_rows)
        rows = torch.arange(batch, device=device)

        def scatter(target: Tensor, segment: Tensor, start: Tensor, count: Tensor) -> None:
            local = torch.arange(segment.shape[1], device=device)[None, :]
            valid = local < count[:, None]
            destination = start[:, None] + local
            rows_expanded = rows[:, None].expand_as(destination)
            target[rows_expanded[valid], destination[valid]] = segment[valid].to(
                target.dtype
            )

        public_lengths = history_lengths + snapshot_lengths
        public_capacity = int(public_lengths.max().item())
        public_tokens = history.new_zeros((batch, public_capacity, self.config.d_model))
        scatter(
            public_tokens, history,
            torch.zeros(batch, device=device, dtype=torch.long), history_lengths,
        )
        scatter(public_tokens, snapshot, history_lengths, snapshot_lengths)
        attention_mask, valid = _attention_layout(public_lengths, public_capacity)
        position_ids = torch.arange(public_capacity, device=device)[None].expand(batch, -1)
        shared = self.public_backbone(
            public_tokens, public_lengths, attention_mask=attention_mask, valid=valid,
            position_ids=position_ids,
        )

        total_lengths = public_lengths + 2 * pair_counts
        actor_capacity = int(total_lengths.max().item())
        actor_tokens = history.new_zeros((batch, actor_capacity, self.config.d_model))
        scatter(actor_tokens, shared, torch.zeros_like(public_lengths), public_lengths)
        scatter(actor_tokens, queries, public_lengths, 2 * pair_counts)
        actor_mask, actor_valid, actor_positions = _isolated_action_layout(
            public_lengths, pair_counts, actor_capacity,
        )
        actor = self.actor_backbone(
            actor_tokens, total_lengths, attention_mask=actor_mask, valid=actor_valid,
            position_ids=actor_positions,
        )

        # 对称融合策略头:同一动作的 offense/defense 表示 concat 后共享投影。
        rows = torch.arange(batch, device=device)
        pair_offsets = pair_positions
        public_prefix = (history_lengths + snapshot_lengths)[:, None]
        # 打包后序列容量 = 本 batch 最大 total_lengths;padding pair 的位置可能
        # 超出容量,先 clamp,其取值随后被 pair_valid 丢弃。
        offense_positions = torch.clamp(
            public_prefix + 2 * pair_offsets, max=actor_tokens.shape[1] - 1,
        )
        defense_positions = torch.clamp(
            public_prefix + 2 * pair_offsets + 1, max=actor_tokens.shape[1] - 1,
        )
        offense_hidden = actor[rows[:, None], offense_positions]
        defense_hidden = actor[rows[:, None], defense_positions]
        action_hiddens = self.action_fusion(
            torch.cat((offense_hidden, defense_hidden), dim=-1)
        )
        action_logits = self.policy_mlp(action_hiddens).squeeze(-1).float()
        raw = actor.new_zeros((batch, NUM_ACTIONS), dtype=torch.float32)
        masked_logits = torch.where(pair_valid, action_logits, torch.zeros_like(action_logits))
        raw.scatter_add_(1, query_action_ids.to(device=device, dtype=torch.long), masked_logits)
        logits = raw.masked_fill(
            ~legal_mask.to(device=device, dtype=torch.bool), float("-inf"),
        )
        output: dict[str, Tensor] = {
            "raw_policy_logits": raw,
            "policy_logits": logits,
        }
        if policy_only:
            return output

        # Critic:公共前缀打包(不含 query token)+ 特权输入 + Value Query。
        public_capacity = max(int(public_lengths.max().item()), 1)
        packed_public = shared.new_zeros((batch, public_capacity, self.config.d_model))
        public_positions = torch.arange(public_capacity, device=device)[None, :].expand(batch, -1)
        public_valid = public_positions < public_lengths[:, None]
        source_rows = rows[:, None].expand_as(public_positions)[public_valid]
        packed_public[source_rows, public_positions[public_valid]] = shared[source_rows, public_positions[public_valid]]

        if critic_factors is None:
            critic_factors = history_factors.new_zeros((batch, 0, TOKEN_WIDTH))
        if critic_lengths is None:
            critic_lengths = critic_factors.ne(0).any(-1).long().sum(-1)
        critic_factors = critic_factors.to(device=device)
        critic_lengths = lengths(critic_lengths, critic_factors.shape[1], "critic_lengths")
        if critic_factors.ndim != 3 or critic_factors.shape[0] != batch or critic_factors.shape[-1] != TOKEN_WIDTH:
            raise ValueError(f"critic_factors must be [batch, critic_tokens, {TOKEN_WIDTH}]")
        # 每行必须先给出三家真实手牌分段,再给出严格有序的后五张牌山。
        for row_index in range(batch):
            private = critic_factors[row_index, :critic_lengths[row_index]]
            segments = private[:, 0]
            if torch.any((segments != 4) & (segments != 5)):
                raise ValueError("critic input contains a public/query/unknown segment")
            future = private[segments == 5]
            if future.shape[0] != 5 or torch.any(
                future[:, 3].to(dtype=torch.long)
                != torch.arange(1, 6, device=device)
            ):
                raise ValueError("critic input must end with ordered future-wall positions 1..5")
            private_hands = private[segments == 4]
            for relative in (2, 3, 4):
                if not torch.any(private_hands[:, 3] == relative):
                    raise ValueError("critic input must contain all three opponent hands")
            if private_hands.shape[0] and torch.any(
                segments[:private_hands.shape[0]] != 4
            ):
                raise ValueError("critic opponent hands must precede future-wall tokens")
        critic_sequence_lengths = public_lengths + critic_lengths + 1
        if torch.any(critic_sequence_lengths > self.config.context_tokens):
            raise ValueError(
                "critic context overflow: public tokens + critic tokens + value query exceed context_tokens"
            )
        critic_private = self.critic_embedding(critic_factors)
        critic_capacity = int(critic_sequence_lengths.max().item())
        critic_sequence = critic_private.new_zeros((batch, critic_capacity, self.config.d_model))
        public_grad_scale = 0.0 if detach_critic_public else float(critic_public_grad_scale)
        if not 0.0 <= public_grad_scale <= 1.0:
            raise ValueError("critic_public_grad_scale must be in [0, 1]")
        if public_grad_scale == 0.0:
            critic_public = packed_public.detach()
        elif public_grad_scale == 1.0:
            critic_public = packed_public
        else:
            detached_public = packed_public.detach()
            critic_public = detached_public + public_grad_scale * (packed_public - detached_public)
        critic_sequence[source_rows, public_positions[public_valid]] = critic_public[public_valid]
        private_positions = public_lengths[:, None] + torch.arange(
            critic_private.shape[1], device=device,
        )[None, :]
        private_valid = torch.arange(critic_private.shape[1], device=device)[None, :] < critic_lengths[:, None]
        private_rows = rows[:, None].expand_as(private_positions)[private_valid]
        critic_sequence[private_rows, private_positions[private_valid]] = critic_private[private_valid]
        value_indices = public_lengths + critic_lengths
        critic_sequence[rows, value_indices] = self.value_query
        critic_hidden = self.critic_backbone(critic_sequence, critic_sequence_lengths)[rows, value_indices]
        assert self.value_head is not None
        output["value"] = self.value_head(critic_hidden).squeeze(-1).float()
        output["critic_hidden"] = critic_hidden
        output["action_hiddens"] = action_hiddens
        return output
