"""V16 单协议 GQA 演员-评论家。

Actor 输入固定为 Objective Facts + Compact Snapshot + 每动作一对
Offense/Defense Query,输出固定 241 维动作空间。旧 isolated-action-query
模型头与历史兼容分支已移除。
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
    QUERY_ROW_WIDTH,
    SLOT_CARDINALITIES,
    SNAPSHOT_CAT_WIDTH,
    SNAPSHOT_KIND_BASE,
    SNAPSHOT_KIND_DORA,
    SNAPSHOT_KIND_SCORE,
    SNAPSHOT_KIND_SUMMARY,
    SNAPSHOT_NUM_WIDTH,
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
    d_model: int = 192
    query_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 24
    ffn_dim: int = 576
    context_tokens: int = 4096
    rope_base: float = 10_000.0
    eps: float = 1e-6
    policy_head_type: str = "symmetric_action_query"

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
        if self.policy_head_type != "symmetric_action_query":
            raise ValueError("policy_head_type must be symmetric_action_query")

    @classmethod
    def preset(cls, size: str) -> "ModelConfig":
        configs = {
            "mid": cls(),
            "large": cls(
                d_model=384, query_heads=12, kv_heads=3, head_dim=32,
                ffn_dim=1152,
            ),
            # V16-small:版本命名保持 V16,只调整隐藏层容量,输入/输出协议不变。
            "v16": cls(
                layers=4,
                shared_layers=3,
                critic_layers=2,
                d_model=192,
                query_heads=12,
                kv_heads=3,
                head_dim=16,
                ffn_dim=576,
                policy_head_type="symmetric_action_query",
            ),
        }
        try:
            return configs[size]
        except KeyError as exc:
            raise ValueError("model size must be 'mid', 'large' or 'v16'") from exc

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "ModelConfig":
        """从 checkpoint/config 映射恢复现行 V16 拓扑。"""
        filtered = dict(values)
        for key in ("offense_" + "fusion", "critic_" + "head_type"):
            filtered.pop(key, None)
        return cls(**filtered)


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
    """V16 动作 Query 的单 token 聚合嵌入。

    ``E_q = E_action + E_queryType + Σ_i E_{type,i}(answer_i)``,再经
    LayerNorm/Projection 到 d_model。Offense 与 Defense 使用各自独立、结构对称
    的 10 组 slot 嵌入;两个分支共用同一个 action/queryType 嵌入与投影层。
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.action = nn.Embedding(NUM_ACTIONS, d_model)
        self.query_type = nn.Embedding(3, d_model)
        self.offense_slots = nn.ModuleList(
            nn.Embedding(SLOT_CARDINALITIES[slot], d_model) for slot in OFFENSE_SLOT_ORDER
        )
        self.defense_slots = nn.ModuleList(
            nn.Embedding(SLOT_CARDINALITIES[slot], d_model) for slot in DEFENSE_SLOT_ORDER
        )
        self.norm = nn.RMSNorm(d_model)
        self.projection = nn.Linear(d_model, d_model, bias=False)
        for embedding in (self.action, self.query_type, *self.offense_slots, *self.defense_slots):
            nn.init.normal_(embedding.weight, std=1.0 / math.sqrt(d_model))

    def forward(self, rows: Tensor) -> Tensor:
        """``rows`` 为 [B, Q, QUERY_ROW_WIDTH] 的整型编码行。"""
        if rows.ndim != 3 or rows.shape[-1] != QUERY_ROW_WIDTH:
            raise ValueError(f"query rows must be [batch, queries, {QUERY_ROW_WIDTH}]")
        rows = rows.long()
        query_type = rows[..., 0]
        action_ids = rows[..., 1]
        answers = rows[..., 5:]
        if torch.any(action_ids < 0) or torch.any(action_ids >= NUM_ACTIONS):
            raise ValueError("query action_id is outside the fixed action space")
        if torch.any(query_type < 0) or torch.any(query_type > 2):
            raise ValueError("query_type must be 0..2")
        embedded = self.action(action_ids) + self.query_type(query_type)
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
    """V16 Compact Snapshot 的分段混合嵌入。

    快照存储为统一形状的行(kind + 4 列 categorical + 7 列连续),每种 kind 使用
    独立的 categorical 基数与连续宽度,产出各自的一条 d_model token:
    base(场况)、dora(每张宝牌指示一条)、score(点数/分差)、summary(3×7 对手摘要)。
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        # 场风 E/S(2)、局数 E1..S4(8)、庄家相对(4)、自身顺位(4)。
        self.base = FactorEmbedding((2, 8, 4, 4), d_model, 3)
        self.dora = FactorEmbedding((34,), d_model, 0)
        self.score = FactorEmbedding((), d_model, SNAPSHOT_NUM_WIDTH)
        self.summary = FactorEmbedding((2, 2), d_model, 5)

    def forward(self, kinds: Tensor, categorical: Tensor, numeric: Tensor) -> Tensor:
        """``kinds`` [B,S]、``categorical`` [B,S,4]、``numeric`` [B,S,7]。"""
        if kinds.ndim != 2 or categorical.ndim != 3 or numeric.ndim != 3:
            raise ValueError("snapshot kinds/categorical/numeric shapes are malformed")
        if categorical.shape[-1] != SNAPSHOT_CAT_WIDTH or numeric.shape[-1] != SNAPSHOT_NUM_WIDTH:
            raise ValueError(
                f"snapshot rows must be {SNAPSHOT_CAT_WIDTH} categorical + "
                f"{SNAPSHOT_NUM_WIDTH} numeric columns"
            )
        if categorical.shape[:2] != kinds.shape or numeric.shape[:2] != kinds.shape:
            raise ValueError("snapshot kind/categorical/numeric row counts differ")
        if torch.any(kinds < 0) or torch.any(kinds >= 4):
            raise ValueError("snapshot kind must be 0..3")
        batch, rows, _width = numeric.shape
        device = numeric.device
        output = numeric.new_zeros((batch, rows, self.d_model))
        kind = kinds.unsqueeze(-1)
        routes = (
            (SNAPSHOT_KIND_BASE, self.base, 4, 3, 7),
            (SNAPSHOT_KIND_DORA, self.dora, 1, 0, 33),
            (SNAPSHOT_KIND_SCORE, self.score, 0, SNAPSHOT_NUM_WIDTH, 0),
            (SNAPSHOT_KIND_SUMMARY, self.summary, 2, 5, 1),
        )
        for kind_code, module, cat_width, num_width, cat_max in routes:
            # 每个模块会在全量行上求值,非本 kind 行的取值被 where 丢弃;先按该
            # 模块的基数截断,避免非本 kind 行越界触发 Embedding 索引错误。
            factors = categorical[:, :, :cat_width].long().clamp(0, cat_max)
            values = numeric[:, :, :num_width] if num_width else None
            embedded = module(factors, values)
            output = torch.where(kind.eq(kind_code), embedded, output)
        return output


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
        self.config = config or ModelConfig.preset("mid")
        self.token_embedding = FactorEmbedding(TOKEN_CARDINALITIES, self.config.d_model, NUMERIC_WIDTH)
        self.critic_embedding = FactorEmbedding(TOKEN_CARDINALITIES, self.config.d_model)
        self.value_query = nn.Parameter(torch.empty(self.config.d_model))
        # The policy traverses ``layers`` public/actor blocks.  The centralized
        # value branch then gets its own ``critic_layers`` blocks after private
        # opponent-hand tokens are appended.
        self.public_backbone = Decoder(self.config, layers=self.config.shared_layers, final_norm=False)
        self.actor_backbone = Decoder(self.config, layers=self.config.layers - self.config.shared_layers)
        self.critic_backbone = Decoder(self.config, layers=self.config.critic_layers)
        # V16:Offense/Defense 对称融合(concat 384→192→SiLU→Policy MLP),
        # 普通初始化、无 zero-init 分支、无 241 维 Q head。
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
        # Top-3 Q scorer:输入 [z_critic; detach(h_a)] →384→192→SiLU→1。
        self.q_scorer = nn.Sequential(
            nn.Linear(2 * self.config.d_model, self.config.d_model),
            nn.SiLU(),
            nn.Linear(self.config.d_model, 1),
        )
        nn.init.normal_(self.value_query, std=self.config.d_model ** -0.5)

    def forward(self, *args, **kwargs):
        """DDP 兼容的 V16 前向分发。"""
        if args:
            raise TypeError("V16 forward only accepts keyword arguments")
        return self.forward_v16(**kwargs)

    def forward_v16(
        self,
        history_factors: Tensor,
        history_numeric: Tensor,
        history_lengths: Tensor,
        snapshot_kinds: Tensor,
        snapshot_cat: Tensor,
        snapshot_num: Tensor,
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
        """V16 前向:Objective Facts + Compact Snapshot + 每动作一对 Query。

        Query token 加入主序列末尾(每动作 Offense/Defense 相邻),策略头对每一
        对做对称融合;Critic 公共前缀只含 Objective Facts + Snapshot(不追加
        Action Query Token),其后拼接特权输入(三家手牌 + 后 5 牌山)与 Value
        Query。
        """
        if self.config.policy_head_type != "symmetric_action_query":
            raise ValueError("forward_v16 requires the v16 symmetric policy head")
        if history_factors.ndim != 3 or history_factors.shape[-1] != TOKEN_WIDTH:
            raise ValueError(f"history_factors must be [batch, tokens, {TOKEN_WIDTH}]")
        if history_numeric.shape != (*history_factors.shape[:2], NUMERIC_WIDTH):
            raise ValueError(f"history_numeric must be [batch, tokens, {NUMERIC_WIDTH}]")
        if snapshot_kinds.ndim != 2 or snapshot_cat.shape != (*snapshot_kinds.shape, SNAPSHOT_CAT_WIDTH):
            raise ValueError("snapshot categorical shape is malformed")
        if snapshot_num.shape != (*snapshot_kinds.shape, SNAPSHOT_NUM_WIDTH):
            raise ValueError("snapshot numeric shape is malformed")
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
            snapshot_lengths, snapshot_kinds.shape[1], "snapshot_lengths",
        )
        pair_counts = lengths(
            query_pair_counts, query_action_ids.shape[1], "query_pair_counts",
        )
        if legal_mask.shape != (batch, NUM_ACTIONS):
            raise ValueError(f"legal_mask must be [batch, {NUM_ACTIONS}]")
        if torch.any(query_action_ids < 0) or torch.any(query_action_ids >= NUM_ACTIONS):
            raise ValueError("query_action_ids are outside the fixed action space")
        if torch.any(history_lengths + snapshot_lengths + 2 * pair_counts > self.config.context_tokens):
            raise ValueError("v16 context overflow: history + snapshot + queries exceed context_tokens")

        history = self.token_embedding(history_factors, history_numeric)
        snapshot = self.snapshot_embeddings(snapshot_kinds, snapshot_cat, snapshot_num)
        queries = self.query_embedding(query_rows)
        total_lengths = history_lengths + snapshot_lengths + 2 * pair_counts
        # 三个段必须按每行自身长度左对齐打包成一条序列。直接把整段 padding
        # 的 tensor cat 起来,会让短行的 snapshot/query 落在 padding 空隙里、
        # 真实内容反而超出 valid 区间,导致同一行的输出随 batchmates 变化
        # (rollout 与 update 重算的 logprob 因此不一致)。
        capacity = int(total_lengths.max().item())
        tokens = history.new_zeros((batch, capacity, self.config.d_model))
        rows = torch.arange(batch, device=device)

        def scatter(segment: Tensor, start: Tensor, count: Tensor) -> None:
            local = torch.arange(segment.shape[1], device=device)[None, :]
            valid = local < count[:, None]
            destination = start[:, None] + local
            rows_expanded = rows[:, None].expand_as(destination)
            tokens[rows_expanded[valid], destination[valid]] = segment[valid].to(
                tokens.dtype
            )

        scatter(history, torch.zeros(batch, device=device, dtype=torch.long), history_lengths)
        scatter(snapshot, history_lengths, snapshot_lengths)
        scatter(queries, history_lengths + snapshot_lengths, 2 * pair_counts)
        attention_mask, valid = _attention_layout(total_lengths, tokens.shape[1])
        position_ids = torch.arange(tokens.shape[1], device=device)[None].expand(batch, -1)
        shared = self.public_backbone(
            tokens, total_lengths, attention_mask=attention_mask, valid=valid,
            position_ids=position_ids,
        )
        actor = self.actor_backbone(
            shared, total_lengths, attention_mask=attention_mask, valid=valid,
            position_ids=position_ids,
        )

        # 对称融合策略头:同一动作的 offense/defense 表示 concat 后共享投影。
        action_capacity = query_action_ids.shape[1]
        rows = torch.arange(batch, device=device)
        pair_offsets = torch.arange(action_capacity, device=device)[None].expand(batch, -1)
        public_prefix = (history_lengths + snapshot_lengths)[:, None]
        # 打包后序列容量 = 本 batch 最大 total_lengths;padding pair 的位置可能
        # 超出容量,先 clamp,其取值随后被 pair_valid 丢弃。
        offense_positions = torch.clamp(
            public_prefix + 2 * pair_offsets, max=tokens.shape[1] - 1,
        )
        defense_positions = torch.clamp(
            public_prefix + 2 * pair_offsets + 1, max=tokens.shape[1] - 1,
        )
        pair_valid = pair_offsets < pair_counts[:, None]
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
        public_lengths = history_lengths + snapshot_lengths
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

    def q_scores_v16(
        self,
        critic_hidden: Tensor,
        action_hiddens: Tensor,
        action_ids: Tensor,
        pair_counts: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor:
        """对候选动作输出原始优势评分 u:输入 [z_critic; detach(h_a)] →
        384→192→SiLU→1。

        ``action_hiddens`` 在进入 scorer 前强制 detach,保证 Q loss 不会经动作
        表示直接更新 Actor;无效候选(越界/缺失)返回 -inf。这里只输出未做
        Dueling 约束的原始评分,最终 Q 由 ``dueling_candidate_q`` 合成。
        """
        batch, action_capacity = action_ids.shape
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] != batch:
            raise ValueError("candidate_ids must be [batch, candidates]")
        device = action_ids.device
        pair_positions = torch.arange(action_capacity, device=device)[None].expand(batch, -1)
        valid_pairs = pair_positions < pair_counts[:, None]
        # action→pair 索引必须用高级索引只写有效 pair。scatter_ 会把 padding 的
        # action_id(编码阶段以 0 补齐)重复写进 action 0 所在列,污染候选索引。
        pair_index = action_ids.new_full((batch, NUM_ACTIONS), -1)
        rows = torch.arange(batch, device=device)[:, None].expand_as(action_ids)
        pair_index[
            rows[valid_pairs], action_ids.to(device=device, dtype=torch.long)[valid_pairs]
        ] = pair_positions[valid_pairs].to(pair_index.dtype)
        # -1 补齐的候选不能直接进 gather(负数索引越界);先 clamp 后按原始 id
        # 判定有效性,无效候选统一返回 -inf。
        safe_ids = candidate_ids.clamp(min=0)
        candidate_positions = pair_index.gather(1, safe_ids.to(device=device, dtype=torch.long))
        valid = (candidate_ids >= 0) & (candidate_positions >= 0)
        safe = candidate_positions.clamp(min=0)
        rows = torch.arange(batch, device=device)[:, None].expand_as(candidate_ids)
        hidden = action_hiddens[rows, safe].detach()
        critic = critic_hidden[:, None, :].expand(batch, candidate_ids.shape[1], -1)
        scores = self.q_scorer(torch.cat((critic, hidden), dim=-1)).squeeze(-1).float()
        return torch.where(valid, scores, scores.new_full((), float("-inf")))


def dueling_candidate_q(
    raw_scores: Tensor,
    candidate_probs: Tensor,
    value: Tensor,
    *,
    detach_value: bool = False,
) -> tuple[Tensor, Tensor]:
    """Dueling-style 候选 Q 合成:u_i → A_i、Q_i = V(s) + A_i。

    对候选集合内的概率重新归一化得到 p_i,再计算均值基线
    ``A_i = u_i - sum(p_j * u_j)``,最终 ``Q_i = V(s) + A_i``。由构造恒有
    ``sum(p_i * Q_i) = V(s)``:Value 只负责绝对局面价值,Top-3 Q 只编码候选
    之间的相对差异。无效候选(评分为 -inf 或 padding 位置)不参与归一化,
    对应 Q 保持 -inf。

    ``detach_value=True`` 时 Q loss 不向 ``value_head`` 回传梯度,Value 仍由
    独立的 return-target loss 训练。
    """
    if raw_scores.shape != candidate_probs.shape:
        raise ValueError("raw_scores and candidate_probs must share the same shape")
    if value.ndim != 1 or value.shape[0] != raw_scores.shape[0]:
        raise ValueError("value must be [batch]")
    valid = torch.isfinite(raw_scores) & (candidate_probs >= 0)
    safe_scores = torch.where(valid, raw_scores, torch.zeros_like(raw_scores))
    safe_probs = torch.where(
        valid, candidate_probs, torch.zeros_like(candidate_probs)
    )
    total = safe_probs.sum(dim=-1, keepdim=True)
    normalized = safe_probs / total.clamp_min(1e-12)
    baseline = (normalized * safe_scores).sum(dim=-1, keepdim=True)
    advantages = torch.where(
        valid, safe_scores - baseline, torch.zeros_like(safe_scores)
    )
    value_term = value.detach() if detach_value else value
    q_values = value_term.unsqueeze(-1) + advantages
    return advantages, torch.where(
        valid, q_values, q_values.new_full((), float("-inf"))
    )
