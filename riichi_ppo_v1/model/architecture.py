"""语义 token 解码器型 GQA 演员-评论家。

v13 契约使用隔离的两 token 动作查询(offense/defense 相邻对),输出固定的
241 维动作空间。历史 v11 checkpoint 兼容已移除,模型头只支持
``isolated_action_query``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .feature_schema import (
    ACTION_QUERY_DEFENSE,
    ACTION_QUERY_OFFENSE,
    ACTION_QUERY_SEGMENT,
)
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
    policy_head_type: str = "isolated_action_query"
    offense_fusion: bool = False
    critic_head_type: str = "state_value"

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
        if self.policy_head_type not in {"isolated_action_query", "symmetric_action_query"}:
            raise ValueError(
                "policy_head_type must be isolated_action_query or symmetric_action_query"
            )
        if self.critic_head_type not in {"state_value", "action_value"}:
            raise ValueError("critic_head_type must be state_value or action_value")

    @classmethod
    def preset(cls, size: str) -> "ModelConfig":
        configs = {
            "mid": cls(),
            "large": cls(d_model=384, query_heads=12, kv_heads=3, head_dim=32, ffn_dim=1152),
            "v16": cls(
                layers=5,
                shared_layers=4,
                critic_layers=2,
                d_model=256,
                query_heads=16,
                kv_heads=4,
                head_dim=16,
                ffn_dim=1088,
                policy_head_type="symmetric_action_query",
            ),
        }
        try:
            return configs[size]
        except KeyError as exc:
            raise ValueError("model size must be 'mid', 'large' or 'v16'") from exc


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


def isolated_action_layout(
    factors: Tensor,
    lengths: Tensor,
    legal_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build and validate the v13 state/action block-sparse causal layout.

    Returns ``mask, valid, position_ids, state_mask, defense_action_ids``.
    Query rows are required to be a suffix of adjacent offense/defense pairs.
    Pair order is intentionally unconstrained: identical position ids and
    isolation must make the policy invariant to candidate permutation.
    """
    if factors.ndim != 3 or factors.shape[-1] != TOKEN_WIDTH:
        raise ValueError("token_factors must be [batch, tokens, 10]")
    batch, tokens, _ = factors.shape
    device = factors.device
    token_index = torch.arange(tokens, device=device)[None].expand(batch, -1)
    valid = token_index < lengths[:, None]
    query = valid & factors[..., 0].eq(ACTION_QUERY_SEGMENT)
    offense = query & factors[..., 9].eq(ACTION_QUERY_OFFENSE)
    defense = query & factors[..., 9].eq(ACTION_QUERY_DEFENSE)
    state = valid & ~query
    first_query = torch.where(query, token_index, tokens).amin(dim=1)
    expected_query = valid & (token_index >= first_query[:, None])
    pair_offset = token_index - first_query[:, None]
    expected_offense = expected_query & pair_offset.remainder(2).eq(0)
    expected_defense = expected_query & pair_offset.remainder(2).eq(1)
    action_ids = factors[..., 2].long() - 1
    previous_ids = torch.cat((action_ids.new_full((batch, 1), -1), action_ids[:, :-1]), dim=1)
    malformed = (
        first_query.eq(tokens)
        | query.sum(dim=1).remainder(2).ne(0)
        | (query != expected_query).any(dim=1)
        | (offense != expected_offense).any(dim=1)
        | (defense != expected_defense).any(dim=1)
        | (defense & action_ids.ne(previous_ids)).any(dim=1)
        | (query & (action_ids.lt(0) | action_ids.ge(NUM_ACTIONS))).any(dim=1)
    )
    offense_counts = torch.zeros((batch, NUM_ACTIONS), dtype=torch.long, device=device)
    offense_counts.scatter_add_(1, action_ids.clamp(0, NUM_ACTIONS - 1), offense.long())
    malformed |= offense_counts.gt(1).any(dim=1)
    if legal_mask is not None:
        if legal_mask.shape != (batch, NUM_ACTIONS):
            raise ValueError(f"legal_mask must be [batch, {NUM_ACTIONS}]")
        malformed |= (offense_counts != legal_mask.long()).any(dim=1)
    if bool(malformed.any()):
        raise ValueError(
            "each legal action must have one adjacent offense/defense query pair; "
            "queries must form a valid suffix with matching unique action ids"
        )

    q_index = token_index[:, :, None]
    k_index = token_index[:, None, :]
    state_q, state_k = state[:, :, None], state[:, None, :]
    query_q = query[:, :, None]
    mask = (
        (state_q & state_k & k_index.le(q_index))
        | (query_q & state_k)
        | (query_q & k_index.eq(q_index))
        | (defense[:, :, None] & k_index.eq(q_index - 1))
    ).unsqueeze(1)
    positions = torch.where(
        state, token_index,
        torch.where(offense, first_query[:, None], first_query[:, None] + 1),
    )
    defense_ids = torch.where(defense, action_ids, -1)
    # Avoid NaNs on padded query rows; valid masking zeroes their outputs.
    padded = ~valid
    mask[:, 0, :, 0] |= padded
    return mask, valid, positions, state, defense_ids


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
        self.register_parameter("query", None)
        if self.config.policy_head_type == "symmetric_action_query":
            # V16:Offense/Defense 对称融合(concat 512→256→SiLU→Policy MLP),
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
            self.offense_projection = None
            self.q_head = None
            self.value_head = nn.Linear(self.config.d_model, 1)
            # Top-3 Q scorer:输入 [z_critic; detach(h_a)] →512→256→SiLU→1。
            self.q_scorer = nn.Sequential(
                nn.Linear(2 * self.config.d_model, self.config.d_model),
                nn.SiLU(),
                nn.Linear(self.config.d_model, 1),
            )
        else:
            self.policy_head = nn.Sequential(
                nn.RMSNorm(self.config.d_model, eps=self.config.eps),
                nn.Linear(self.config.d_model, self.config.d_model),
                nn.SiLU(),
                nn.Linear(self.config.d_model, 1),
            )
            if self.config.offense_fusion:
                self.offense_projection: nn.Module | None = nn.Linear(
                    self.config.d_model, self.config.d_model
                )
                nn.init.zeros_(self.offense_projection.weight)
                nn.init.zeros_(self.offense_projection.bias)
            else:
                self.offense_projection = None
            if self.config.critic_head_type == "action_value":
                self.q_head: nn.Module | None = nn.Linear(self.config.d_model, NUM_ACTIONS)
                self.value_head: nn.Module | None = None
            else:
                self.value_head = nn.Linear(self.config.d_model, 1)
                self.q_head = None
        nn.init.normal_(self.value_query, std=self.config.d_model ** -0.5)

    def _isolated_public(
        self, token_factors: Tensor, token_numeric: Tensor, token_lengths: Tensor,
        legal_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        tokens = self.token_embedding(token_factors, token_numeric)
        layout = isolated_action_layout(token_factors, token_lengths, legal_mask)
        attention_mask, valid, position_ids, state_mask, defense_ids = layout
        shared = self.public_backbone(
            tokens, token_lengths, attention_mask=attention_mask, valid=valid,
            position_ids=position_ids,
        )
        actor = self.actor_backbone(
            shared, token_lengths, attention_mask=attention_mask, valid=valid,
            position_ids=position_ids,
        )
        return shared, actor, state_mask, defense_ids, valid

    def _isolated_logits(
        self, actor: Tensor, defense_ids: Tensor, legal_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch = actor.shape[0]
        raw = actor.new_zeros((batch, NUM_ACTIONS), dtype=torch.float32)
        defense = defense_ids.ge(0)
        rows, positions = torch.nonzero(defense, as_tuple=True)
        policy_hidden = actor[rows, positions]
        if self.offense_projection is not None:
            policy_hidden = policy_hidden + self.offense_projection(actor[rows, positions - 1])
        scores = self.policy_head(policy_hidden).squeeze(-1).float()
        raw[rows, defense_ids[rows, positions]] = scores
        if legal_mask is None:
            inferred = torch.zeros_like(raw, dtype=torch.bool)
            inferred[rows, defense_ids[rows, positions]] = True
            legal_mask = inferred
        logits = raw.masked_fill(~legal_mask.to(device=raw.device, dtype=torch.bool), float("-inf"))
        return raw, logits

    def forward_policy(
        self,
        token_factors: Tensor,
        token_numeric: Tensor,
        legal_mask: Tensor | None = None,
        token_lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run only the shared-public and actor branches."""
        if token_factors.shape[1] > self.config.context_tokens:
            raise ValueError(f"context overflow: {token_factors.shape[1]} > {self.config.context_tokens}")
        if token_lengths is None:
            token_lengths = token_factors.ne(0).any(-1).long().sum(-1)
        token_lengths = token_lengths.to(device=token_factors.device, dtype=torch.long)
        if token_lengths.shape != (token_factors.shape[0],):
            raise ValueError("token_lengths must have one entry per batch row")
        if torch.any(token_lengths < 0) or torch.any(token_lengths > token_factors.shape[1]):
            raise ValueError("token_lengths exceed supplied token rows")
        _shared, actor, _state, defense_ids, _valid = self._isolated_public(
            token_factors, token_numeric, token_lengths, legal_mask
        )
        raw, logits = self._isolated_logits(actor, defense_ids, legal_mask)
        return {"raw_policy_logits": raw, "policy_logits": logits}

    def forward(
        self,
        token_factors: Tensor,
        token_numeric: Tensor,
        legal_mask: Tensor | None = None,
        token_lengths: Tensor | None = None,
        *,
        critic_factors: Tensor | None = None,
        critic_lengths: Tensor | None = None,
        detach_critic_public: bool = False,
        critic_public_grad_scale: float = 1.0,
        policy_only: bool = False,
    ) -> dict[str, Tensor]:
        if policy_only:
            return self.forward_policy(
                token_factors, token_numeric, legal_mask, token_lengths
            )
        if token_factors.shape[1] > self.config.context_tokens:
            raise ValueError(f"context overflow: {token_factors.shape[1]} > {self.config.context_tokens}")
        if token_lengths is None:
            token_lengths = token_factors.ne(0).any(-1).long().sum(-1)
        token_lengths = token_lengths.to(device=token_factors.device, dtype=torch.long)
        if token_lengths.shape != (token_factors.shape[0],):
            raise ValueError("token_lengths must have one entry per batch row")
        if torch.any(token_lengths < 0) or torch.any(token_lengths > token_factors.shape[1]):
            raise ValueError("token_lengths exceed supplied token rows")
        tokens = self.token_embedding(token_factors, token_numeric)
        batch, padded, width = tokens.shape
        rows = torch.arange(batch, device=tokens.device)
        public_sequence, actor_sequence, state_mask, defense_ids, _valid = self._isolated_public(
            token_factors, token_numeric, token_lengths, legal_mask
        )
        raw, logits = self._isolated_logits(actor_sequence, defense_ids, legal_mask)
        public_lengths = state_mask.long().sum(-1)
        public_capacity = int(public_lengths.max().item())
        packed_public = public_sequence.new_zeros((batch, public_capacity, width))
        source_rows, source_positions = torch.nonzero(state_mask, as_tuple=True)
        packed_positions = state_mask.long().cumsum(dim=1)[
            source_rows, source_positions,
        ] - 1
        packed_public[source_rows, packed_positions] = public_sequence[
            source_rows, source_positions,
        ]
        public_sequence = packed_public
        if critic_factors is None:
            critic_factors = token_factors.new_zeros((batch, 0, TOKEN_WIDTH))
        if critic_lengths is None:
            critic_lengths = critic_factors.ne(0).any(-1).long().sum(-1)
        critic_factors = critic_factors.to(device=token_factors.device)
        critic_lengths = critic_lengths.to(device=token_factors.device, dtype=torch.long)
        if critic_factors.ndim != 3 or critic_factors.shape[0] != batch or critic_factors.shape[-1] != TOKEN_WIDTH:
            raise ValueError("critic_factors must be [batch, critic_tokens, 10]")
        if critic_lengths.shape != (batch,):
            raise ValueError("critic_lengths must have one entry per batch row")
        if torch.any(critic_lengths < 0) or torch.any(critic_lengths > critic_factors.shape[1]):
            raise ValueError("critic_lengths exceed supplied critic rows")
        critic_sequence_lengths = public_lengths + critic_lengths + 1
        if torch.any(critic_sequence_lengths > self.config.context_tokens):
            raise ValueError("critic context overflow: public tokens + critic tokens + two queries exceed context_tokens")
        critic_private = self.critic_embedding(critic_factors)
        critic_capacity = int(critic_sequence_lengths.max().item())
        critic_sequence = critic_private.new_zeros((batch, critic_capacity, width))
        # Preserve every shared-public representation, including the learned
        # policy query at the end of each row.  Value gradients improve this
        # shared prefix but cannot update the actor-only policy tail.
        public_positions = torch.arange(public_sequence.shape[1], device=tokens.device)[None, :].expand(batch, -1)
        public_valid = public_positions < public_lengths[:, None]
        public_grad_scale = 0.0 if detach_critic_public else float(critic_public_grad_scale)
        if not 0.0 <= public_grad_scale <= 1.0:
            raise ValueError("critic_public_grad_scale must be in [0, 1]")
        if public_grad_scale == 0.0:
            critic_public = public_sequence.detach()
        elif public_grad_scale == 1.0:
            critic_public = public_sequence
        else:
            detached_public = public_sequence.detach()
            # Preserve the exact forward value while scaling only the value
            # branch gradient entering the shared public representation.
            critic_public = detached_public + public_grad_scale * (
                public_sequence - detached_public
            )
        critic_sequence[
            rows[:, None].expand_as(public_positions)[public_valid], public_positions[public_valid]
        ] = critic_public[public_valid]
        private_positions = public_lengths[:, None] + torch.arange(
            critic_private.shape[1], device=tokens.device
        )[None, :]
        private_valid = torch.arange(critic_private.shape[1], device=tokens.device)[None, :] < critic_lengths[:, None]
        critic_sequence[rows[:, None].expand_as(private_positions)[private_valid], private_positions[private_valid]] = critic_private[private_valid]
        value_indices = public_lengths + critic_lengths
        critic_sequence[rows, value_indices] = self.value_query
        critic_hidden = self.critic_backbone(critic_sequence, critic_sequence_lengths)[rows, value_indices]
        # Autocast 使前面的矩阵乘落在 BF16,但策略概率与价值估计要进入 PPO 的
        # 数值敏感 ratio/loss 路径,离开模型前先提升精度。
        output = {
            "raw_policy_logits": raw,
            "policy_logits": logits,
        }
        if self.q_head is not None:
            output["q_values"] = self.q_head(critic_hidden).float()
        else:
            assert self.value_head is not None
            output["value"] = self.value_head(critic_hidden).squeeze(-1).float()
        return output

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
        512→256→SiLU→1。

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
