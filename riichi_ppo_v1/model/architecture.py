"""V18 当前局面快照 GQA Actor-Critic。

Actor 输入为完整状态快照序列（Shared 公共前缀 + 三个 Opponent Analysis + 按 action ID
升序的 Offense/Defense Query），所有有效 token 使用连续 RoPE 位置；Shared 公共 backbone 用
双向 GQA；Actor-only 层用结构化隔离 mask；Critic 在公共表示后拼接三家真实闭手与未来五张牌
（独立尾部），不接收 Analysis/Action token。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dense_embedding import StateTokenEmbedding
from .encoding_protocol import (
    CONTEXT_TOKENS,
    KIND_BOS,
    KIND_CRITIC_FUTURE,
    KIND_CRITIC_HAND,
    KIND_SEP_ACTIONS,
    KIND_SEP_CRITIC,
    SEGMENT_ACTIONS,
    SEGMENT_ANALYSIS,
    SEGMENT_SHARED,
    TOKEN_NUMERIC_WIDTH,
    TOKEN_ROW_WIDTH,
)
from .schema import NUM_ACTIONS


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
    context_tokens: int = CONTEXT_TOKENS
    rope_base: float = 10_000.0
    eps: float = 1e-6
    dense_slot_dim: int = 32
    dense_fusion_dim: int = 512
    policy_head_type: str = "current_state_snapshot"

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
        if self.dense_slot_dim < 8:
            raise ValueError("dense_slot_dim must be at least 8")
        if self.dense_fusion_dim < self.d_model:
            raise ValueError("dense_fusion_dim must be at least d_model")
        if self.policy_head_type != "current_state_snapshot":
            raise ValueError("policy_head_type must be current_state_snapshot")

    @classmethod
    def preset(cls, size: str) -> "ModelConfig":
        if size != "v18":
            raise ValueError("model size must be 'v18'")
        return cls()

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "ModelConfig":
        """从纯 V18 checkpoint/config 映射恢复精确拓扑。"""
        return cls(**dict(values))


class GQA(nn.Module):
    """RoPE + GQA 注意力（causal 与否完全由 attention_mask 决定）。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.qh, self.kvh, self.head_dim = config.query_heads, config.kv_heads, config.head_dim
        self.qkv = nn.Linear(config.d_model, (self.qh + 2 * self.kvh) * self.head_dim, bias=False)
        self.out = nn.Linear(self.qh * self.head_dim, config.d_model, bias=False)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], attention_mask: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        shape = lambda value, heads: value.view(batch, tokens, heads, self.head_dim).transpose(1, 2)
        q_raw, k_raw, v_raw = self.qkv(x).split(
            (self.qh * self.head_dim, self.kvh * self.head_dim, self.kvh * self.head_dim), dim=-1
        )
        q, k, v = _rope(shape(q_raw, self.qh), rope), _rope(shape(k_raw, self.kvh), rope), shape(v_raw, self.kvh)
        repeat = self.qh // self.kvh
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        value = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask[:, None], dropout_p=0.0,
        )
        return self.out(value.transpose(1, 2).reshape(batch, tokens, self.qh * self.head_dim))


def _rope_values(position_ids: Tensor, head_dim: int, dtype: torch.dtype, base: float) -> tuple[Tensor, Tensor]:
    positions = position_ids.to(dtype=torch.float32)
    frequencies = torch.exp(
        torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32)
        * (-math.log(base) / head_dim)
    )
    angle = positions[..., None] * frequencies
    return angle.cos().to(dtype).unsqueeze(1), angle.sin().to(dtype).unsqueeze(1)


def _rope(x: Tensor, values: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = values
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def _first_key_escape(mask: Tensor, valid_query: Tensor) -> Tensor:
    """padding 行需要至少一个有限 key 以避免 softmax NaN；随后被 valid 清零。"""
    first_key = torch.zeros_like(mask)
    first_key[..., 0] = True
    return (mask & valid_query) | (first_key & ~valid_query)


def _bidirectional_layout(lengths: Tensor, tokens: int) -> tuple[Tensor, Tensor]:
    """公共前缀的双向 GQA 布局。"""
    device = lengths.device
    positions = torch.arange(tokens, device=device)[None, :]
    valid = positions < lengths[:, None]
    mask = valid[:, :, None] & valid[:, None, :]
    return _first_key_escape(mask, valid[:, :, None]), valid


def _actor_structured_layout(
    segments: Tensor,
    kinds: Tensor,
    lengths: Tensor,
    tokens: int,
) -> tuple[Tensor, Tensor]:
    """根据 segment/kind 类别构造 Actor 结构化 mask。

    - Shared(segment 1，含共享分隔符) ↔ Shared 双向；
    - Analysis(segment 2 + SEP_ACTIONS 分隔符) → Shared ∪ Analysis；
    - 每个 Action pair（kind 11/12 中相邻两行）→ Shared ∪ Analysis ∪ 本 pair；
    - 不同 pair 互不可见；padding 不可见。
    """
    device = segments.device
    positions = torch.arange(tokens, device=device)[None, :]
    valid = positions < lengths[:, None]
    is_shared = segments.eq(SEGMENT_SHARED)
    is_analysis = segments.eq(SEGMENT_ANALYSIS)
    is_sep_actions = segments.eq(SEGMENT_ACTIONS) & kinds.eq(KIND_SEP_ACTIONS)
    is_action = kinds.eq(11) | kinds.eq(12)
    action_mask = is_action & valid
    action_ranks = action_mask.long().cumsum(dim=1) - 1
    pair_id = torch.where(action_mask, action_ranks // 2, torch.zeros_like(action_ranks))
    same_pair = pair_id[:, :, None].eq(pair_id[:, None, :]) & is_action[:, :, None] & is_action[:, None, :]
    from_shared = is_shared[:, :, None] & is_shared[:, None, :]
    from_analysis = is_analysis[:, :, None] & (is_shared[:, None, :] | is_analysis[:, None, :])
    from_sep_actions = is_sep_actions[:, :, None] & is_sep_actions[:, None, :]
    from_action = is_action[:, :, None] & (
        is_shared[:, None, :] | is_analysis[:, None, :] | is_sep_actions[:, None, :] | same_pair
    )
    mask = from_shared | from_analysis | from_sep_actions | from_action
    mask = mask & valid[:, :, None] & valid[:, None, :]
    return _first_key_escape(mask, valid[:, :, None]), valid


def _critic_layout(lengths: Tensor, tokens: int) -> tuple[Tensor, Tensor]:
    """Critic 分支：全部有效 token 双向（Value Query 位于末尾并可读全部）。"""
    return _bidirectional_layout(lengths, tokens)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(config.d_model, eps=config.eps), nn.RMSNorm(config.d_model, eps=config.eps)
        self.attention = GQA(config)
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
        attention_mask: Tensor,
        valid: Tensor,
    ) -> Tensor:
        tokens = x.shape[1]
        if tokens > self.context_tokens:
            raise ValueError(f"context overflow: {tokens} > {self.context_tokens}")
        position_ids = torch.arange(tokens, device=x.device)[None].expand(x.shape[0], -1)
        rope = _rope_values(position_ids, x.shape[-1] // self.blocks[0].attention.qh, x.dtype, self.rope_base)
        for block in self.blocks:
            x = block(x, rope, attention_mask, valid)
        return self.norm(x)


def _segment_map(kind: Tensor) -> Tensor:
    """kind → 期望 segment 的整批查表（1..14 与 101..111 之外的 kind 映射为 0）。

    等价于逐 token 的类别→segment 映射，纯向量化、无 per-row Python 循环。
    """
    from .encoding_protocol import (
        SEGMENT_ACTIONS,
        SEGMENT_ANALYSIS,
        SEGMENT_CRITIC_FUTURE,
        SEGMENT_CRITIC_PRIVATE,
        SEGMENT_SHARED,
    )

    is_sep = (kind >= 101) & (kind <= 111)
    shared = ((kind >= 1) & (kind <= 9)) | ((kind >= 101) & (kind <= 108))
    analysis = (kind == 10) | (kind == 109)
    actions = (kind == 11) | (kind == 12) | (kind == 110)
    critic_private = (kind == 13) | (kind == 111)
    critic_future = kind == 14
    invalid = ~(shared | analysis | actions | critic_private | critic_future) & ~is_sep
    return torch.where(
        invalid, torch.zeros_like(kind),
        torch.where(
            shared, torch.full_like(kind, SEGMENT_SHARED),
            torch.where(
                analysis, torch.full_like(kind, SEGMENT_ANALYSIS),
                torch.where(
                    actions, torch.full_like(kind, SEGMENT_ACTIONS),
                    torch.where(
                        critic_private, torch.full_like(kind, SEGMENT_CRITIC_PRIVATE),
                        torch.full_like(kind, SEGMENT_CRITIC_FUTURE),
                    ),
                ),
            ),
        ),
    )


def _assert_structure(factors: Tensor, lengths: Tensor, *, critic: bool) -> None:
    """轻量结构校验：段/类别一致性、BOS 开头、分隔符位置、action 数量（fail closed）。

    整批向量化实现（旧实现逐行 tolist + Python 循环，实测一次 forward 占 ~32ms）。
    """
    from .encoding_protocol import (
        KIND_BOS,
        KIND_CRITIC_FUTURE,
        KIND_CRITIC_HAND,
        KIND_SEP_CRITIC,
        SEGMENT_ACTIONS,
    )

    batch, tokens, _width = factors.shape
    if batch == 0:
        return
    device = factors.device
    lengths = lengths.to(device=device, dtype=torch.long)
    positions = torch.arange(tokens, device=device)[None, :]
    valid = positions < lengths[:, None]
    kind = factors[..., 1].long()
    seg = factors[..., 0].long()
    expected_seg = _segment_map(kind)
    if bool((valid & (seg != expected_seg)).any()):
        raise ValueError("token segment field disagrees with its kind schema")
    if critic:
        if bool((lengths == 0).any()):
            raise ValueError("critic rows must not be empty")
        if bool((factors[:, 0, 1].long() != KIND_SEP_CRITIC).any()):
            raise ValueError("critic rows must start with SEP_CRITIC")
        non_critic = valid & ~(
            (kind == KIND_SEP_CRITIC) | (kind == KIND_CRITIC_HAND) | (kind == KIND_CRITIC_FUTURE)
        )
        if bool(non_critic.any()):
            raise ValueError("critic rows contain a non-critic kind")
        return
    if bool((factors[:, 0, 1].long() != KIND_BOS).any()):
        raise ValueError("actor sequence must start with BOS")
    is_sep = (kind >= 101) & (kind <= 111)
    action_rows = valid & ~is_sep & (expected_seg == SEGMENT_ACTIONS)
    counts = action_rows.sum(dim=1)
    if bool(((counts == 0) | (counts % 2 != 0)).any()):
        raise ValueError("actor sequence must contain a positive even action count")


class KyokuTransformerActorCritic(nn.Module):
    """当前局面快照 Actor-Critic：Shared 双向 + Actor 结构化 + Critic 独立尾部。"""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig.preset("v18")
        self.token_embedding = StateTokenEmbedding(
            self.config.d_model,
            dense_slot_dim=self.config.dense_slot_dim,
            dense_fusion_dim=self.config.dense_fusion_dim,
            eps=self.config.eps,
        )
        self.value_query = nn.Parameter(torch.empty(self.config.d_model))
        self.public_backbone = Decoder(self.config, layers=self.config.shared_layers, final_norm=False)
        self.actor_backbone = Decoder(self.config, layers=self.config.layers - self.config.shared_layers)
        self.critic_backbone = Decoder(self.config, layers=self.config.critic_layers)
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
        actor_factors: Tensor,
        actor_numeric: Tensor,
        actor_lengths: Tensor,
        query_action_ids: Tensor,
        query_pair_counts: Tensor,
        legal_mask: Tensor,
    ) -> dict[str, Tensor]:
        return self.forward(
            actor_factors=actor_factors,
            actor_numeric=actor_numeric,
            actor_lengths=actor_lengths,
            query_action_ids=query_action_ids,
            query_pair_counts=query_pair_counts,
            legal_mask=legal_mask,
            policy_only=True,
        )

    def forward(
        self,
        actor_factors: Tensor,
        actor_numeric: Tensor,
        actor_lengths: Tensor,
        query_action_ids: Tensor,
        query_pair_counts: Tensor,
        legal_mask: Tensor,
        *,
        critic_factors: Tensor | None = None,
        critic_lengths: Tensor | None = None,
        detach_critic_public: bool = False,
        critic_public_grad_scale: float = 1.0,
        critic_private_embedding_grad_scale: float = 1.0,
        policy_only: bool = False,
    ) -> dict[str, Tensor]:
        if self.config.policy_head_type != "current_state_snapshot":
            raise ValueError("V18 forward requires current_state_snapshot")
        batch, actor_capacity, _width = actor_factors.shape
        if actor_numeric.shape != (batch, actor_capacity, TOKEN_NUMERIC_WIDTH):
            raise ValueError("actor_numeric must match [batch, tokens, 8]")
        if actor_lengths.shape != (batch,):
            raise ValueError("actor_lengths must have one entry per batch row")
        if query_action_ids.ndim != 2 or query_action_ids.shape[0] != batch:
            raise ValueError("query_action_ids must be [batch, pairs]")
        if query_pair_counts.shape != (batch,):
            raise ValueError("query_pair_counts must be [batch]")
        if legal_mask.shape != (batch, NUM_ACTIONS):
            raise ValueError(f"legal_mask must be [batch, {NUM_ACTIONS}]")
        if torch.any(actor_lengths < 0) or torch.any(actor_lengths > actor_capacity):
            raise ValueError("actor_lengths out of range")
        if torch.any(actor_lengths > self.config.context_tokens):
            raise ValueError("actor context overflow")
        _assert_structure(actor_factors, actor_lengths, critic=False)

        device = actor_factors.device
        actor_lengths = actor_lengths.to(device=device, dtype=torch.long)
        actor_embeddings = self.token_embedding(actor_factors, actor_numeric)

        # —— Shared 公共前缀（segment 1）双向 backbone ——
        shared_mask = actor_factors[..., 0].eq(SEGMENT_SHARED)
        shared_lengths = (shared_mask & (torch.arange(actor_capacity, device=device)[None] < actor_lengths[:, None])).sum(dim=1)
        shared_capacity = max(int(shared_lengths.max().item()), 1)
        shared_tokens = actor_embeddings[:, :shared_capacity]
        shared_mask_bool, shared_valid = _bidirectional_layout(shared_lengths, shared_capacity)
        shared_hidden = self.public_backbone(
            shared_tokens, shared_lengths, attention_mask=shared_mask_bool, valid=shared_valid,
        )

        # —— Actor 层：Shared 表示 + Analysis + Action Query 的完整序列 ——
        actor_input = actor_embeddings.clone()
        shared_positions = torch.arange(shared_capacity, device=device)[None, :]
        replace = shared_positions < shared_lengths[:, None]
        actor_input[:, :shared_capacity] = torch.where(
            replace[..., None], shared_hidden, actor_input[:, :shared_capacity],
        )
        actor_mask_bool, actor_valid = _actor_structured_layout(
            actor_factors[..., 0], actor_factors[..., 1], actor_lengths, actor_capacity,
        )
        actor_hidden = self.actor_backbone(
            actor_input, actor_lengths, attention_mask=actor_mask_bool, valid=actor_valid,
        )

        # —— 策略头：按 action query kind 定位 pair，映射到 action id ——
        action_kind_mask = actor_factors[..., 1].eq(11) | actor_factors[..., 1].eq(12)
        query_mask = action_kind_mask & (
            torch.arange(actor_capacity, device=device)[None] < actor_lengths[:, None]
        )
        pair_counts = query_pair_counts.to(device=device, dtype=torch.long)
        action_capacity = query_action_ids.shape[1]
        if torch.any(pair_counts < 0) or torch.any(pair_counts > action_capacity):
            raise ValueError("query_pair_counts out of range")
        # 整批向量化：一次 nonzero 取代逐行 torch.nonzero（每行一次 GPU→CPU 同步，
        # 旧实现 256 行/批造成大量同步开销）。nonzero 默认返回 CPU 张量，随后统一
        # 搬到设备上参与索引；结果按行主序，行内 O/D 相邻成对。
        flat_positions = torch.nonzero(query_mask).to(device=device)
        if flat_positions.shape[0] != 2 * int(pair_counts.sum()):
            raise ValueError("action query rows do not match query_pair_counts")
        if flat_positions.shape[0] > 0:
            row_indices = flat_positions[:, 0]
            col_indices = flat_positions[:, 1]
            offense_hidden = actor_hidden[row_indices[0::2], col_indices[0::2]]
            defense_hidden = actor_hidden[row_indices[1::2], col_indices[1::2]]
            pair_hiddens = self.action_fusion(
                torch.cat((offense_hidden, defense_hidden), dim=-1)
            )
            action_logits = self.policy_mlp(pair_hiddens).squeeze(-1).float()
        else:
            action_logits = actor_hidden.new_zeros((0,), dtype=torch.float32)

        raw = torch.zeros((batch, NUM_ACTIONS), dtype=torch.float32, device=device)
        if action_logits.shape[0] > 0:
            pair_row = torch.repeat_interleave(
                torch.arange(batch, device=device), pair_counts
            )
            starts = (pair_counts.cumsum(0) - pair_counts).repeat_interleave(pair_counts)
            within = torch.arange(action_logits.shape[0], device=device) - starts
            flat_ids = query_action_ids.to(device=device, dtype=torch.long)[pair_row, within]
            # 行内升序（canonical 契约保证），跨行重复属正常；只检测相邻同行的重复。
            if flat_ids.numel() > 0:
                same_row = pair_row[1:] == pair_row[:-1]
                if torch.any(same_row & (flat_ids[1:] == flat_ids[:-1])):
                    raise ValueError("query action ids must be unique (duplicate action id rejected)")
            raw.view(-1).scatter_add_(0, pair_row * NUM_ACTIONS + flat_ids, action_logits)
        logits = raw.masked_fill(~legal_mask.to(device=device, dtype=torch.bool), float("-inf"))
        output: dict[str, Tensor] = {"raw_policy_logits": raw, "policy_logits": logits}
        if policy_only:
            return output

        # —— Critic：Shared 表示 + 私有行 + Value Query ——
        if critic_factors is None:
            critic_factors = actor_factors.new_zeros((batch, 0, TOKEN_ROW_WIDTH))
        if critic_lengths is None:
            critic_lengths = critic_factors.ne(0).any(-1).long().sum(-1)
        if critic_factors.ndim != 3 or critic_factors.shape[-1] != TOKEN_ROW_WIDTH:
            raise ValueError(f"critic_factors must be [batch, tokens, {TOKEN_ROW_WIDTH}]")
        if critic_lengths.shape != (batch,):
            raise ValueError("critic_lengths must be [batch]")
        _assert_structure(critic_factors, critic_lengths, critic=True)
        critic_capacity = int(critic_factors.shape[1])
        if torch.any(critic_lengths > self.config.context_tokens - shared_lengths - 1):
            raise ValueError("critic context overflow")
        critic_embeddings = self.token_embedding(critic_factors, critic_factors.new_zeros((batch, critic_capacity, TOKEN_NUMERIC_WIDTH)))
        private_grad_scale = float(critic_private_embedding_grad_scale)
        if not 0.0 <= private_grad_scale <= 1.0:
            raise ValueError("critic_private_embedding_grad_scale must be in [0, 1]")
        if private_grad_scale == 0.0:
            critic_embeddings = critic_embeddings.detach()
        elif private_grad_scale != 1.0:
            detached_private = critic_embeddings.detach()
            critic_embeddings = detached_private + private_grad_scale * (
                critic_embeddings - detached_private
            )
        critic_total_lengths = shared_lengths + critic_lengths + 1
        critic_total = int(critic_total_lengths.max().item())
        critic_sequence = critic_embeddings.new_zeros((batch, critic_total, self.config.d_model))
        # 公共表示（可 detach/缩放）。
        public_grad_scale = 0.0 if detach_critic_public else float(critic_public_grad_scale)
        if not 0.0 <= public_grad_scale <= 1.0:
            raise ValueError("critic_public_grad_scale must be in [0, 1]")
        if public_grad_scale == 0.0:
            shared_for_critic = shared_hidden.detach()
        elif public_grad_scale == 1.0:
            shared_for_critic = shared_hidden
        else:
            detached = shared_hidden.detach()
            shared_for_critic = detached + public_grad_scale * (shared_hidden - detached)
        rows = torch.arange(batch, device=device)
        public_positions = torch.arange(shared_capacity, device=device)[None, :].expand(batch, -1)
        public_valid_mask = public_positions < shared_lengths[:, None]
        public_source_rows = rows[:, None].expand_as(public_positions)[public_valid_mask]
        critic_sequence[public_source_rows, public_positions[public_valid_mask]] = shared_for_critic[public_valid_mask]
        private_positions = shared_lengths[:, None] + torch.arange(critic_capacity, device=device)[None, :]
        private_valid_mask = torch.arange(critic_capacity, device=device)[None, :] < critic_lengths[:, None]
        private_source_rows = rows[:, None].expand_as(private_positions)[private_valid_mask]
        critic_sequence[private_source_rows, private_positions[private_valid_mask]] = critic_embeddings[private_valid_mask]
        value_indices = shared_lengths + critic_lengths
        critic_sequence[rows, value_indices] = self.value_query
        critic_mask_bool, critic_valid = _critic_layout(critic_total_lengths, critic_total)
        critic_hidden = self.critic_backbone(
            critic_sequence, critic_total_lengths, attention_mask=critic_mask_bool, valid=critic_valid,
        )[rows, value_indices]
        assert self.value_head is not None
        output["value"] = self.value_head(critic_hidden).squeeze(-1).float()
        output["critic_hidden"] = critic_hidden
        return output
