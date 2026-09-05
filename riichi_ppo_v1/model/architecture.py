"""V19 当前局面快照 GQA Actor-Critic。

Actor 输入为完整状态快照序列（Shared 公共前缀 + 三个 Opponent Analysis + 信念 token +
按 action ID 升序的 Offense/Defense Query），所有有效 token 使用连续 RoPE 位置；Shared
公共 backbone 用双向 GQA；Actor-only 层用结构化隔离 mask；Critic 在公共表示后拼接三家
真实闭手（独立尾部，不接收 Analysis/Action/Belief token）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .belief_network import BeliefNetwork
from .dense_embedding import StateTokenEmbedding
from .encoding_protocol import (
    CONTEXT_TOKENS,
    KIND_BELIEF,
    KIND_BOS,
    KIND_CRITIC_HAND,
    KIND_RIICHI_CARD,
    KIND_SEP_ACTIONS,
    KIND_SEP_CRITIC,
    SEGMENT_ACTIONS,
    SEGMENT_ANALYSIS,
    SEGMENT_BELIEF,
    SEGMENT_CRITIC_PRIVATE,
    SEGMENT_SHARED,
    TOKEN_NUMERIC_WIDTH,
    TOKEN_ROW_WIDTH,
)
from .schema import NUM_ACTIONS


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 5
    shared_layers: int = 3
    critic_layers: int = 1
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
    def preset(cls, size: str) -> ModelConfig:
        if size != "v19":
            raise ValueError("model size must be 'v19'")
        return cls()

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> ModelConfig:
        """从纯 V19 checkpoint/config 映射恢复精确拓扑。"""
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
        def shape(value: Tensor, heads: int) -> Tensor:
            return value.view(batch, tokens, heads, self.head_dim).transpose(1, 2)
        q_raw, k_raw, v_raw = self.qkv(x).split(
            (self.qh * self.head_dim, self.kvh * self.head_dim, self.kvh * self.head_dim), dim=-1
        )
        q, k, v = _rope(shape(q_raw, self.qh), rope), _rope(shape(k_raw, self.kvh), rope), shape(v_raw, self.kvh)
        repeat = self.qh // self.kvh
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        # 锁定 mem_efficient 后端(B3,仅 CUDA):结构化 bool mask 下 flash
        # 不可用,math 后端慢 5.2 倍(实测 78.8ms vs 15.1ms);锁定后约束不满足
        # 会 fail-fast 报错而非静默 5 倍变慢。当前生产默认即选中 mem_efficient,
        # 锁定不改变数值(单测 torch.equal 断言)。CPU 无 mem_efficient 内核,
        # 保持默认调度。
        if attention_mask.is_cuda:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                value = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_mask[:, None], dropout_p=0.0,
                )
        else:
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
    """根据 segment/kind 类别构造 Actor 结构化 mask（V19 信念可见性规则）。

    - Shared(segment 1，含共享分隔符) ↔ Shared 双向；
    - Analysis(segment 2 + SEP_ACTIONS 分隔符) → Shared ∪ Analysis；
    - Belief(segment 5) → Shared ∪ Belief（信念互见；不读 analysis/动作/padding）；
    - 每个 Action pair（kind 11/12 中相邻两行）→ Shared ∪ Analysis ∪ 本 pair ∪ Belief；
    - 不同 pair 互不可见；analysis 不读信念；padding 不可见。
    """
    device = segments.device
    positions = torch.arange(tokens, device=device)[None, :]
    valid = positions < lengths[:, None]
    is_shared = segments.eq(SEGMENT_SHARED)
    is_analysis = segments.eq(SEGMENT_ANALYSIS)
    is_sep_actions = segments.eq(SEGMENT_ACTIONS) & kinds.eq(KIND_SEP_ACTIONS)
    is_belief = segments.eq(SEGMENT_BELIEF)
    is_action = kinds.eq(11) | kinds.eq(12)
    action_mask = is_action & valid
    action_ranks = action_mask.long().cumsum(dim=1) - 1
    pair_id = torch.where(action_mask, action_ranks // 2, torch.zeros_like(action_ranks))
    same_pair = pair_id[:, :, None].eq(pair_id[:, None, :]) & is_action[:, :, None] & is_action[:, None, :]
    from_shared = is_shared[:, :, None] & is_shared[:, None, :]
    from_analysis = is_analysis[:, :, None] & (is_shared[:, None, :] | is_analysis[:, None, :])
    from_sep_actions = is_sep_actions[:, :, None] & is_sep_actions[:, None, :]
    from_belief = is_belief[:, :, None] & (is_shared[:, None, :] | is_belief[:, None, :])
    from_action = is_action[:, :, None] & (
        is_shared[:, None, :] | is_analysis[:, None, :] | is_sep_actions[:, None, :]
        | is_belief[:, None, :] | same_pair
    )
    mask = from_shared | from_analysis | from_sep_actions | from_belief | from_action
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
    """kind → 期望 segment 的整批查表（V19：1..15 与 101..111 之外的 kind 映射为 0）。

    等价于逐 token 的类别→segment 映射，纯向量化、无 per-row Python 循环。
    """
    is_sep = (kind >= 101) & (kind <= 111)
    shared = ((kind >= 1) & (kind <= 9)) | (kind == KIND_RIICHI_CARD) | (
        (kind >= 101) & (kind <= 108)
    )
    analysis = (kind == 10) | (kind == 109)
    actions = (kind == 11) | (kind == 12) | (kind == 110)
    critic_private = (kind == 13) | (kind == 111)
    belief = kind == KIND_BELIEF
    invalid = ~(shared | analysis | actions | critic_private | belief) & ~is_sep
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
                        torch.full_like(kind, SEGMENT_BELIEF),
                    ),
                ),
            ),
        ),
    )


def _assert_structure(factors: Tensor, lengths: Tensor, *, critic: bool) -> None:
    """轻量结构校验：段/类别一致性、BOS 开头、分隔符位置、action 数量（fail closed）。

    整批向量化实现（旧实现逐行 tolist + Python 循环，实测一次 forward 占 ~32ms）。
    """
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
            (kind == KIND_SEP_CRITIC) | (kind == KIND_CRITIC_HAND)
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
        self.config = config or ModelConfig.preset("v19")
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
        self.belief_network = BeliefNetwork(self.config.d_model)
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
        shared_capacity: int | None = None,
        kind_row_plan: dict[int, Any] | None = None,
        validate_structure: bool = True,
        belief_public_grad_scale: float = 1.0,
    ) -> dict[str, Tensor]:
        """policy-only 前向(冻结 SFT reference 预计算的唯一消费入口)。

        独立 code object:``torch_compile_reference`` 编译本方法而非整个模块,
        dynamo 按本函数独立维护缓存(默认 recompile_limit=8),不与训练模型
        ``forward`` 的缓存槽位互相挤占(两者若共享同一 code object 缓存,
        变体多的一方会把另一方挤出缓存、触发 eager 回退)。透传参数与
        ``forward(policy_only=True)`` 完全一致:host 预计算的
        ``shared_capacity``/``kind_row_plan`` 与 ``validate_structure`` 原样
        下传,保持免同步快速路径。
        """
        return self.forward(
            actor_factors=actor_factors,
            actor_numeric=actor_numeric,
            actor_lengths=actor_lengths,
            query_action_ids=query_action_ids,
            query_pair_counts=query_pair_counts,
            legal_mask=legal_mask,
            policy_only=True,
            shared_capacity=shared_capacity,
            kind_row_plan=kind_row_plan,
            validate_structure=validate_structure,
            belief_public_grad_scale=belief_public_grad_scale,
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
        validate_structure: bool = True,
        belief_public_grad_scale: float = 1.0,
        shared_capacity: int | None = None,
        critic_total_capacity: int | None = None,
        kind_row_plan: dict[int, Any] | None = None,
        critic_kind_row_plan: dict[int, Any] | None = None,
    ) -> dict[str, Tensor]:
        """V19 前向。

        ``validate_structure=False`` 时跳过全部 GPU 侧结构校验(训练期由 Rust
        编码器 fail-closed 生成 + SFT 契约校验 + 单测覆盖,重复校验每次
        forward 引入十余次 GPU→CPU 同步);默认 True 保持历史行为。

        ``shared_capacity``/``critic_total_capacity`` 为 host 侧预计算的容量
        (collate 在 numpy 里算好传入);为 None 时回退到 GPU ``max().item()``
        同步推导,保证其他调用方零改动。

        ``kind_row_plan``/``critic_kind_row_plan`` 为 host 侧类别行表
        (``dense_embedding.compute_kind_row_plan``,分别对应 actor/critic
        输入的展平行号);传入时 token_embedding 走静态键表路径,免去
        argsort/tolist 同步;None 时保持旧路径。

        ``belief_public_grad_scale`` 在 z_pool → encoder 边界实施与 critic
        相同的 detach+重标度 trick(0.0 完全 detach;1.0 原样;0.25 为训练分册
        定版耦合比),只缩放信念分支回传公共 backbone 的梯度,前向不变。
        """
        if self.config.policy_head_type != "current_state_snapshot":
            raise ValueError("V19 forward requires current_state_snapshot")
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
        if validate_structure:
            if torch.any(actor_lengths < 0) or torch.any(actor_lengths > actor_capacity):
                raise ValueError("actor_lengths out of range")
            if torch.any(actor_lengths > self.config.context_tokens):
                raise ValueError("actor context overflow")
            _assert_structure(actor_factors, actor_lengths, critic=False)

        device = actor_factors.device
        actor_lengths = actor_lengths.to(device=device, dtype=torch.long)
        actor_embeddings = self.token_embedding(
            actor_factors, actor_numeric, kind_row_plan=kind_row_plan,
        )

        # —— Shared 公共前缀（segment 1）双向 backbone ——
        shared_mask = actor_factors[..., 0].eq(SEGMENT_SHARED)
        shared_lengths = (
            shared_mask & (torch.arange(actor_capacity, device=device)[None] < actor_lengths[:, None])
        ).sum(dim=1)
        if shared_capacity is None:
            shared_capacity = max(int(shared_lengths.max().item()), 1)
        else:
            # host 侧 capacity:collate 在 numpy 里按同一语义(segment==SHARED 的
            # 有效行数)取批内最大;padding 行 segment 为 0 不污染计数。
            shared_capacity = max(int(shared_capacity), 1)
        shared_tokens = actor_embeddings[:, :shared_capacity]
        shared_mask_bool, shared_valid = _bidirectional_layout(shared_lengths, shared_capacity)
        shared_hidden = self.public_backbone(
            shared_tokens, shared_lengths, attention_mask=shared_mask_bool, valid=shared_valid,
        )

        # —— 信念网络：从公共表示生成五头预测与注入 token ——
        belief_grad_scale = float(belief_public_grad_scale)
        if not 0.0 <= belief_grad_scale <= 1.0:
            raise ValueError("belief_public_grad_scale must be in [0, 1]")
        if belief_grad_scale == 0.0:
            shared_for_belief = shared_hidden.detach()
        elif belief_grad_scale == 1.0:
            shared_for_belief = shared_hidden
        else:
            detached_belief = shared_hidden.detach()
            shared_for_belief = detached_belief + belief_grad_scale * (
                shared_hidden - detached_belief
            )
        belief = self.belief_network(shared_for_belief)
        belief_tokens = belief["belief_tokens"]

        # —— Actor 层：Shared 表示 + Analysis + Belief token + Action Query ——
        actor_input = actor_embeddings.clone()
        # bf16 autocast 下信念网络的 Linear 输出为 bf16,而 actor 序列张量是
        # float32;注入前统一到序列 dtype,避免 Index put dtype 不匹配。
        belief_tokens = belief_tokens.to(dtype=actor_input.dtype)
        shared_positions = torch.arange(shared_capacity, device=device)[None, :]
        replace = shared_positions < shared_lengths[:, None]
        actor_input[:, :shared_capacity] = torch.where(
            replace[..., None], shared_hidden, actor_input[:, :shared_capacity],
        )
        pair_counts = query_pair_counts.to(device=device, dtype=torch.long)
        action_capacity = query_action_ids.shape[1]
        if validate_structure:
            if torch.any(pair_counts < 0) or torch.any(pair_counts > action_capacity):
                raise ValueError("query_pair_counts out of range")
        # 信念块插入位：SEP_ACTIONS 行之后、第一对 O/D Query 之前。
        # 最后一个信念 token 距第一对 Query 恒距 1（输入分册 §6 不变式）。
        sep_pos = actor_lengths - 2 * pair_counts - 1
        aug_capacity = actor_capacity + 30
        aug_lengths = actor_lengths + 30
        aug_actor_input = actor_input.new_zeros((batch, aug_capacity, self.config.d_model))
        src_positions = torch.arange(actor_capacity, device=device)[None, :].expand(batch, -1)
        # 插入后：sep_pos 及之前的行位置不变，其后所有行整体后移 30（信念块占位）。
        dst_positions = torch.where(
            src_positions <= sep_pos[:, None], src_positions, src_positions + 30,
        )
        batch_rows = torch.arange(batch, device=device)[:, None]
        aug_actor_input[
            batch_rows.expand(batch, actor_capacity).reshape(-1),
            dst_positions.reshape(-1),
        ] = actor_input.reshape(batch * actor_capacity, self.config.d_model)
        belief_slots = torch.arange(30, device=device)[None, :]
        belief_dst = sep_pos[:, None] + 1 + belief_slots
        aug_actor_input[
            batch_rows.expand(batch, 30).reshape(-1),
            belief_dst.reshape(-1),
        ] = belief_tokens.reshape(batch * 30, self.config.d_model)

        # 增广 segment/kind：原序列行按插入映射回填，信念行填 SEGMENT_BELIEF/KIND_BELIEF。
        aug_segments = actor_factors.new_zeros((batch, aug_capacity))
        aug_kinds = actor_factors.new_zeros((batch, aug_capacity))
        aug_segments[
            batch_rows.expand(batch, actor_capacity).reshape(-1),
            dst_positions.reshape(-1),
        ] = actor_factors[..., 0].reshape(-1)
        aug_kinds[
            batch_rows.expand(batch, actor_capacity).reshape(-1),
            dst_positions.reshape(-1),
        ] = actor_factors[..., 1].reshape(-1)
        aug_segments[batch_rows.expand(batch, 30).reshape(-1), belief_dst.reshape(-1)] = SEGMENT_BELIEF
        aug_kinds[batch_rows.expand(batch, 30).reshape(-1), belief_dst.reshape(-1)] = KIND_BELIEF
        actor_mask_bool, actor_valid = _actor_structured_layout(
            aug_segments, aug_kinds, aug_lengths, aug_capacity,
        )
        actor_hidden = self.actor_backbone(
            aug_actor_input, aug_lengths, attention_mask=actor_mask_bool, valid=actor_valid,
        )

        # —— 策略头：按 action query kind 定位 pair，映射到 action id ——
        action_kind_mask = aug_kinds.eq(11) | aug_kinds.eq(12)
        query_mask = action_kind_mask & (
            torch.arange(aug_capacity, device=device)[None] < aug_lengths[:, None]
        )
        if validate_structure:
            # canonical 契约:action query 行为增广序列尾部连续 2×pair_count 行
            # (O 先 D 后相邻成对;来源:Rust 编码器 fail-closed 构造 +
            # test_v19_architecture 的契约断言)。校验路径比旧实现更严:旧实现
            # 只核对数量,这里核对整个 tail 窗口位置一一对应。
            tail_window = (
                torch.arange(aug_capacity, device=device)[None]
                >= aug_lengths[:, None] - 2 * pair_counts[:, None]
            ) & (
                torch.arange(aug_capacity, device=device)[None] < aug_lengths[:, None]
            )
            if bool((query_mask != tail_window).any()):
                raise ValueError(
                    "action query rows are not the canonical contiguous tail window"
                )
        # 算术索引取代 torch.nonzero/repeat_interleave:两者输出形状依赖数据,
        # 必然 GPU→CPU 同步,且是 torch.compile fullgraph 的断裂源。无效槽位
        # clamp 到界内取占位 hidden,logits 以 where 置 0,scatter 加 0 无影响
        # (x + 0.0 == x,softmax/log 对 ±0.0 亦不变)。tail 窗口/offense 列均
        # 按增广长度计算;增广序列最后 2×pair_count 行仍为 O/D 对。
        pair_index = torch.arange(action_capacity, device=device)
        valid_pair = pair_index[None, :] < pair_counts[:, None]
        offense_cols = (
            aug_lengths[:, None] - 2 * pair_counts[:, None] + 2 * pair_index[None, :]
        )
        col_upper = max(int(aug_capacity) - 1, 0)
        safe_cols = offense_cols.clamp(0, col_upper)
        rows_matrix = torch.arange(batch, device=device)[:, None].expand(
            batch, action_capacity
        )
        offense_hidden = actor_hidden[rows_matrix, safe_cols]
        defense_hidden = actor_hidden[rows_matrix, (safe_cols + 1).clamp(0, col_upper)]
        pair_hiddens = self.action_fusion(
            torch.cat((offense_hidden, defense_hidden), dim=-1)
        )
        action_logits = self.policy_mlp(pair_hiddens).squeeze(-1).float()
        action_logits = torch.where(
            valid_pair, action_logits, action_logits.new_zeros(()),
        )
        flat_ids = query_action_ids.to(device=device, dtype=torch.long)
        if validate_structure and action_capacity > 1:
            # 行内升序(canonical 契约保证),相邻重复即非法;跨行重复属正常。
            duplicate = valid_pair[:, 1:] & (flat_ids[:, 1:] == flat_ids[:, :-1])
            if bool(duplicate.any()):
                raise ValueError(
                    "query action ids must be unique (duplicate action id rejected)"
                )
        raw = torch.zeros((batch, NUM_ACTIONS), dtype=torch.float32, device=device)
        targets = rows_matrix * NUM_ACTIONS + flat_ids
        raw.view(-1).scatter_add_(0, targets.reshape(-1), action_logits.reshape(-1))
        logits = raw.masked_fill(~legal_mask.to(device=device, dtype=torch.bool), float("-inf"))
        output: dict[str, Tensor] = {
            "raw_policy_logits": raw,
            "policy_logits": logits,
            "belief_hand_logits": belief["belief_hand_logits"],
            "belief_shanten_logits": belief["belief_shanten_logits"],
            "belief_wait_logits": belief["belief_wait_logits"],
            "belief_danger_logits": belief["belief_danger_logits"],
            "belief_loss_pred": belief["belief_loss_pred"],
            "belief_tokens": belief_tokens,
        }
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
        if validate_structure:
            _assert_structure(critic_factors, critic_lengths, critic=True)
        critic_capacity = int(critic_factors.shape[1])
        if validate_structure and torch.any(
            critic_lengths > self.config.context_tokens - shared_lengths - 1
        ):
            raise ValueError("critic context overflow")
        critic_embeddings = self.token_embedding(
            critic_factors,
            critic_factors.new_zeros((batch, critic_capacity, TOKEN_NUMERIC_WIDTH)),
            kind_row_plan=critic_kind_row_plan,
        )
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
        if critic_total_capacity is None:
            critic_total = int(critic_total_lengths.max().item())
        else:
            # host 侧 capacity:collate 按 max(shared_len + critic_len + 1) 预计算。
            critic_total = int(critic_total_capacity)
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
        value_indices = shared_lengths + critic_lengths
        public_positions = torch.arange(shared_capacity, device=device)[None, :].expand(batch, -1)
        # 算术 gather/scatter 取代布尔掩码索引:布尔变长索引内部 nonzero 强制
        # GPU→CPU 同步,且是 torch.compile 图断裂源(同文件策略头同做法)。
        # 目标非法槽位 clamp 到该行 value 位置(随后被 value_query 覆盖,与旧
        # 路径「先写 shared/private 再覆盖 value 行」的最终结果一致);源非法
        # 槽位 clamp 到该行最后一个有效 shared 列(读值不影响任何有效行)。
        public_valid = public_positions < shared_lengths[:, None]
        safe_public_dst = torch.where(public_valid, public_positions, value_indices[:, None])
        safe_public_src = torch.where(public_valid, public_positions, (shared_lengths - 1).clamp_min(0)[:, None])
        public_rows = rows[:, None].expand_as(public_positions)
        critic_sequence[public_rows.reshape(-1), safe_public_dst.reshape(-1)] = shared_for_critic[
            public_rows.reshape(-1), safe_public_src.reshape(-1)
        ]
        private_slots = torch.arange(critic_capacity, device=device)[None, :]
        private_positions = shared_lengths[:, None] + private_slots
        private_valid = private_slots < critic_lengths[:, None]
        safe_private_dst = torch.where(private_valid, private_positions, value_indices[:, None])
        safe_private_src = torch.where(private_valid, private_slots, (critic_lengths - 1).clamp_min(0)[:, None])
        private_rows = rows[:, None].expand_as(private_positions)
        critic_sequence[private_rows.reshape(-1), safe_private_dst.reshape(-1)] = critic_embeddings[
            private_rows.reshape(-1), safe_private_src.reshape(-1)
        ]
        critic_sequence[rows, value_indices] = self.value_query
        critic_mask_bool, critic_valid = _critic_layout(critic_total_lengths, critic_total)
        critic_hidden = self.critic_backbone(
            critic_sequence, critic_total_lengths, attention_mask=critic_mask_bool, valid=critic_valid,
        )[rows, value_indices]
        assert self.value_head is not None
        output["value"] = self.value_head(critic_hidden).squeeze(-1).float()
        output["critic_hidden"] = critic_hidden
        return output
