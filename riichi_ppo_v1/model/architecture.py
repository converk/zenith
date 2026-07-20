"""Semantic-token GQA actor-critic with a fixed 241-action policy head."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

NUM_ACTIONS = 241
TOKEN_WIDTH = 10
NUMERIC_WIDTH = 8
TOKEN_CARDINALITIES = (8, 32, 256, 8, 8, 16, 4, 16, 256, 4)


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 6
    shared_layers: int = 4
    critic_layers: int = 2
    d_model: int = 192
    query_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 24
    ffn_dim: int = 576
    context_tokens: int = 4096
    rope_base: float = 10_000.0
    eps: float = 1e-6

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

    @classmethod
    def preset(cls, size: str) -> "ModelConfig":
        configs = {
            "mid": cls(),
            "large": cls(layers=12, d_model=384, query_heads=12, kv_heads=3, head_dim=32, ffn_dim=1152),
        }
        try:
            return configs[size]
        except KeyError as exc:
            raise ValueError("model size must be 'mid' or 'large'") from exc


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


def _rope_values(tokens: int, head_dim: int, device: torch.device, dtype: torch.dtype, base: float) -> tuple[Tensor, Tensor]:
    positions = torch.arange(tokens, device=device, dtype=torch.float32)
    frequencies = torch.exp(torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) * (-math.log(base) / head_dim))
    angle = positions[:, None] * frequencies[None]
    return angle.cos().to(dtype), angle.sin().to(dtype)


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

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        tokens = x.shape[1]
        if tokens > self.context_tokens:
            raise ValueError(f"context overflow: {tokens} > {self.context_tokens}")
        rope = _rope_values(tokens, x.shape[-1] // self.blocks[0].attention.qh, x.device, x.dtype, self.rope_base)
        attention_mask, valid = _attention_layout(lengths, tokens)
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
        self.query = nn.Parameter(torch.empty(self.config.d_model))
        self.value_query = nn.Parameter(torch.empty(self.config.d_model))
        # The policy still traverses ``layers`` public blocks, while value
        # learning trains the shared prefix directly.  The aggregate block
        # budget remains unchanged: shared + actor-only + critic.
        self.public_backbone = Decoder(self.config, layers=self.config.shared_layers, final_norm=False)
        self.actor_backbone = Decoder(self.config, layers=self.config.layers - self.config.shared_layers)
        self.critic_backbone = Decoder(self.config, layers=self.config.critic_layers)
        self.policy_head = nn.Linear(self.config.d_model, NUM_ACTIONS)
        self.value_head = nn.Linear(self.config.d_model, 1)
        nn.init.normal_(self.query, std=self.config.d_model ** -0.5)
        nn.init.normal_(self.value_query, std=self.config.d_model ** -0.5)

    def forward(
        self,
        token_factors: Tensor,
        token_numeric: Tensor,
        legal_mask: Tensor | None = None,
        token_lengths: Tensor | None = None,
        *,
        critic_factors: Tensor | None = None,
        critic_lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if token_factors.shape[1] + 1 > self.config.context_tokens:
            raise ValueError(f"context overflow: {token_factors.shape[1] + 1} > {self.config.context_tokens}")
        if token_lengths is None:
            token_lengths = token_factors.ne(0).any(-1).long().sum(-1)
        token_lengths = token_lengths.to(device=token_factors.device, dtype=torch.long)
        if token_lengths.shape != (token_factors.shape[0],):
            raise ValueError("token_lengths must have one entry per batch row")
        if torch.any(token_lengths < 0) or torch.any(token_lengths > token_factors.shape[1]):
            raise ValueError("token_lengths exceed supplied token rows")
        tokens = self.token_embedding(token_factors, token_numeric)
        batch, padded, width = tokens.shape
        sequence = tokens.new_zeros((batch, padded + 1, width))
        sequence[:, :padded] = tokens
        rows = torch.arange(batch, device=tokens.device)
        sequence[rows, token_lengths] = self.query
        public_lengths = token_lengths + 1
        public_sequence = self.public_backbone(sequence, public_lengths)
        hidden = self.actor_backbone(public_sequence, token_lengths + 1)[rows, token_lengths]
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
        public_positions = torch.arange(padded + 1, device=tokens.device)[None, :].expand(batch, -1)
        public_valid = public_positions < public_lengths[:, None]
        critic_sequence[
            rows[:, None].expand_as(public_positions)[public_valid], public_positions[public_valid]
        ] = public_sequence[public_valid]
        private_positions = public_lengths[:, None] + torch.arange(
            critic_private.shape[1], device=tokens.device
        )[None, :]
        private_valid = torch.arange(critic_private.shape[1], device=tokens.device)[None, :] < critic_lengths[:, None]
        critic_sequence[rows[:, None].expand_as(private_positions)[private_valid], private_positions[private_valid]] = critic_private[private_valid]
        value_indices = public_lengths + critic_lengths
        critic_sequence[rows, value_indices] = self.value_query
        critic_hidden = self.critic_backbone(critic_sequence, critic_sequence_lengths)[rows, value_indices]
        # Autocast makes the preceding matrix multiplies BF16, but policy
        # probabilities and value estimates feed PPO's numerically sensitive
        # ratio/loss path.  Promote them before leaving the model, as in
        # exp/training's ActorCritic.
        raw = self.policy_head(hidden).float()
        if legal_mask is None:
            logits = raw
        else:
            if legal_mask.shape != (batch, NUM_ACTIONS):
                raise ValueError("legal_mask must be [batch, 241]")
            logits = raw.masked_fill(~legal_mask.to(device=raw.device, dtype=torch.bool), float("-inf"))
        return {
            "raw_policy_logits": raw,
            "policy_logits": logits,
            "value": self.value_head(critic_hidden).squeeze(-1).float(),
        }
