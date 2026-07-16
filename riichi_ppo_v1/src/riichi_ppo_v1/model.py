"""V5 semantic-token GQA actor-critic with a fixed 241-action policy head."""

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
TOKEN_SCHEMA_VERSION = 5
MODEL_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 6
    d_model: int = 256
    query_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 32
    ffn_dim: int = 768
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
    """Summed categorical factors plus a continuous Fourier feature channel."""

    def __init__(self, cardinalities: tuple[int, ...], d_model: int, numeric_dim: int) -> None:
        super().__init__()
        offsets, end = [], 0
        for size in cardinalities:
            if size < 1:
                raise ValueError("factor cardinalities must be positive")
            offsets.append(end)
            end += size - 1
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.table = nn.Embedding(end + 1, d_model, padding_idx=0)
        self.numeric = nn.Linear(numeric_dim, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)
        nn.init.normal_(self.table.weight, std=1.0 / math.sqrt(d_model))
        with torch.no_grad():
            self.table.weight[0].zero_()

    def forward(self, factors: Tensor, numeric: Tensor) -> Tensor:
        if factors.ndim != 3 or factors.shape[-1] != self.offsets.numel():
            raise ValueError("token_factors must be [batch, tokens, 10]")
        if numeric.shape != (*factors.shape[:2], NUMERIC_WIDTH):
            raise ValueError("token_numeric must match token_factors with width 8")
        factors = factors.long()
        indices = torch.where(factors.eq(0), 0, factors + self.offsets)
        active = factors.ne(0).sum(-1, keepdim=True).clamp_min(1)
        embedded = self.table(indices).sum(-2) / active.sqrt()
        return self.norm(embedded + self.numeric(numeric.float()))


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
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.context_tokens, self.rope_base = config.context_tokens, config.rope_base
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.layers))
        self.norm = nn.RMSNorm(config.d_model, eps=config.eps)

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
    """Public V5 actor/value model with a learned query and masked 241-head."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig.preset("mid")
        self.token_embedding = FactorEmbedding(TOKEN_CARDINALITIES, self.config.d_model, NUMERIC_WIDTH)
        self.query = nn.Parameter(torch.empty(self.config.d_model))
        self.backbone = Decoder(self.config)
        self.policy_head = nn.Linear(self.config.d_model, NUM_ACTIONS)
        self.value_head = nn.Linear(self.config.d_model, 1)
        nn.init.normal_(self.query, std=self.config.d_model ** -0.5)

    def forward(self, token_factors: Tensor, token_numeric: Tensor, legal_mask: Tensor | None = None, token_lengths: Tensor | None = None) -> dict[str, Tensor]:
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
        hidden = self.backbone(sequence, token_lengths + 1)[rows, token_lengths]
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
            "value": self.value_head(hidden).squeeze(-1).float(),
        }
