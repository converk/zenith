"""Causal Transformer actor-critic for KyokuEventTuple V3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

NUM_ACTIONS = 241
TOKEN_DIM = 8


@dataclass(frozen=True)
class ModelConfig:
    d_embed: int
    d_model: int
    layers: int
    heads: int
    ffn_dim: int
    rope_base: float = 10_000.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.d_model % self.heads or (self.d_model // self.heads) % 2:
            raise ValueError("d_model / heads must be an even integer for RoPE")

    @classmethod
    def preset(cls, size: str) -> "ModelConfig":
        configs = {
            "mid": cls(192, 192, 8, 6, 512),
            "large": cls(256, 384, 12, 12, 1152),
        }
        try:
            return configs[size]
        except KeyError as exc:
            raise ValueError("model size must be 'mid' or 'large'") from exc


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        freqs = torch.einsum("bl,d->bld", positions.to(self.inv_freq.dtype), self.inv_freq)
        angles = torch.repeat_interleave(freqs, 2, dim=-1).unsqueeze(1).to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        return x * angles.cos() + rotated * angles.sin()


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.d_model // config.heads
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RoPE(self.head_dim, config.rope_base)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(t: Tensor) -> Tensor:
            return t.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        q, k = self.rope(q, positions), self.rope(k, positions)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(batch, length, width))

    def forward_cached(
        self, x: Tensor, positions: Tensor, past: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Incremental causal attention for a uniform-length KV prefix.

        ``past`` contains RoPE-rotated K/V for confirmed event tokens only.
        Callers deliberately never retain snapshot or decision-token K/V.
        """
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(t: Tensor) -> Tensor:
            return t.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        q, k = self.rope(q, positions), self.rope(k, positions)
        if past is None:
            keys, values, prefix = k, v, 0
        else:
            past_k, past_v = past
            if past_k.shape[:2] != (batch, self.heads) or past_k.shape != past_v.shape:
                raise ValueError("cached K/V shape is incompatible with the input batch")
            keys, values, prefix = torch.cat((past_k, k), dim=2), torch.cat((past_v, v), dim=2), past_k.shape[2]
        # Query i may see every prefix key and current keys 0..i.
        causal = torch.arange(length, device=x.device).unsqueeze(1) >= (
            torch.arange(prefix + length, device=x.device).unsqueeze(0) - prefix
        )
        y = F.scaled_dot_product_attention(q, keys, values, attn_mask=causal, dropout_p=0.0, is_causal=False)
        return self.out(y.transpose(1, 2).reshape(batch, length, width)), (keys, values)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.up = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), positions)
        return x + self.ffn(self.ffn_norm(x))

    def forward_cached(
        self, x: Tensor, positions: Tensor, past: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attention, cache = self.attn.forward_cached(self.attn_norm(x), positions, past)
        x = x + attention
        return x + self.ffn(self.ffn_norm(x)), cache


class EventEmbedding(nn.Module):
    """The eight protocol fields have independent vocabularies and embeddings."""

    VOCABS = (48, 5, 5, 39, 39, 39, 19, 32)

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.tables = nn.ModuleList(
            nn.Embedding(vocab, dim, padding_idx=0 if index == 0 else None)
            for index, vocab in enumerate(self.VOCABS)
        )

    def forward(self, ids: Tensor) -> Tensor:
        if ids.ndim != 3 or ids.shape[-1] != TOKEN_DIM:
            raise ValueError("input_ids must be [batch, length, 8]")
        return sum(table(ids[..., field].long()) for field, table in enumerate(self.tables))


class KyokuTransformerActorCritic(nn.Module):
    """Masked 241-action actor-critic over a single decision sequence."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig.preset("mid")
        self.embedding = EventEmbedding(self.config.d_embed)
        self.input_proj: nn.Module = (
            nn.Identity() if self.config.d_embed == self.config.d_model
            else nn.Linear(self.config.d_embed, self.config.d_model, bias=False)
        )
        self.decision_token = nn.Parameter(torch.empty(self.config.d_model))
        self.blocks = nn.ModuleList(Block(self.config) for _ in range(self.config.layers))
        self.norm = RMSNorm(self.config.d_model, self.config.eps)
        self.policy_head = nn.Linear(self.config.d_model, NUM_ACTIONS)
        self.value_head = nn.Linear(self.config.d_model, 1)
        nn.init.normal_(self.decision_token, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        legal_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        sequence_lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        attention_mask, lengths = self._metadata(input_ids, attention_mask, sequence_lengths)
        x = self.input_proj(self.embedding(input_ids))
        batch, length, width = x.shape
        extended = x.new_zeros((batch, length + 1, width))
        extended[:, :length] = x
        row = torch.arange(batch, device=x.device)
        extended[row, lengths] = self.decision_token
        positions = torch.arange(length + 1, device=x.device).expand(batch, -1)
        for block in self.blocks:
            extended = block(extended, positions)
        hidden = self.norm(extended[row, lengths])
        raw = self.policy_head(hidden)
        if legal_mask is None:
            logits = raw
        else:
            if legal_mask.shape != (batch, NUM_ACTIONS):
                raise ValueError("legal_mask must be [batch, 241]")
            legal_mask = legal_mask.to(device=raw.device, dtype=torch.bool)
            if not legal_mask.any(dim=-1).all():
                raise ValueError("every policy row must contain a legal action")
            logits = raw.masked_fill(~legal_mask, float("-inf"))
        return {"raw_policy_logits": raw, "policy_logits": logits, "value": self.value_head(hidden).squeeze(-1)}

    def forward_cached(
        self,
        history_ids: Tensor,
        snapshot_ids: Tensor,
        legal_mask: Tensor,
        past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None,
        history_length: int = 0,
    ) -> tuple[dict[str, Tensor], tuple[tuple[Tensor, Tensor], ...]]:
        """Run one rollout decision with a cache of confirmed event history.

        Inputs are unpadded and have a uniform length within their batch.  The
        returned cache contains only ``history_ids`` appended to ``past``;
        snapshot state and the learned decision token remain temporary.
        """
        if history_ids.ndim != 3 or snapshot_ids.ndim != 3 or history_ids.shape[0] != snapshot_ids.shape[0]:
            raise ValueError("history_ids and snapshot_ids must be [batch, length, 8]")
        if history_ids.shape[-1] != TOKEN_DIM or snapshot_ids.shape[-1] != TOKEN_DIM:
            raise ValueError("cached inputs must use eight-field tokens")
        batch = history_ids.shape[0]
        if legal_mask.shape != (batch, NUM_ACTIONS):
            raise ValueError("legal_mask must be [batch, 241]")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must have one entry per Transformer block")
        cache = past_key_values
        if history_ids.shape[1]:
            history = self.input_proj(self.embedding(history_ids))
            positions = torch.arange(history_length, history_length + history.shape[1], device=history.device).expand(batch, -1)
            next_cache: list[tuple[Tensor, Tensor]] = []
            for layer, block in enumerate(self.blocks):
                history, layer_cache = block.forward_cached(history, positions, None if cache is None else cache[layer])
                next_cache.append(layer_cache)
            cache = tuple(next_cache)
        if cache is None:
            # A decision always has a snapshot, but retain a clear failure for
            # malformed callers rather than silently using an empty prefix.
            raise ValueError("cached forward requires a non-empty confirmed history")
        snapshot = self.input_proj(self.embedding(snapshot_ids))
        decision = self.decision_token.view(1, 1, -1).expand(batch, 1, -1)
        suffix = torch.cat((snapshot, decision), dim=1)
        positions = torch.arange(history_length + history_ids.shape[1], history_length + history_ids.shape[1] + suffix.shape[1], device=suffix.device).expand(batch, -1)
        for layer, block in enumerate(self.blocks):
            suffix, _temporary = block.forward_cached(suffix, positions, cache[layer])
        hidden = self.norm(suffix[:, -1])
        raw = self.policy_head(hidden)
        mask = legal_mask.to(device=raw.device, dtype=torch.bool)
        if not mask.any(dim=-1).all():
            raise ValueError("every policy row must contain a legal action")
        return {
            "raw_policy_logits": raw,
            "policy_logits": raw.masked_fill(~mask, float("-inf")),
            "value": self.value_head(hidden).squeeze(-1),
        }, cache

    @staticmethod
    def _metadata(
        ids: Tensor, attention: Tensor | None, lengths: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if ids.ndim != 3 or ids.shape[-1] != TOKEN_DIM:
            raise ValueError("input_ids must be [batch, length, 8]")
        batch, length, _ = ids.shape
        if attention is None:
            attention = ids[..., 0].ne(0)
        attention = attention.to(device=ids.device, dtype=torch.bool)
        if attention.shape != (batch, length):
            raise ValueError("attention_mask must be [batch, length]")
        inferred = attention.long().sum(-1)
        expected = torch.arange(length, device=ids.device).unsqueeze(0) < inferred.unsqueeze(1)
        if not torch.equal(attention, expected) or (inferred == 0).any():
            raise ValueError("attention_mask must be non-empty right-side padding")
        if lengths is None:
            lengths = inferred
        lengths = lengths.to(device=ids.device, dtype=torch.long)
        if lengths.shape != (batch,) or not torch.equal(lengths, inferred):
            raise ValueError("sequence_lengths must equal attention token counts")
        return attention, lengths
