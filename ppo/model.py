from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ToyPPOConfig:
    d_model: int = 384
    num_layers: int = 8
    num_heads: int = 12
    ffn_hidden_dim: int = 1152
    policy_hidden_dim: int = 1536
    value_hidden_dim: int = 1536
    num_tile_types: int = 34
    count_vocab_size: int = 5
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")


def make_toy_ppo_config(model_size: str) -> ToyPPOConfig:
    """Builds a named model size for quick capacity comparisons."""
    configs = {
        "small": ToyPPOConfig(
            d_model=128,
            num_layers=2,
            num_heads=4,
            ffn_hidden_dim=384,
            policy_hidden_dim=512,
            value_hidden_dim=512,
        ),
        "medium": ToyPPOConfig(
            d_model=256,
            num_layers=4,
            num_heads=8,
            ffn_hidden_dim=768,
            policy_hidden_dim=1024,
            value_hidden_dim=1024,
        ),
        "large": ToyPPOConfig(),
    }
    try:
        return configs[model_size]
    except KeyError as exc:
        choices = ", ".join(sorted(configs))
        raise ValueError(f"model_size must be one of: {choices}") from exc


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(-2)
        position_ids = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("l,d->ld", position_ids, self.inv_freq)
        freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        cos = freqs.cos().to(dtype=x.dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype=x.dtype)[None, None, :, :]
        return (x * cos) + (self._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: ToyPPOConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden_dim)
        self.up_proj = nn.Linear(config.d_model, config.ffn_hidden_dim)
        self.down_proj = nn.Linear(config.ffn_hidden_dim, config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FullSelfAttentionWithRoPE(nn.Module):
    def __init__(self, config: ToyPPOConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.rope = RotaryPositionEmbedding(config.head_dim, config.rope_base)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x), batch_size, seq_len)
        k = self._split_heads(self.k_proj(x), batch_size, seq_len)
        v = self._split_heads(self.v_proj(x), batch_size, seq_len)

        # RoPE is disabled for tile-count tokens. Tile identity is already
        # represented by tile_embedding, so positional rotation is optional.
        # q = self.rope(q)
        # k = self.rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_dropout(self.out_proj(context))

    def _split_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: ToyPPOConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = FullSelfAttentionWithRoPE(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLUFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class TileCountEmbedding(nn.Module):
    def __init__(self, config: ToyPPOConfig) -> None:
        super().__init__()
        self.config = config
        self.tile_embedding = nn.Embedding(config.num_tile_types, config.d_model)
        self.count_embedding = nn.Embedding(config.count_vocab_size, config.d_model)
        tile_ids = torch.arange(config.num_tile_types, dtype=torch.long)
        self.register_buffer("tile_ids", tile_ids, persistent=False)

    def forward(self, hand_counts: Tensor) -> Tensor:
        if hand_counts.dim() != 2 or hand_counts.size(1) != self.config.num_tile_types:
            raise ValueError("hand_counts must have shape [B, 34]")
        counts = hand_counts.long().clamp(0, self.config.count_vocab_size - 1)
        tile_ids = self.tile_ids.expand(hand_counts.size(0), -1)
        return self.tile_embedding(tile_ids) + self.count_embedding(counts)


class TileCountTransformerActorCritic(nn.Module):
    def __init__(self, config: ToyPPOConfig | None = None) -> None:
        super().__init__()
        self.config = config or ToyPPOConfig()
        self.embedding = TileCountEmbedding(self.config)
        self.decision_token = nn.Parameter(torch.empty(1, 1, self.config.d_model))
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(self.config) for _ in range(self.config.num_layers)]
        )
        self.policy_head = nn.Sequential(
            RMSNorm(self.config.d_model, self.config.rms_norm_eps),
            nn.Linear(self.config.d_model, self.config.policy_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.policy_hidden_dim, self.config.num_tile_types),
        )
        self.value_head = nn.Sequential(
            RMSNorm(self.config.d_model, self.config.rms_norm_eps),
            nn.Linear(self.config.d_model, self.config.value_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.value_hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.decision_token, mean=0.0, std=0.02)

    def forward(
        self,
        hand_counts: Tensor,
        legal_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        x = self.embedding(hand_counts)
        decision = self.decision_token.expand(x.size(0), -1, -1)
        x = torch.cat([x, decision], dim=1)

        for layer in self.layers:
            x = layer(x)

        h_decision = x[:, -1, :]
        raw_policy_logits = self.policy_head(h_decision)
        policy_logits = raw_policy_logits
        if legal_mask is not None:
            legal_mask = legal_mask.to(device=policy_logits.device, dtype=torch.bool)
            policy_logits = policy_logits.masked_fill(
                ~legal_mask, torch.finfo(policy_logits.dtype).min
            )
        value = self.value_head(h_decision).squeeze(-1)
        return {
            "policy_logits": policy_logits,
            "raw_policy_logits": raw_policy_logits,
            "value": value,
        }


def make_legal_mask(hand_counts: Tensor) -> Tensor:
    return hand_counts > 0
