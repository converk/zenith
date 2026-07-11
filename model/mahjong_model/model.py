"""MJAI 九维事件序列的 Transformer PPO actor-critic 模型。

模型接收状态机生成的 ``input_ids[B, L, 9]``。每个九维字段使用独立 embedding，
九个向量相加后投影到 Transformer 主干；RoPE 在每层 full attention 的 Q/K 上编码
事件的相对位置。模型输出 KyokuActionSpace V2 的 241 维策略 logits 与一个价值估计，
可直接用于现有 PPO 的 actor-critic 接口。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


NUM_ACTIONS = 241
TOKEN_DIM = 9


@dataclass(frozen=True)
class MahjongModelConfig:
    """Transformer 宽度与层数配置。"""

    d_embed: int
    d_model: int
    num_layers: int
    num_heads: int
    ffn_hidden_dim: int
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")


def make_mahjong_model_config(model_size: str) -> MahjongModelConfig:
    """Builds one of the documented mid/large model sizes."""

    configs = {
        "mid": MahjongModelConfig(
            d_embed=256,
            d_model=320,
            num_layers=10,
            num_heads=10,
            ffn_hidden_dim=800,
        ),
        "large": MahjongModelConfig(
            d_embed=256,
            d_model=384,
            num_layers=12,
            num_heads=12,
            ffn_hidden_dim=1152,
        ),
    }
    try:
        return configs[model_size]
    except KeyError as exc:
        choices = ", ".join(sorted(configs))
        raise ValueError(f"model_size must be one of: {choices}") from exc


class RMSNorm(nn.Module):
    """RMS normalization without mean centering."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Applies RoPE to Q/K tensors shaped ``[B, H, L, head_dim]``."""

    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, position_ids: Tensor) -> Tensor:
        frequencies = torch.einsum(
            "bl,d->bld",
            position_ids.to(dtype=self.inv_freq.dtype),
            self.inv_freq,
        )
        angles = torch.repeat_interleave(frequencies, repeats=2, dim=-1)
        cos = angles.cos().to(dtype=x.dtype).unsqueeze(1)
        sin = angles.sin().to(dtype=x.dtype).unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: MahjongModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden_dim)
        self.up_proj = nn.Linear(config.d_model, config.ffn_hidden_dim)
        self.down_proj = nn.Linear(config.ffn_hidden_dim, config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FullSelfAttentionWithRoPE(nn.Module):
    def __init__(self, config: MahjongModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.rope = RotaryPositionEmbedding(config.head_dim, config.rope_base)

    def forward(self, x: Tensor, attention_mask: Tensor, position_ids: Tensor) -> Tensor:
        batch_size, sequence_length, _ = x.shape
        q = self._split_heads(self.q_proj(x), batch_size, sequence_length)
        k = self._split_heads(self.k_proj(x), batch_size, sequence_length)
        v = self._split_heads(self.v_proj(x), batch_size, sequence_length)
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        key_mask = attention_mask[:, None, None, :]
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        return self.out_proj(context)

    def _split_heads(self, x: Tensor, batch_size: int, sequence_length: int) -> Tensor:
        return x.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: MahjongModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = FullSelfAttentionWithRoPE(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLUFeedForward(config)

    def forward(self, x: Tensor, attention_mask: Tensor, position_ids: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask, position_ids)
        return x + self.ffn(self.ffn_norm(x))


class KyokuEventEmbedding(nn.Module):
    """Embeds one V3 nine-field event token into a ``d_embed`` vector."""

    def __init__(self, config: MahjongModelConfig) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(48, config.d_embed, padding_idx=0)
        self.actor_embedding = nn.Embedding(5, config.d_embed)
        self.target_embedding = nn.Embedding(5, config.d_embed)
        self.tile_embedding = nn.Embedding(39, config.d_embed)
        self.tile2_embedding = nn.Embedding(39, config.d_embed)
        self.tile3_embedding = nn.Embedding(39, config.d_embed)
        self.value_embedding = nn.Embedding(19, config.d_embed)
        self.flag_embedding = nn.Embedding(32, config.d_embed)
        self.step_embedding = nn.Embedding(18, config.d_embed)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 3 or input_ids.shape[-1] != TOKEN_DIM:
            raise ValueError("input_ids must have shape [batch, sequence_length, 9]")
        token_ids = input_ids.long()
        return (
            self.type_embedding(token_ids[..., 0])
            + self.actor_embedding(token_ids[..., 1])
            + self.target_embedding(token_ids[..., 2])
            + self.tile_embedding(token_ids[..., 3])
            + self.tile2_embedding(token_ids[..., 4])
            + self.tile3_embedding(token_ids[..., 5])
            + self.value_embedding(token_ids[..., 6])
            + self.flag_embedding(token_ids[..., 7])
            + self.step_embedding(token_ids[..., 8])
        )


class KyokuTransformerActorCritic(nn.Module):
    """PPO actor-critic over append-only KyokuEventTuple V3 sequences."""

    def __init__(self, config: MahjongModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = KyokuEventEmbedding(config)
        self.input_proj: nn.Module
        if config.d_embed == config.d_model:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(config.d_embed, config.d_model)
        self.decision_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.policy_head = nn.Linear(config.d_model, NUM_ACTIONS)
        self.value_head = nn.Linear(config.d_model, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.decision_token, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        legal_mask: Tensor | None = None,
        *,
        attention_mask: Tensor | None = None,
        sequence_lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Returns PPO policy logits and values.

        The positional arguments match the existing PPO convention
        ``agent(observations, legal_mask)``. Future MJAI rollouts can additionally pass the
        state-machine ``attention_mask`` and ``sequence_lengths`` by keyword. If omitted,
        valid positions are inferred from ``TYPE != PAD``.
        """

        attention_mask, sequence_lengths = self._resolve_sequence_metadata(
            input_ids,
            attention_mask,
            sequence_lengths,
        )
        x = self.input_proj(self.embedding(input_ids))
        x, extended_mask, position_ids = self._append_decision_tokens(
            x,
            attention_mask,
            sequence_lengths,
        )
        for layer in self.layers:
            x = layer(x, extended_mask, position_ids)

        batch_indices = torch.arange(x.shape[0], device=x.device)
        h_decision = self.final_norm(x[batch_indices, sequence_lengths])
        raw_policy_logits = self.policy_head(h_decision)
        policy_logits = self._apply_legal_mask(raw_policy_logits, legal_mask)
        value = self.value_head(h_decision).squeeze(-1)
        return {
            "policy_logits": policy_logits,
            "raw_policy_logits": raw_policy_logits,
            "value": value,
        }

    @staticmethod
    def _resolve_sequence_metadata(
        input_ids: Tensor,
        attention_mask: Tensor | None,
        sequence_lengths: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 3 or input_ids.shape[-1] != TOKEN_DIM:
            raise ValueError("input_ids must have shape [batch, sequence_length, 9]")
        batch_size, sequence_length, _ = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids[..., 0].ne(0)
        else:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("attention_mask must have shape [batch, sequence_length]")
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

        inferred_lengths = attention_mask.long().sum(dim=1)
        if torch.any(inferred_lengths == 0):
            raise ValueError("every sequence must contain at least one non-padding token")
        expected_mask = (
            torch.arange(sequence_length, device=input_ids.device)[None, :]
            < inferred_lengths[:, None]
        )
        if not torch.equal(attention_mask, expected_mask):
            raise ValueError("attention_mask must use right-side padding")
        if sequence_lengths is None:
            sequence_lengths = inferred_lengths
        else:
            if sequence_lengths.shape != (batch_size,):
                raise ValueError("sequence_lengths must have shape [batch]")
            sequence_lengths = sequence_lengths.to(device=input_ids.device, dtype=torch.long)
            if not torch.equal(sequence_lengths, inferred_lengths):
                raise ValueError("sequence_lengths must equal the attention_mask token counts")
        return attention_mask, sequence_lengths

    def _append_decision_tokens(
        self,
        x: Tensor,
        attention_mask: Tensor,
        sequence_lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, sequence_length, hidden_dim = x.shape
        extended_length = sequence_length + 1
        extended = x.new_zeros((batch_size, extended_length, hidden_dim))
        extended[:, :sequence_length] = x
        extended_mask = torch.zeros(
            (batch_size, extended_length),
            dtype=torch.bool,
            device=x.device,
        )
        extended_mask[:, :sequence_length] = attention_mask
        batch_indices = torch.arange(batch_size, device=x.device)
        extended[batch_indices, sequence_lengths] = self.decision_token[0, 0]
        extended_mask[batch_indices, sequence_lengths] = True
        position_ids = torch.arange(extended_length, device=x.device).expand(batch_size, -1)
        return extended, extended_mask, position_ids

    @staticmethod
    def _apply_legal_mask(raw_logits: Tensor, legal_mask: Tensor | None) -> Tensor:
        if legal_mask is None:
            return raw_logits
        if legal_mask.shape != raw_logits.shape:
            raise ValueError(f"legal_mask must have shape {tuple(raw_logits.shape)}")
        legal_mask = legal_mask.to(device=raw_logits.device, dtype=torch.bool)
        return raw_logits.masked_fill(~legal_mask, torch.finfo(raw_logits.dtype).min)


def build_model(model_size: str) -> KyokuTransformerActorCritic:
    """Builds the single PPO model used by this project."""

    return KyokuTransformerActorCritic(make_mahjong_model_config(model_size))
