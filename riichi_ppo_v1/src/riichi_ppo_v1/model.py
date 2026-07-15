"""Causal Transformer actor-critic for V4 event blocks and board groups."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

NUM_ACTIONS = 241
BOARD_TOKENS = 12
BOARD_FIELDS = 160


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


class EventBlockEncoder(nn.Module):
    """Embed V4's pre-decoded compact event fields, never a 64-bit vocabulary."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.micro_kind = nn.Embedding(5, dim, padding_idx=0)
        self.micro_actor = nn.Embedding(5, dim, padding_idx=0)
        self.micro_tile = nn.Embedding(39, dim, padding_idx=0)
        self.micro_flag = nn.Embedding(8, dim, padding_idx=0)
        self.micro_position = nn.Embedding(4, dim)
        self.meld_value = nn.Embedding(40, dim, padding_idx=0)
        self.meld_position = nn.Embedding(8, dim)
        self.turn_norm = RMSNorm(dim, 1e-6)
        self.meld_norm = RMSNorm(dim, 1e-6)

    def forward(self, kinds: Tensor, turn: Tensor, meld: Tensor) -> Tensor:
        if kinds.ndim != 2 or turn.shape != (*kinds.shape, 4, 4) or meld.shape != (*kinds.shape, 8):
            raise ValueError("V4 event fields have incompatible shapes")
        turn = turn.long()
        micro = (
            self.micro_kind(turn[..., 0])
            + self.micro_actor(turn[..., 1])
            + self.micro_tile(turn[..., 2])
            + self.micro_flag(turn[..., 3])
            + self.micro_position(torch.arange(4, device=kinds.device)).view(1, 1, 4, -1)
        )
        active = turn[..., 0].ne(0).unsqueeze(-1)
        turn_embedding = self.turn_norm((micro * active).sum(-2) / active.sum(-2).clamp_min(1))
        meld = meld.long()
        meld_embedding = self.meld_value(meld) + self.meld_position(torch.arange(8, device=kinds.device)).view(1, 1, 8, -1)
        meld_embedding = self.meld_norm(meld_embedding.sum(-2))
        return torch.where(kinds.eq(1).unsqueeze(-1), turn_embedding, torch.where(kinds.eq(2).unsqueeze(-1), meld_embedding, torch.zeros_like(turn_embedding)))


class BoardStateEncoder(nn.Module):
    """Encode the twelve fixed V4 board groups from their raw u8 fields."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.value = nn.Embedding(64, dim, padding_idx=0)
        self.field = nn.Embedding(BOARD_FIELDS, dim)
        self.group = nn.Embedding(3, dim)
        self.norm = RMSNorm(dim, 1e-6)

    def forward(self, board: Tensor) -> Tensor:
        if board.ndim != 3 or board.shape[1:] != (BOARD_TOKENS, BOARD_FIELDS):
            raise ValueError(f"board_state must be [batch, {BOARD_TOKENS}, {BOARD_FIELDS}]")
        board = board.long()
        values = self.value(board)
        fields = self.field(torch.arange(BOARD_FIELDS, device=board.device)).view(1, 1, BOARD_FIELDS, -1)
        active = board.ne(0).unsqueeze(-1)
        pooled = (values + fields) * active
        pooled = pooled.sum(-2) / active.sum(-2).clamp_min(1)
        groups = self.group((torch.arange(BOARD_TOKENS, device=board.device) % 3)).view(1, BOARD_TOKENS, -1)
        return self.norm(pooled + groups)


class KyokuTransformerActorCritic(nn.Module):
    """Masked 241-action actor-critic over a single decision sequence."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig.preset("mid")
        self.event_embedding = EventBlockEncoder(self.config.d_embed)
        self.board_embedding = BoardStateEncoder(self.config.d_embed)
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
        block_kinds: Tensor,
        turn_fields: Tensor,
        meld_fields: Tensor,
        board_state: Tensor,
        legal_mask: Tensor | None = None,
        block_lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        lengths = self._metadata(block_kinds, turn_fields, meld_fields, board_state, block_lengths)
        events = self.input_proj(self.event_embedding(block_kinds, turn_fields, meld_fields))
        board = self.input_proj(self.board_embedding(board_state))
        batch, length, width = events.shape
        extended = events.new_zeros((batch, length + BOARD_TOKENS + 1, width))
        # ``events`` is right-padded already, so the full assignment is valid
        # for every row.  Keep the per-row board and decision positions on the
        # device: converting ``lengths[index]`` to Python would synchronize
        # once per decision in a rollout inference batch.
        extended[:, :length] = events
        row = torch.arange(batch, device=events.device)
        board_positions = lengths.unsqueeze(1) + torch.arange(BOARD_TOKENS, device=events.device)
        extended[row.unsqueeze(1), board_positions] = board
        decision_index = lengths + BOARD_TOKENS
        extended[row, decision_index] = self.decision_token
        positions = torch.arange(extended.shape[1], device=events.device).expand(batch, -1)
        for block in self.blocks:
            extended = block(extended, positions)
        hidden = self.norm(extended[row, decision_index])
        raw = self.policy_head(hidden)
        if legal_mask is None:
            logits = raw
        else:
            if legal_mask.shape != (batch, NUM_ACTIONS):
                raise ValueError("legal_mask must be [batch, 241]")
            legal_mask = legal_mask.to(device=raw.device, dtype=torch.bool)
            logits = raw.masked_fill(~legal_mask, float("-inf"))
        return {"raw_policy_logits": raw, "policy_logits": logits, "value": self.value_head(hidden).squeeze(-1)}

    @staticmethod
    def _metadata(block_kinds: Tensor, turn: Tensor, meld: Tensor, board: Tensor, lengths: Tensor | None) -> Tensor:
        if block_kinds.ndim != 2 or turn.shape != (*block_kinds.shape, 4, 4) or meld.shape != (*block_kinds.shape, 8):
            raise ValueError("V4 event fields have incompatible shapes")
        if board.shape != (block_kinds.shape[0], BOARD_TOKENS, BOARD_FIELDS):
            raise ValueError("V4 board state has incompatible batch shape")
        if lengths is None:
            lengths = block_kinds.ne(0).long().sum(-1)
        lengths = lengths.to(device=block_kinds.device, dtype=torch.long)
        if lengths.shape != (block_kinds.shape[0],):
            raise ValueError("block_lengths must have one entry per batch row")
        return lengths
