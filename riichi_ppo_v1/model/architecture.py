"""Semantic-token decoder-only GQA actor-critic.

v13 adds isolated two-token action queries while retaining the legacy fixed
241-way head so old checkpoints remain loadable for ablations and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .schema import ACTION_QUERY_DEFENSE, ACTION_QUERY_OFFENSE, ACTION_QUERY_SEGMENT

NUM_ACTIONS = 241
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
    policy_head_type: str = "legacy_fixed"

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
        if self.policy_head_type not in {"legacy_fixed", "isolated_action_query"}:
            raise ValueError("policy_head_type must be legacy_fixed or isolated_action_query")

    @classmethod
    def preset(cls, size: str) -> "ModelConfig":
        configs = {
            "mid": cls(),
            "large": cls(d_model=384, query_heads=12, kv_heads=3, head_dim=32, ffn_dim=1152),
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
            raise ValueError("legal_mask must be [batch, 241]")
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
        if self.config.policy_head_type == "legacy_fixed":
            self.query = nn.Parameter(torch.empty(self.config.d_model))
            self.policy_head: nn.Module = nn.Linear(self.config.d_model, NUM_ACTIONS)
            nn.init.normal_(self.query, std=self.config.d_model ** -0.5)
        else:
            self.register_parameter("query", None)
            self.policy_head = nn.Sequential(
                nn.RMSNorm(self.config.d_model, eps=self.config.eps),
                nn.Linear(self.config.d_model, self.config.d_model),
                nn.SiLU(),
                nn.Linear(self.config.d_model, 1),
            )
        self.value_head = nn.Linear(self.config.d_model, 1)
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
        scores = self.policy_head(actor[rows, positions]).squeeze(-1).float()
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
        extra = int(self.config.policy_head_type == "legacy_fixed")
        if token_factors.shape[1] + extra > self.config.context_tokens:
            raise ValueError(f"context overflow: {token_factors.shape[1] + 1} > {self.config.context_tokens}")
        if token_lengths is None:
            token_lengths = token_factors.ne(0).any(-1).long().sum(-1)
        token_lengths = token_lengths.to(device=token_factors.device, dtype=torch.long)
        if token_lengths.shape != (token_factors.shape[0],):
            raise ValueError("token_lengths must have one entry per batch row")
        if torch.any(token_lengths < 0) or torch.any(token_lengths > token_factors.shape[1]):
            raise ValueError("token_lengths exceed supplied token rows")
        if self.config.policy_head_type == "isolated_action_query":
            _shared, actor, _state, defense_ids, _valid = self._isolated_public(
                token_factors, token_numeric, token_lengths, legal_mask
            )
            raw, logits = self._isolated_logits(actor, defense_ids, legal_mask)
            return {"raw_policy_logits": raw, "policy_logits": logits}
        tokens = self.token_embedding(token_factors, token_numeric)
        batch, padded, width = tokens.shape
        sequence = tokens.new_zeros((batch, padded + 1, width))
        sequence[:, :padded] = tokens
        rows = torch.arange(batch, device=tokens.device)
        sequence[rows, token_lengths] = self.query
        public_sequence = self.public_backbone(sequence, token_lengths + 1)
        hidden = self.actor_backbone(public_sequence, token_lengths + 1)[rows, token_lengths]
        raw = self.policy_head(hidden).float()
        if legal_mask is None:
            logits = raw
        else:
            if legal_mask.shape != (batch, NUM_ACTIONS):
                raise ValueError("legal_mask must be [batch, 241]")
            logits = raw.masked_fill(~legal_mask.to(device=raw.device, dtype=torch.bool), float("-inf"))
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
        extra = int(self.config.policy_head_type == "legacy_fixed")
        if token_factors.shape[1] + extra > self.config.context_tokens:
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
        rows = torch.arange(batch, device=tokens.device)
        if self.config.policy_head_type == "isolated_action_query":
            public_sequence, actor_sequence, state_mask, defense_ids, _valid = self._isolated_public(
                token_factors, token_numeric, token_lengths, legal_mask
            )
            raw, logits = self._isolated_logits(actor_sequence, defense_ids, legal_mask)
            public_lengths = state_mask.long().sum(-1)
            public_capacity = int(public_lengths.max().item())
            packed_public = public_sequence.new_zeros((batch, public_capacity, width))
            for row in range(batch):
                state_length = int(public_lengths[row])
                packed_public[row, :state_length] = public_sequence[row, state_mask[row]]
            public_sequence = packed_public
        else:
            sequence = tokens.new_zeros((batch, padded + 1, width))
            sequence[:, :padded] = tokens
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
        # Autocast makes the preceding matrix multiplies BF16, but policy
        # probabilities and value estimates feed PPO's numerically sensitive
        # ratio/loss path.  Promote them before leaving the model, as in
        # exp/training's ActorCritic.
        if self.config.policy_head_type == "legacy_fixed":
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
