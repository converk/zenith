import torch
import torch.nn as nn

from riichienv_ml.models.backbone import ResNetBackbone


class QNetwork(nn.Module):
    """Dueling Q-Network for both offline CQL and online DQN training.

    Uses separate value and advantage heads: q = v + (a - a.mean()).
    Optionally includes an auxiliary head for rank prediction.

    Input:  (B, in_channels, tile_dim)
    Output: (B, num_actions) Q-values
    """
    def __init__(self, in_channels: int = 74, num_actions: int = 82,
                 conv_channels: int = 64, num_blocks: int = 3, fc_dim: int = 256,
                 tile_dim: int = 34, aux_dims: int | None = None, **kwargs):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels, conv_channels, num_blocks, fc_dim, tile_dim)
        # Dueling DQN: separate value and advantage streams
        self.v_head = nn.Linear(fc_dim, 1)
        self.a_head = nn.Linear(fc_dim, num_actions)
        # Auxiliary head (rank prediction)
        self.aux_head = nn.Linear(fc_dim, aux_dims) if aux_dims else None

    def _compute_q(self, v: torch.Tensor, a: torch.Tensor,
                   mask: torch.Tensor | None = None) -> torch.Tensor:
        """Dueling DQN: q = v + a - mean(a).

        When mask is provided (online DQN), mean is computed only over legal
        actions (Mortal-style). Without mask (offline CQL), uses the full mean
        for backward compatibility.
        """
        if mask is not None:
            mask_bool = mask.bool() if mask.dtype != torch.bool else mask
            a_masked = a.masked_fill(~mask_bool, 0.0)
            a_mean = a_masked.sum(dim=-1, keepdim=True) / mask_bool.sum(dim=-1, keepdim=True).clamp(min=1)
            q = v + a - a_mean
            q = q.masked_fill(~mask_bool, -torch.inf)
        else:
            q = v + a - a.mean(dim=-1, keepdim=True)
        return q

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.backbone(x)
        v = self.v_head(features)       # (B, 1)
        a = self.a_head(features)       # (B, num_actions)
        return self._compute_q(v, a, mask)

    def forward_with_aux(self, x: torch.Tensor,
                         mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (q_values, aux_logits). aux_logits is None if no aux_head."""
        features = self.backbone(x)
        v = self.v_head(features)
        a = self.a_head(features)
        q = self._compute_q(v, a, mask)
        aux = self.aux_head(features) if self.aux_head is not None else None
        return q, aux
