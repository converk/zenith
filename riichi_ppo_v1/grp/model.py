"""Small GRP MLP following smly/RiichiEnv riichienv-ml (Suphx-style reward)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


FEATURE_DIM = 20
NUM_PLAYERS = 4


def grp_features_from_scores(
    initial_scores: Sequence[float],
    end_scores: Sequence[float],
    *,
    chang: int = 0,
    ju: int = 0,
    ben: int = 0,
    liqibang: int = 0,
    player: int = 0,
) -> np.ndarray:
    """Build the 20-dim GRP feature row for one player.

    Encoding matches riichienv-ml: four initial scores /25000, four end
    scores /25000, four kyoku deltas /12000, field wind /3, dealer kyoku /3,
    honba /4, riichi sticks /4, then a player one-hot.
    """
    if player < 0 or player >= NUM_PLAYERS:
        raise ValueError(f"player must be in [0, {NUM_PLAYERS})")
    initial = np.asarray([float(x) for x in initial_scores], dtype=np.float32)
    end = np.asarray([float(x) for x in end_scores], dtype=np.float32)
    if initial.shape != (NUM_PLAYERS,) or end.shape != (NUM_PLAYERS,):
        raise ValueError("GRP features require exactly four scores per side")
    delta = end - initial
    row = np.zeros(FEATURE_DIM, dtype=np.float32)
    row[0:4] = initial / 25000.0
    row[4:8] = end / 25000.0
    row[8:12] = delta / 12000.0
    row[12] = float(chang) / 3.0
    row[13] = float(ju) / 3.0
    row[14] = float(ben) / 4.0
    row[15] = float(liqibang) / 4.0
    row[16 + player] = 1.0
    return row


def reward_from_rank_probs(
    rank_probs: np.ndarray,
    pts_weight: Sequence[float] = (10, 4, -4, -10),
) -> float:
    """Convert a 4-class final-rank probability vector to a GRP reward."""
    probs = np.asarray(rank_probs, dtype=np.float64)
    if probs.shape != (NUM_PLAYERS,):
        raise ValueError("rank probabilities must have shape (4,)")
    weights = np.asarray(pts_weight, dtype=np.float64)
    if weights.shape != (NUM_PLAYERS,):
        raise ValueError("pts_weight must have four entries")
    return float(np.dot(probs, weights) - float(np.mean(weights)))


class RankPredictor(nn.Module):
    """MLP 20 -> 128 -> 64 -> 4 with dropout 0.1 (riichienv-ml default)."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = int(input_dim)
        for width in hidden:
            layers.extend([
                nn.Linear(current, int(width)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
            ])
            current = int(width)
        layers.append(nn.Linear(current, NUM_PLAYERS))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    @torch.inference_mode()
    def predict_rank_probs(
        self,
        features: np.ndarray | torch.Tensor,
        device: torch.device | str = "cpu",
    ) -> np.ndarray:
        self.eval()
        tensor = (
            torch.as_tensor(features, dtype=torch.float32, device=device)
            if not isinstance(features, torch.Tensor)
            else features.to(device)
        )
        logits = self.forward(tensor)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return probs

    @torch.inference_mode()
    def predict_rank(self, features: np.ndarray) -> np.ndarray:
        probs = self.predict_rank_probs(features)
        return np.argmax(probs, axis=-1)

    @classmethod
    def from_checkpoint(cls, path: str) -> "RankPredictor":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**payload.get("model_kwargs", {}))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def save_grp_checkpoint(model: RankPredictor, path: str, **extra: Any) -> None:
    torch.save(
        {
            "model_kwargs": {
                "input_dim": FEATURE_DIM,
                "hidden": (128, 64),
                "dropout": 0.1,
            },
            "state_dict": model.state_dict(),
            "feature_dim": FEATURE_DIM,
            "num_players": NUM_PLAYERS,
            **extra,
        },
        path,
    )
