import numpy as np
import torch


class ExtendedEncoder:
    """Encodes observation into (215, tile_dim) tensor using a single consolidated Rust call.

    Args:
        tile_dim: Number of tile types (34 for 4P, 27 for 3P).
    """
    def __init__(self, tile_dim: int = 34):
        self.tile_dim = tile_dim

    def encode(self, obs) -> torch.Tensor:
        raw = obs.encode_extended()
        return torch.from_numpy(
            np.frombuffer(raw, dtype=np.float32).reshape(215, self.tile_dim).copy()
        )
