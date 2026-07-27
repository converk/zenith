import numpy as np
import torch


class ObservationEncoder:
    """Encodes observation into (74, tile_dim) spatial tensor using obs.encode().

    Args:
        tile_dim: Number of tile types (34 for 4P, 27 for 3P).
    """
    def __init__(self, tile_dim: int = 34):
        self.tile_dim = tile_dim

    def encode(self, obs) -> torch.Tensor:
        """Returns (74, tile_dim) float32 tensor from the Rust observation encoder."""
        feat_bytes = obs.encode()
        feat_numpy = np.frombuffer(feat_bytes, dtype=np.float32).reshape(74, self.tile_dim).copy()
        return torch.from_numpy(feat_numpy)
