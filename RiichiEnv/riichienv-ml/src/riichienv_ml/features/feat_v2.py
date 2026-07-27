import numpy as np
import torch


class DiscardHistoryEncoder:
    """Encodes observation into (78, tile_dim) tensor: base 74ch + 4ch discard decay.

    Args:
        tile_dim: Number of tile types (34 for 4P, 27 for 3P).
    """
    def __init__(self, tile_dim: int = 34):
        self.tile_dim = tile_dim

    def encode(self, obs) -> torch.Tensor:
        base = np.frombuffer(obs.encode(), dtype=np.float32).reshape(74, self.tile_dim).copy()
        decay = np.frombuffer(obs.encode_discard_history_decay(), dtype=np.float32).reshape(4, self.tile_dim).copy()
        combined = np.concatenate([base, decay], axis=0)
        return torch.from_numpy(combined)


class DiscardHistoryShantenEncoder:
    """Encodes observation into (94, tile_dim) tensor: base 74ch + 4ch discard decay + 16ch shanten.

    Args:
        tile_dim: Number of tile types (34 for 4P, 27 for 3P).
    """
    def __init__(self, tile_dim: int = 34):
        self.tile_dim = tile_dim

    def encode(self, obs) -> torch.Tensor:
        base = np.frombuffer(obs.encode(), dtype=np.float32).reshape(74, self.tile_dim).copy()
        decay = np.frombuffer(obs.encode_discard_history_decay(), dtype=np.float32).reshape(4, self.tile_dim).copy()
        shanten = np.frombuffer(obs.encode_shanten_efficiency(), dtype=np.float32).reshape(4, 4).copy()
        shanten_broadcast = np.repeat(shanten.reshape(16, 1), self.tile_dim, axis=1)
        combined = np.concatenate([base, decay, shanten_broadcast], axis=0)
        return torch.from_numpy(combined)
