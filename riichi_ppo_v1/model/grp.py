"""GRP(全局排名预测)模型与输入契约常量(Mortal 方案)。

输入为每个 StartKyoku 的 7 维全局状态
``[grand_kyoku, honba, kyotaku, s0/1e4, s1/1e4, s2/1e4, s3/1e4]``;
每个半庄的完整 prefix 序列监督最终四人排名的 4! = 24 类全排列标签。
模型结构:7 → 2 层 GRU(hidden=64) → concat hidden → Linear(128,128) → ReLU →
Linear(128,24)。训练完成后完全冻结,PPO 阶段只读。
"""

from __future__ import annotations

from itertools import permutations

import torch
from torch import Tensor, nn

# 输入契约与结构超参(单一来源)。
GRP_INPUT_SIZE = 7  # [grand_kyoku, honba, kyotaku, s0/1e4, s1/1e4, s2/1e4, s3/1e4]
GRP_HIDDEN = 64
GRP_LAYERS = 2
GRP_NUM_CLASSES = 24  # 4! 四人最终排名全排列

# 排名 utility(Mortal pts [3,1,-1,-3] 按 1/3 归一化)。
GRP_UTILITY: tuple[float, float, float, float] = (
    1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0,
)


def _make_permutations() -> tuple[Tensor, Tensor]:
    """固化 4! 全排列 ``perms (24,4)`` 与转置 ``perms_t (4,24)``。"""
    perms = torch.tensor(list(permutations(range(4))), dtype=torch.long)
    return perms, perms.transpose(0, 1)


def expected_rank_utility(matrix: Tensor) -> Tensor:
    """把 ``calc_matrix`` 输出的 [..., 4, 4] 玩家排名概率转为期望 utility。

    ``matrix[..., player, rank]`` = P(该玩家最终排名为 rank);
    返回 [... , 4],即每个玩家的 ``Σ_rank P(rank)·U(rank)``。
    """
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"GRP matrix must end with [4, 4], got {tuple(matrix.shape)}")
    utility = matrix.new_tensor(GRP_UTILITY)
    return matrix @ utility


class GRPModel(nn.Module):
    """Mortal 式 GRP:GRU(7→64, 2 层) + fc(128→128→24)。"""

    def __init__(self, hidden_size: int = GRP_HIDDEN, num_layers: int = GRP_LAYERS) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            GRP_INPUT_SIZE, hidden_size, num_layers=num_layers, batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * num_layers, hidden_size * num_layers),
            nn.ReLU(),
            nn.Linear(hidden_size * num_layers, GRP_NUM_CLASSES),
        )
        perms, perms_t = _make_permutations()
        self.register_buffer("perms", perms)
        self.register_buffer("perms_t", perms_t)

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        """``features`` [B,T,7] float32、``lengths`` [B] long → logits [B,24]。

        只使用 GRU 末层 hidden 拼接(与 Mortal ``forward_packed`` 一致)。
        """
        if features.ndim != 3 or features.shape[-1] != GRP_INPUT_SIZE:
            raise ValueError(f"GRP features must be [batch, steps, {GRP_INPUT_SIZE}]")
        packed = nn.utils.rnn.pack_padded_sequence(
            features.float(), lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _output, state = self.rnn(packed)
        state = state.transpose(0, 1).flatten(1)  # [B, hidden*layers]=[B,128]
        return self.fc(state)

    @torch.inference_mode()
    def calc_matrix(self, logits: Tensor) -> Tensor:
        """24 类 logits [N,24] → [N,4,4] 玩家排名概率矩阵。

        ``matrix[player][rank]`` = ``Σ_{perm: perm[player]==rank}`` softmax 概率,
        每个玩家的行和为 1。
        """
        probs = torch.softmax(logits.float(), dim=-1)
        matrix = torch.zeros(
            (*logits.shape[:-1], 4, 4), dtype=probs.dtype, device=probs.device,
        )
        for player in range(4):
            for rank in range(4):
                cond = self.perms_t[player] == rank  # 该玩家在该类排列中排名=rank
                matrix[..., player, rank] = probs[..., cond].sum(-1)
        return matrix

    def get_label(self, rank_by_player: Tensor) -> Tensor:
        """[N,4] ``rank_by_player``(player → 最终顺位 0..3)→ [N] 24 类索引。

        纯整数索引运算,不含梯度;训练期与推理期均可调用。
        """
        batch_size = int(rank_by_player.shape[0])
        perms = self.perms.expand(batch_size, -1, -1).transpose(0, 1)  # (24, N, 4)
        mappings = (perms == rank_by_player).all(-1).nonzero()
        labels = torch.zeros(batch_size, dtype=torch.int64, device=mappings.device)
        labels[mappings[:, 1]] = mappings[:, 0]
        return labels

    def freeze(self) -> None:
        """离线训练完成后完全冻结;PPO 不更新 GRP 参数。"""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
