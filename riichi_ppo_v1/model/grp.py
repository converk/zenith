"""GRP(全局排名预测)模型与输入契约常量(Mortal 方案,V18 扩展)。

输入为每个 StartKyoku 的 21 维全局状态(V18 契约,字段顺序单一来源):

``[0]  grand_kyoku     0..7(E1..E4=0..3、S/W1..4=4..7)
 [1]  honba
 [2]  kyotaku
 [3:7]  s0..s3 / 1e4
 [7]  game_type        0=东风、1=半庄、2=西风(整局经历风数 - 1)
 [8]  prev_result_type 0=首局、1=荣和、2=自摸、3=流局、4=中止
 [9:13]  wins0..3      各玩家截至本小局开始的累计和了次数
 [13:17] dealins0..3   各玩家累计放铳次数
 [17:21] tenpai0..3    各玩家累计听牌流局次数]``

全部新增字段只来自公开小局结果与局风(边界状态),不包含手牌、牌河、未来牌等
局内发展信息。每个半庄的完整 prefix 序列监督最终四人排名的 4! = 24 类全排列标签。
模型结构:21 → 2 层 GRU(hidden=96) → concat hidden(192) → Linear(192,192) → ReLU →
Linear(192,24)。训练完成后完全冻结,PPO 阶段只读。
"""

from __future__ import annotations

from itertools import permutations

import torch
from torch import Tensor, nn

# 输入契约与结构超参(单一来源;构造参数默认值取此处常量)。
GRP_INPUT_SIZE = 21
GRP_INPUT_LAYOUT = (
    "grand_kyoku", "honba", "kyotaku",
    "s0", "s1", "s2", "s3",
    "game_type", "prev_result_type",
    "wins0", "wins1", "wins2", "wins3",
    "dealins0", "dealins1", "dealins2", "dealins3",
    "tenpai0", "tenpai1", "tenpai2", "tenpai3",
)
GRP_HIDDEN = 96
GRP_LAYERS = 2
GRP_NUM_CLASSES = 24  # 4! 四人最终排名全排列

# 局风类型(整局经历的风数 - 1;与离线 bakaze 集合、在线 game_mode 映射一致)。
GAME_TYPE_EAST = 0
GAME_TYPE_HALF = 1
GAME_TYPE_WEST = 2

# 上一小局结果类型(0=首局/无结果,与 KyokuResult.result_type 对齐)。
PREV_RESULT_NONE = 0  # 首局
PREV_RESULT_RON = 1
PREV_RESULT_TSUMO = 2
PREV_RESULT_RYUKYOKU = 3
PREV_RESULT_ABORT = 4

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
    """Mortal 式 GRP:GRU(21→96, 2 层) + fc(192→192→24)。

    构造参数带默认常量,PPO 加载时按 checkpoint 的 ``model_config`` 传入,
    避免与离线训练形状硬编码耦合。
    """

    def __init__(
        self,
        input_size: int = GRP_INPUT_SIZE,
        hidden_size: int = GRP_HIDDEN,
        num_layers: int = GRP_LAYERS,
        num_classes: int = GRP_NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_classes = int(num_classes)
        self.rnn = nn.GRU(
            self.input_size, self.hidden_size, num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_layers, self.hidden_size * self.num_layers),
            nn.ReLU(),
            nn.Linear(self.hidden_size * self.num_layers, self.num_classes),
        )
        perms, perms_t = _make_permutations()
        self.register_buffer("perms", perms)
        self.register_buffer("perms_t", perms_t)

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        """``features`` [B,T,input_size] float32、``lengths`` [B] long → logits [B,24]。

        只使用 GRU 末层 hidden 拼接(与 Mortal ``forward_packed`` 一致)。
        """
        if features.ndim != 3 or features.shape[-1] != self.input_size:
            raise ValueError(
                f"GRP features must be [batch, steps, {self.input_size}]"
            )
        packed = nn.utils.rnn.pack_padded_sequence(
            features.float(), lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _output, state = self.rnn(packed)
        state = state.transpose(0, 1).flatten(1)  # [B, hidden*layers]=[B,192]
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
