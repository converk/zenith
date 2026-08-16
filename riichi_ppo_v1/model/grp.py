"""V16 轻量 GRP(全局排名预测)模型与输入契约常量。

输入为小局边界的比赛状态 + 上一小局结果(首局 START),每个 prefix 监督该视角
玩家的最终排名;模型输出 P(rank=1..4),总参数 50–70K,PPO 阶段完全冻结。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# 输入契约:9 个类别字段与 13 个连续字段(单一来源)。
# rank(4)/场风(2)/局数(8)/庄家相对(4)/结果类型(5:START,ron,tsumo,ryukyoku,
# abort)/胜者相对(5,含 N/A)/放铳相对(5,含 N/A)/流局听牌掩码(16)/连庄(2)。
GRP_CATEGORIES: tuple[int, ...] = (4, 2, 8, 4, 5, 5, 5, 16, 2)
GRP_NUMERIC_FEATURES = 13  # 4 家点数 + 3 相对分差 + honba + 立直棒 + 4 家分差
GRP_HIDDEN = 64
GRP_LAYERS = 2
GRP_EMBED_DIM = 4
GRP_HEAD_HIDDEN = 32

# 排名 utility(契约 reward-v16.md §1;V16 奖励范围放大后取 ×2,末位 -24)。
GRP_UTILITY: tuple[float, float, float, float] = (24.0, 8.0, -12.0, -24.0)


def expected_value(rank_logits: Tensor) -> Tensor:
    """把 rank logits 转为期望 utility:Σ P(rank)·U(rank)。"""
    probabilities = torch.softmax(rank_logits, dim=-1)
    utility = rank_logits.new_tensor(GRP_UTILITY)
    return probabilities @ utility


class GRPModel(nn.Module):
    """Linear 64 → 2 层 GRU(64)→ Linear 64→32 → SiLU → Linear 32→4。"""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, GRP_EMBED_DIM) for cardinality in GRP_CATEGORIES
        )
        input_dim = GRP_EMBED_DIM * len(GRP_CATEGORIES) + GRP_NUMERIC_FEATURES
        self.projection = nn.Linear(input_dim, GRP_HIDDEN)
        self.gru = nn.GRU(
            GRP_HIDDEN, GRP_HIDDEN, num_layers=GRP_LAYERS, batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(GRP_HIDDEN, GRP_HEAD_HIDDEN),
            nn.SiLU(),
            nn.Linear(GRP_HEAD_HIDDEN, 4),
        )

    def forward(
        self,
        categorical: Tensor,
        numeric: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """``categorical`` [B,T,9] long、``numeric`` [B,T,13]、``lengths`` [B]。"""
        if categorical.ndim != 3 or categorical.shape[-1] != len(GRP_CATEGORIES):
            raise ValueError(f"GRP categorical must be [batch, steps, {len(GRP_CATEGORIES)}]")
        if numeric.ndim != 3 or numeric.shape[-1] != GRP_NUMERIC_FEATURES:
            raise ValueError(f"GRP numeric must be [batch, steps, {GRP_NUMERIC_FEATURES}]")
        parts = [
            embedding(categorical[..., index].long())
            for index, embedding in enumerate(self.embeddings)
        ]
        projected = self.projection(torch.cat((*parts, numeric.float()), dim=-1))
        packed = nn.utils.rnn.pack_padded_sequence(
            projected, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        output, _hidden = self.gru(packed)
        output, _lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return self.head(output)

    def freeze(self) -> None:
        """离线训练完成后完全冻结;PPO 不更新 GRP 参数。"""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
