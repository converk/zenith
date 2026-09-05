"""V19 信念网络：对三家隐藏手牌/向听/听牌/危险度/打点的不完美信息推断。

信念网络接在共享 public_backbone 之后：
- 输入 ``z_pool = mean-pool(shared_hidden)``（全局池化，三家共用同一表示）；
- 一个隐藏层（256→512 + SiLU）；
- 五个监督头：Hand（逐格计数 softmax）、Shanten（逐家 softmax）、Wait（逐格
  sigmoid BCE）、Danger（逐格 sigmoid BCE）、Loss（sigmoid 归一化回归）；
- 三家共享同一个线性转换矩阵（282 → 10×d_model），把每家的信念摘要压成
  10 个 256 维 token，作为模型内部产物注入 Actor 尾段（不进 Rust 编码器）。

信念 token 是策略的一部分：训练/推理同一条前向路径，不依赖外部标签。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# 三家对手（相对观察者）。
BELIEF_PLAYERS = 3
# 牌种数（34）与逐牌种计数上限（0..4）。
HAND_TILE_KINDS = 34
HAND_COUNT_CLASSES = 5
# 向听类别：0..8（0 = 听牌）。
SHANTEN_CLASSES = 9
# 听牌牌种：34 种 + 第 35 位 N/A（非听牌）。
WAIT_CLASSES = 35
# 危险度/打点均为 34 种牌。
DANGER_CLASSES = 34
LOSS_CLASSES = 34
# 信念摘要维度 = 170（手牌软概率分布）+ 9（向听）+ 35（听牌）+ 34（危险度）
# + 34（归一化打点）= 282。
SUMMARY_DIM = HAND_TILE_KINDS * HAND_COUNT_CLASSES + SHANTEN_CLASSES + WAIT_CLASSES + DANGER_CLASSES + LOSS_CLASSES
# 每玩家信念 token 数（定版 10/家，不做消融）。
DEFAULT_TOKEN_COUNT = 10


class BeliefNetwork(nn.Module):
    """V19 信念网络：共享表示 → 五头预测 → 三家各 10 token。

    ``forward`` 返回：
    - ``belief_hand_logits``    [B,3,34,5]：逐格计数 softmax 输入的 logits；
    - ``belief_shanten_logits`` [B,3,9]：逐家向听 softmax 输入的 logits；
    - ``belief_wait_logits``    [B,3,35]：逐格听牌 sigmoid BCE 的 logits；
    - ``belief_danger_logits``  [B,3,34]：逐格危险度 sigmoid BCE 的 logits；
    - ``belief_loss_pred``      [B,3,34]：sigmoid 归一化回归预测（点数/24000
      clip 到 [0,1] 的目标）；
    - ``belief_summary``        [B,3,282]：三家共享同一拼接顺序的摘要
      （softmax(hand) + softmax(shanten) + sigmoid(wait) + sigmoid(danger)
      + loss_pred）；
    - ``belief_tokens``         [B,30,d]：三家 ×10 的注入 token，玩家主序
      [rel0 的 10 token, rel1 的 10, rel2 的 10]；由共享转换矩阵生成。
    """

    def __init__(self, d_model: int = 256, hidden: int = 512, token_count: int = DEFAULT_TOKEN_COUNT) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.hidden = int(hidden)
        self.token_count = int(token_count)
        if self.token_count < 1:
            raise ValueError("token_count must be positive")
        # 公共隐藏层：256 → 512 + SiLU（设计文档 §2.3，用户已定单层）。
        self.hidden_layer = nn.Sequential(
            nn.Linear(self.d_model, self.hidden),
            nn.SiLU(),
        )
        # 五个输出头直接展平三家维度；三家共享同一头，但保留各家独立输出切片。
        self.hand_head = nn.Linear(
            self.hidden, BELIEF_PLAYERS * HAND_TILE_KINDS * HAND_COUNT_CLASSES,
        )
        self.shanten_head = nn.Linear(
            self.hidden, BELIEF_PLAYERS * SHANTEN_CLASSES,
        )
        self.wait_head = nn.Linear(
            self.hidden, BELIEF_PLAYERS * WAIT_CLASSES,
        )
        self.danger_head = nn.Linear(
            self.hidden, BELIEF_PLAYERS * DANGER_CLASSES,
        )
        self.loss_head = nn.Linear(
            self.hidden, BELIEF_PLAYERS * LOSS_CLASSES,
        )
        # 共享转换矩阵：每家 282 维摘要 → 10×d_model token（三家共用同一矩阵）。
        self.token_matrix = nn.Linear(SUMMARY_DIM, self.token_count * self.d_model)

    def forward(self, shared_hidden: Tensor) -> dict[str, Tensor]:
        """从共享公共表示生成五个信念头与注入 token。"""
        # 全局池化：对序列维度取均值，得到 [B, d]。
        z_pool = shared_hidden.mean(dim=1)
        hidden = self.hidden_layer(z_pool)
        batch = hidden.shape[0]

        # 五个头的原始输出（logits），展平 [B, 3×...] 再 reshape 为 [B,3,...]。
        hand_logits = self.hand_head(hidden).view(
            batch, BELIEF_PLAYERS, HAND_TILE_KINDS, HAND_COUNT_CLASSES,
        )
        shanten_logits = self.shanten_head(hidden).view(
            batch, BELIEF_PLAYERS, SHANTEN_CLASSES,
        )
        wait_logits = self.wait_head(hidden).view(
            batch, BELIEF_PLAYERS, WAIT_CLASSES,
        )
        danger_logits = self.danger_head(hidden).view(
            batch, BELIEF_PLAYERS, DANGER_CLASSES,
        )
        # 回归头经 sigmoid 归一化到 [0,1]（目标 = 点数/24000 clip）。
        loss_pred = torch.sigmoid(self.loss_head(hidden)).view(
            batch, BELIEF_PLAYERS, LOSS_CLASSES,
        )

        # 三家共享同一摘要拼接顺序：
        # softmax(hand) 展平 170 维 + softmax(shanten) 9 维 + sigmoid(wait)
        # 35 维 + sigmoid(danger) 34 维 + loss_pred 34 维 = 282 维。
        hand_feature = torch.softmax(hand_logits, dim=-1).reshape(batch, BELIEF_PLAYERS, -1)
        shanten_feature = torch.softmax(shanten_logits, dim=-1)
        wait_feature = torch.sigmoid(wait_logits)
        danger_feature = torch.sigmoid(danger_logits)
        summary = torch.cat(
            [hand_feature, shanten_feature, wait_feature, danger_feature, loss_pred],
            dim=-1,
        )  # [B,3,282]

        # 共享转换矩阵按家应用：每家 282 维 → token_count×d，再 reshape 为
        # [B, 3×token_count, d]（玩家主序：rel0 的 10 token、rel1、rel2）。
        summary_flat = summary.reshape(-1, SUMMARY_DIM)
        belief_tokens = self.token_matrix(summary_flat).view(
            batch, BELIEF_PLAYERS * self.token_count, self.d_model,
        )
        return {
            "belief_hand_logits": hand_logits,
            "belief_shanten_logits": shanten_logits,
            "belief_wait_logits": wait_logits,
            "belief_danger_logits": danger_logits,
            "belief_loss_pred": loss_pred,
            "belief_summary": summary,
            "belief_tokens": belief_tokens,
        }
