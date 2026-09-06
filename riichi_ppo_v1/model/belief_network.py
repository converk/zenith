"""V19 信念网络：由 player query hidden 生成五头预测、摘要与注入 token。

信念 backbone（1 层、与 Critic 同构）在 ``architecture.py`` 中消费完整
``shared_hidden`` 序列 + 每玩家 3 个查询 token（共 9 个），本模块只负责
「头 + 摘要 + token」：
- 输入 ``player_query_hidden [B,3,3,256]``（玩家 × 3 查询 × d_model）；
- 五个共享逐家小头（256 → 各头输出维度）对**每个查询分别应用**，再按
  查询维取平均 logits，输出形状与定稿协议完全一致：
  ``belief_hand_logits [B,3,34,5]``、``belief_shanten_logits [B,3,9]``、
  ``belief_wait_logits [B,3,35]``、``belief_danger_logits [B,3,34]``、
  ``belief_loss_pred [B,3,34]``；
- 三家共享同一个线性转换矩阵（282 → 10×d_model），把每家的信念摘要压成
  10 个 256 维 token，作为模型内部产物注入 Actor 尾段（不进 Rust 编码器）。

信念 token 是策略的一部分：训练/推理同一条前向路径，不依赖外部标签。
``token_matrix`` 结构与训练方式均不变——**只**由 actor/policy 梯度更新，
监督损失不经过 token 路径；且 token 路径的输入为 ``detach(summary)``，
策略/BC 损失止于转换矩阵，不进入五头/backbone/belief_query（梯度隔离，
见 `audit/reports/v19/design/V19_信念网络策略梯度隔离_实施方案.md`）。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# 三家对手（相对观察者：0=下家、1=对面、2=上家，与 belief_labels 顺序一致）。
BELIEF_PLAYERS = 3
# 每玩家查询 token 数（v19 60% 定版：3 个查询）。
BELIEF_QUERIES_PER_PLAYER = 3
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
    """V19 信念网络：player query 特征 → 五头预测 → 三家各 10 token。

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

    三个查询先在 logits 空间取平均，再交给归一化/摘要，因此输出形状与
    v19 输入协议中的三家摘要/30 token 完全一致。所有头都是三家共享的小头
    （256 → 各头输出），参数规模比旧「512 隐藏层 + 展平 3 家」方案更小。
    """

    def __init__(self, d_model: int = 256, token_count: int = DEFAULT_TOKEN_COUNT) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.token_count = int(token_count)
        if self.token_count < 1:
            raise ValueError("token_count must be positive")
        # 五个共享逐家小头：每个头输出单家单查询的 logits 宽度。
        self.hand_head = nn.Linear(self.d_model, HAND_TILE_KINDS * HAND_COUNT_CLASSES)
        self.shanten_head = nn.Linear(self.d_model, SHANTEN_CLASSES)
        self.wait_head = nn.Linear(self.d_model, WAIT_CLASSES)
        self.danger_head = nn.Linear(self.d_model, DANGER_CLASSES)
        self.loss_head = nn.Linear(self.d_model, LOSS_CLASSES)
        # 共享转换矩阵：每家 282 维摘要 → 10×d_model token（三家共用同一矩阵）。
        self.token_matrix = nn.Linear(SUMMARY_DIM, self.token_count * self.d_model)

    def forward(self, player_query_hidden: Tensor) -> dict[str, Tensor]:
        """从 player query hidden 生成五个信念头与注入 token。"""
        if player_query_hidden.ndim != 4 or player_query_hidden.shape[1:] != (
            BELIEF_PLAYERS,
            BELIEF_QUERIES_PER_PLAYER,
            self.d_model,
        ):
            raise ValueError(
                "player_query_hidden must be [B,3,3,d_model], got "
                f"{tuple(player_query_hidden.shape)}"
            )
        batch = player_query_hidden.shape[0]
        # 展平成 [B*3*3, d_model]，每家每查询独立过共享小头。
        flat = player_query_hidden.reshape(-1, self.d_model)

        # 各头输出 [B*9, out] → [B,3,3,out...]，再按查询维平均 logits。
        hand_logits = self.hand_head(flat).view(
            batch, BELIEF_PLAYERS, BELIEF_QUERIES_PER_PLAYER,
            HAND_TILE_KINDS, HAND_COUNT_CLASSES,
        ).mean(dim=2)
        shanten_logits = self.shanten_head(flat).view(
            batch, BELIEF_PLAYERS, BELIEF_QUERIES_PER_PLAYER, SHANTEN_CLASSES,
        ).mean(dim=2)
        wait_logits = self.wait_head(flat).view(
            batch, BELIEF_PLAYERS, BELIEF_QUERIES_PER_PLAYER, WAIT_CLASSES,
        ).mean(dim=2)
        danger_logits = self.danger_head(flat).view(
            batch, BELIEF_PLAYERS, BELIEF_QUERIES_PER_PLAYER, DANGER_CLASSES,
        ).mean(dim=2)
        # 回归头输出 logits，按查询平均后再 sigmoid 归一化到 [0,1]
        # （目标 = 点数/24000 clip）。
        loss_logits = self.loss_head(flat).view(
            batch, BELIEF_PLAYERS, BELIEF_QUERIES_PER_PLAYER, LOSS_CLASSES,
        ).mean(dim=2)
        loss_pred = torch.sigmoid(loss_logits)

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
        # 梯度隔离：token_matrix 是「信念 → 策略 token」的接口，只由
        # actor/policy 梯度更新；因此输入摘要必须先 detach，策略/BC 损失
        # 沿 30 个信念 token 回传时止步于转换矩阵，不再进入五头/backbone/
        # belief_query。信念网络（五头 + backbone + query）的梯度只来自
        # 五头监督标签（供 SFT 与 PPO 共用）。
        summary_for_tokens = summary.detach()
        summary_flat = summary_for_tokens.reshape(-1, SUMMARY_DIM)
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
