"""V19 信念网络：由 player query hidden 生成四头预测、摘要与注入 token。

信念 backbone（1 层、与 Critic 同构）在 ``architecture.py`` 中消费完整
``shared_hidden`` 序列 + 每玩家 3 个查询 token（共 9 个），本模块只负责
「头 + 摘要 + token」：
- 输入 ``player_query_hidden [B,3,3,256]``（玩家 × 3 查询 × d_model）；
- 四个共享逐家小头（256 → 各头输出维度）对**每个查询分别应用**，再按
  查询维取平均 logits，输出形状与模糊化协议完全一致：
  ``belief_hand_logits [B,3,16,3]``（花色×段位 16 组 × 计数桶）、
  ``belief_wait_logits [B,3,5]``（听牌 + 宽度桶，类别 0 = 非听牌）、
  ``belief_danger_logits [B,3,34]``、
  ``belief_loss_bucket_logits [B,3,34,6]``（铳点损失分桶，2026-09-08 起
  取代逐牌精确回归——用户决策：精确预测损失点数压力过大，按
  1000/5000/9000/13000/17000 分桶模糊化）；
- 三家共享同一个线性转换矩阵（121 → 10×d_model），把每家的信念摘要压成
  10 个 256 维 token，作为模型内部产物注入 Actor 尾段（不进 Rust 编码器）。

2026-09-08 信念头精简（用户决策）：删除 shanten 头——是否听牌由 wait 头
类别 0（非听）隐含提供，向听数本身对局面影响小、不值得预测。

信念 token 是策略的一部分：训练/推理同一条前向路径，不依赖外部标签。
``token_matrix`` 结构与训练方式均不变——**只**由 actor/policy 梯度更新，
监督损失不经过 token 路径；且 token 路径的输入为 ``detach(summary)``，
策略/BC 损失止于转换矩阵，不进入四头/backbone/belief_query（梯度隔离，
见 `audit/reports/v19/design/V19_信念网络策略梯度隔离_实施方案.md`）。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# 三家对手（相对观察者：0=下家、1=对面、2=上家，与 belief_labels 顺序一致）。
BELIEF_PLAYERS = 3
# 每玩家查询 token 数（v19 60% 定版：3 个查询）。
BELIEF_QUERIES_PER_PLAYER = 3
# Hand 模糊化：花色×段位 16 组 × 计数桶 {0,1,≥2}。
HAND_GROUPS = 16
HAND_COUNT_CLASSES = 3
# Wait 模糊化：听牌 + 待牌宽度桶 {非听,1,2,3-5,≥6}。
WAIT_CLASSES = 5
# 危险度 34 种牌；铳点损失分桶 34 种牌 × 桶数。
DANGER_CLASSES = 34
TILE_KINDS = 34
# 铳点损失分桶边界（原始点数；用户 2026-09-08 决策：1000-5000 一档、
# 5000-9000 一档，以此类推；上限与旧归一化一致 clamp 到 24000）。
# 类别区间：[0,1k) / [1k,5k) / [5k,9k) / [9k,13k) / [13k,17k) / [17k,24k]。
LOSS_BUCKET_EDGES = (1000.0, 5000.0, 9000.0, 13000.0, 17000.0)
LOSS_BUCKET_CLASSES = len(LOSS_BUCKET_EDGES) + 1  # 6
# 各桶中心（原始点数，末桶取代表点），供期望值特征与可解释指标使用。
LOSS_BUCKET_CENTERS = (500.0, 3000.0, 7000.0, 11000.0, 15000.0, 20500.0)
# 损失归一化上限（与旧 loss_target_norm 一致）。
LOSS_NORM_MAX = 24000.0
# 信念摘要维度 = 48（手牌 16 组 × 3 桶）+ 5（听牌宽度）
# + 34（危险度）+ 34（铳点桶期望，归一化 [0,1]）= 121。
SUMMARY_DIM = (
    HAND_GROUPS * HAND_COUNT_CLASSES + WAIT_CLASSES + DANGER_CLASSES + TILE_KINDS
)
# 每玩家信念 token 数（定版 10/家，不做消融）。
DEFAULT_TOKEN_COUNT = 10


def loss_bucket_targets(raw_loss: Tensor) -> Tensor:
    """把逐牌原始铳点损失映射为分桶类别索引（shape 不变，long）。

    ``torch.bucketize(right=True)``：v 落在 [edges[i-1], edges[i]) 为类 i；
    v=0 → 类 0；v=1000 → 类 1（[1000,5000)）；v=5000 → 类 2；v≥17000 → 末类。
    """
    edges = torch.tensor(
        LOSS_BUCKET_EDGES, device=raw_loss.device, dtype=raw_loss.float().dtype,
    )
    return torch.bucketize(raw_loss.float(), edges, right=True).long()


def loss_bucket_centers(device: torch.device) -> Tensor:
    """各桶中心（归一化到 [0,1]，除以 LOSS_NORM_MAX），shape [K]。"""
    return torch.tensor(
        [center / LOSS_NORM_MAX for center in LOSS_BUCKET_CENTERS],
        device=device, dtype=torch.float32,
    )


class BeliefNetwork(nn.Module):
    """V19 信念网络：player query 特征 → 四头预测 → 三家各 10 token。

    ``forward`` 返回：
    - ``belief_hand_logits``       [B,3,16,3]：16 组计数桶 softmax logits；
    - ``belief_wait_logits``       [B,3,5]：听牌+宽度桶 softmax logits
      （类别 0 = 非听牌，即是否听牌的精确信号）；
    - ``belief_danger_logits``     [B,3,34]：逐格危险度 sigmoid BCE logits；
    - ``belief_loss_bucket_logits``[B,3,34,6]：逐格铳点损失分桶 softmax logits；
    - ``belief_loss_expected``     [B,3,34]：分桶期望的归一化铳点
      （Σ p_k·center_k，取代旧逐牌回归的 belief_loss_pred）；
    - ``belief_summary``           [B,3,121]：三家共享同一拼接顺序的摘要
      （softmax(hand) + softmax(wait) + sigmoid(danger) + loss_expected）；
    - ``belief_tokens``            [B,30,d]：三家 ×10 的注入 token，玩家主序
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
        # 四个共享逐家小头：每个头输出单家单查询的 logits 宽度。
        self.hand_head = nn.Linear(self.d_model, HAND_GROUPS * HAND_COUNT_CLASSES)
        self.wait_head = nn.Linear(self.d_model, WAIT_CLASSES)
        self.danger_head = nn.Linear(self.d_model, DANGER_CLASSES)
        self.loss_bucket_head = nn.Linear(self.d_model, TILE_KINDS * LOSS_BUCKET_CLASSES)
        # 共享转换矩阵：每家 121 维摘要 → 10×d_model token（三家共用同一矩阵）。
        self.token_matrix = nn.Linear(SUMMARY_DIM, self.token_count * self.d_model)

    def forward(self, player_query_hidden: Tensor) -> dict[str, Tensor]:
        """从 player query hidden 生成四个信念头与注入 token。"""
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
            HAND_GROUPS, HAND_COUNT_CLASSES,
        ).mean(dim=2)
        wait_logits = self.wait_head(flat).view(
            batch, BELIEF_PLAYERS, BELIEF_QUERIES_PER_PLAYER, WAIT_CLASSES,
        ).mean(dim=2)
        danger_logits = self.danger_head(flat).view(
            batch, BELIEF_PLAYERS, BELIEF_QUERIES_PER_PLAYER, DANGER_CLASSES,
        ).mean(dim=2)
        # 铳点分桶头：logits 按查询平均后 softmax，取桶期望作归一化预测。
        loss_bucket_logits = self.loss_bucket_head(flat).view(
            batch, BELIEF_PLAYERS, BELIEF_QUERIES_PER_PLAYER,
            TILE_KINDS, LOSS_BUCKET_CLASSES,
        ).mean(dim=2)
        loss_prob = torch.softmax(loss_bucket_logits, dim=-1)
        centers = loss_bucket_centers(loss_bucket_logits.device).view(
            1, 1, 1, LOSS_BUCKET_CLASSES,
        )
        loss_expected = (loss_prob * centers).sum(dim=-1)  # [B,3,34]，[0,1]

        # 三家共享同一摘要拼接顺序：
        # softmax(hand) 展平 48 维 + softmax(wait) 5 维 + sigmoid(danger)
        # 34 维 + loss_expected 34 维 = 121 维。
        hand_feature = torch.softmax(hand_logits, dim=-1).reshape(batch, BELIEF_PLAYERS, -1)
        wait_feature = torch.softmax(wait_logits, dim=-1)
        danger_feature = torch.sigmoid(danger_logits)
        summary = torch.cat(
            [hand_feature, wait_feature, danger_feature, loss_expected],
            dim=-1,
        )  # [B,3,121]

        # 共享转换矩阵按家应用：每家 121 维 → token_count×d，再 reshape 为
        # [B, 3×token_count, d]（玩家主序：rel0 的 10 token、rel1、rel2）。
        # 梯度隔离：token_matrix 是「信念 → 策略 token」的接口，只由
        # actor/policy 梯度更新；因此输入摘要必须先 detach，策略/BC 损失
        # 沿 30 个信念 token 回传时止步于转换矩阵，不再进入四头/backbone/
        # belief_query。信念网络（四头 + backbone + query）的梯度只来自
        # 四头监督标签（供 SFT 与 PPO 共用）。
        summary_for_tokens = summary.detach()
        summary_flat = summary_for_tokens.reshape(-1, SUMMARY_DIM)
        belief_tokens = self.token_matrix(summary_flat).view(
            batch, BELIEF_PLAYERS * self.token_count, self.d_model,
        )
        return {
            "belief_hand_logits": hand_logits,
            "belief_wait_logits": wait_logits,
            "belief_danger_logits": danger_logits,
            "belief_loss_bucket_logits": loss_bucket_logits,
            "belief_loss_expected": loss_expected,
            "belief_summary": summary,
            "belief_tokens": belief_tokens,
        }
