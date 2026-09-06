"""V19 逐动作信念读出（BeliefActionReadout）。

信念网络输出三家摘要（28 个信念 token 是全局信息），而策略需要知道
「当前合法动作对应的牌」在信念中的含义。本模块把每个合法动作的
``primary_tile_code`` 对应的信念特征（危险度/打点/听牌/向听等）投影成
``d_model`` 向量，加到 ``pair_hiddens`` 上，使策略头能够逐动作读取信念。

设计要点（实施方案 §3.2）：
- 特征维度 21：3 家 × (danger[tile], loss[tile], tenpai_prob, shanten_expected,
  max_danger, max_loss, wait[tile])；
- ``tile_code==0`` 的动作（pass/无牌类）保留全局项、置零逐牌项；
- 投影矩阵**零初始化**：未训练时读出是 no-op，不影响策略；
- 本模块**不接受任何监督损失**，只由 actor 的 BC/PPO 损失训练；
- ``detach=True`` 时仅 detach 特征张量（阻止 actor 损失修改信念头/共享层），
  读出投影自身仍接收 actor 梯度；``detach=False`` 时按原图回传信念头。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# 三家对手（相对观察者）：0=下家、1=对面、2=上家（与 belief_labels 顺序一致）。
BELIEF_PLAYERS = 3
# 牌种数（34）与听牌 N/A 位。
TILE_KINDS = 34
WAIT_N_A_INDEX = 34
# 逐动作读出特征维度 = 3 家 × 7 项。
FEATURE_DIM = 3 * 7
# 向听类别数（0..8），与 belief_network.SHANTEN_CLASSES 同步。
SHANTEN_KLASSES = 9


class BeliefActionReadout(nn.Module):
    """逐动作信念特征 → d_model 的零初始化读出投影。"""

    def __init__(self, d_model: int = 256, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.feature_dim = int(feature_dim)
        if self.feature_dim != FEATURE_DIM:
            raise ValueError("BeliefActionReadout feature_dim is fixed at 21")
        # 零初始化：读出从 no-op 起步，训练前后向与关闭读出的策略一致。
        self.proj = nn.Linear(self.feature_dim, self.d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # [1,1,21] 布尔掩码：标记哪些特征是「逐牌」项（tile_code=0 时置零）。
        # 顺序：danger(3) + loss(3) + tenpai(3) + shanten(3) + max_danger(3)
        # + max_loss(3) + wait(3)；逐牌项为前 3、次 3 与最后 3。
        tile_column_mask = torch.zeros(1, 1, self.feature_dim, dtype=torch.bool)
        tile_column_mask[..., :6] = True
        tile_column_mask[..., -3:] = True
        self.register_buffer("tile_column_mask", tile_column_mask)

    def forward(
        self,
        belief: dict[str, Tensor],
        tile_codes: Tensor,
        *,
        detach: bool = True,
    ) -> Tensor:
        """构造逐动作信念特征并投影为 ``[B,Q,d_model]``。

        ``belief`` 为 ``BeliefNetwork`` 的输出字典；``tile_codes`` 为每个
        合法动作 Query 行的 ``primary_tile_code``（0=N/A，1..34）。
        """
        tile_codes = tile_codes.to(device=self.proj.weight.device, dtype=torch.long)
        batch, queries = tile_codes.shape
        device = tile_codes.device
        # 越界保护：牌种 1..34 → 聚类到 0..33；``tile_code==0`` 的行随后被
        # 掩码置零（全局项保留），不会错误地读取第 0 种牌。
        clamped = tile_codes.clamp(0, TILE_KINDS - 1)

        danger_prob = torch.sigmoid(belief["belief_danger_logits"].float())
        loss_pred = belief["belief_loss_pred"].float()
        wait_prob = torch.sigmoid(belief["belief_wait_logits"].float())
        shanten_prob = torch.softmax(
            belief["belief_shanten_logits"].float(), dim=-1,
        )
        # 向听期望 = Σ k·P(k)；听牌概率 = 1 - P(N/A)。
        shanten_expected = (
            shanten_prob
            * torch.arange(SHANTEN_KLASSES, device=device, dtype=shanten_prob.dtype)
            .view(1, 1, SHANTEN_KLASSES)
        ).sum(dim=-1)
        tenpai_prob = 1.0 - wait_prob[..., WAIT_N_A_INDEX]
        max_danger = danger_prob.max(dim=-1).values
        max_loss = loss_pred.max(dim=-1).values

        # 逐牌 gather：输入 [B,3,34] / [B,3,35] → [B,Q,3]（玩家序不变）。
        player_tiles = clamped[:, None, :].expand(batch, BELIEF_PLAYERS, queries)
        danger_tile = danger_prob.gather(-1, player_tiles).transpose(1, 2)
        loss_tile = loss_pred.gather(-1, player_tiles).transpose(1, 2)
        wait_tile = wait_prob[..., :TILE_KINDS].gather(-1, player_tiles).transpose(1, 2)
        # 全局项按玩家广播到每个动作。
        tenpai = tenpai_prob[:, None, :].expand(batch, queries, BELIEF_PLAYERS)
        shanten = shanten_expected[:, None, :].expand(batch, queries, BELIEF_PLAYERS)
        max_danger_b = max_danger[:, None, :].expand(batch, queries, BELIEF_PLAYERS)
        max_loss_b = max_loss[:, None, :].expand(batch, queries, BELIEF_PLAYERS)

        features = torch.cat(
            [danger_tile, loss_tile, tenpai, shanten, max_danger_b, max_loss_b, wait_tile],
            dim=-1,
        )
        # tile_code==0：逐牌项置零，全局项保留。
        zero_mask = (tile_codes == 0)[:, :, None]
        features = torch.where(
            zero_mask & self.tile_column_mask,
            torch.zeros((), dtype=features.dtype, device=features.device),
            features,
        )
        if detach:
            features = features.detach()
        return self.proj(features)
