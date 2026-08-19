# Research: Mortal 的 GRP 方案与 Reward 设计

**日期**: 2026-08-19 | **来源**: `https://github.com/Equim-chan/Mortal`
(commit edb448b, 2023-08)

## 1. GRP(Global Ranking Prediction)数据

- 特征:每个 StartKyoku 生成一行 7 维:
  `[grand_kyoku, honba, kyotaku, s0/10000, s1/10000, s2/10000, s3/10000]`
  - `grand_kyoku`: `E1=0, E4=3, S1=4, S4=7, W1=8, W4=11`(超出部分 `_ => 7 +
    kyoku` 的兜底在标准 4 圈半庄下不会出现)
  - `s` 是玩家原始分数 / 10000(如 25000 → 2.5)
- prefix 形式:一个半庄的所有 StartKyoku 前缀序列 `feature[:i+1]
  (i = 0..T-1)` 全部作为输入样本,监督最终四人排名。
- `rank_by_player`:由 `Rankings::new(final_scores)` 得到,分数降序、同分按
  座位号稳定排序(与项目现有 `rank_among` 一致)。
- 训练标签:Mortal 用 24 类全排列 `get_label(rank_by_player)` 映射到
  permutation 索引。

## 2. GRP 模型结构(Mortal `model.py`)

```python
class GRP(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        self.rnn = nn.GRU(input_size=GRP_SIZE, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * num_layers, hidden_size * num_layers),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size * num_layers, 24),  # 4! 排列
        )
```

- `forward`:输入 `packed -> rnn -> state.transpose(0,1).flatten(1) -> fc`
  → 24 logits(只用末层 hidden,不用逐时间步输出)。
- 参数:GRU(7→64, 2 层) + fc(128→128→24) ≈ 42K 参数。
- `perms`/`perms_t`:`register_buffer` 固化 24 个排列及其转置。

## 3. calc_matrix(24 类 → [4,4] 玩家排名概率)

```python
def calc_matrix(self, logits):
    probs = logits.softmax(-1)                       # (N, 24)
    matrix = torch.zeros(batch_size, 4, 4)
    for player in range(4):
        for rank in range(4):
            cond = self.perms_t[player] == rank      # 该玩家在该类排列中排名=rank?
            matrix[:, player, rank] = probs[:, cond].sum(-1)
    return matrix                                     # (N, player, rank)
```

- 语义:`matrix[player, rank]` = P(玩家 player 的最终排名为 rank)。
- 对每行(即每个 player)概率和为 1。

## 4. Reward Calculator(Mortal `reward_calculator.py`)

```python
pts = [3, 1, -1, -3]   # 排名 utility(与本项目要求的 [1, 1/3, -1/3, -1] 差一个常系数 1/3)

def calc_rank_prob(self, player_id, grp_feature, rank_by_player):
    matrix = self.calc_grp(grp_feature)              # (T, player, rank)
    final_ranking = torch.zeros((1, 4))
    final_ranking[0, rank_by_player[player_id]] = 1. # 终局真实排名分布
    rank_prob = torch.cat((matrix[:, player_id], final_ranking))  # (T+1, 4)
    return rank_prob

def calc_delta_pt(self, player_id, grp_feature, rank_by_player):
    rank_prob = self.calc_rank_prob(...)
    exp_pts = rank_prob @ self.pts                    # 每边界的期望 utility
    reward = exp_pts[1:] - exp_pts[:-1]               # δ 序列
    return reward
```

- 每小局 reward = 下一小局开始时 expected rank utility − 本小局开始时 expected
  rank utility;最后一局 = 真实最终排名 utility − 最后一局开始时 GRP expected
  utility。
- Suphx/Mortal 风格;无点差分量、无 σ 归一化(clip 可保留在项目里以稳定幅度)。

## 5. GRP 训练(Mortal `train_grp.py`)

- 优化器:AdamW(项目要求 `lr=1e-5`、`batch_size=512`)。
- 每 `save_every` steps 在 validation 上测 `val_loss`;本项目要求「保存
  validation loss 最低的 checkpoint」(Mortal 只保存固定 state_file,本方案
  强化为 best-by-val-loss)。
- 数据流:按文件批量读 tar/gz 半庄 → 每半庄所有 prefix → pad_sequence +
  pack_padded_sequence → GRU → 24 类 CE。

## 6. 本项目落地差异(V16 → V17)

| 项 | V16 旧 GRP | V17 新 GRP(Mortal) |
|---|---|---|
| 输入 | 9 类别 + 13 数值(4 视角旋转) | 7 维全局状态(无视角旋转) |
| 输出 | 4 类 rank 分布 | 24 类全排列 |
| 结构 | Embedding + Linear + GRU + per-step head | GRU(2×64) + fc(128→128→24)(仅末层) |
| 训练 | batch 64, lr 1e-3, acc-based best | batch 512, lr 1e-5, val-loss-based best |
| reward | 70% GRP(utility 24/8/-12/-24)+30% 点差 | 100% GRP(utility [1,1/3,-1/3,-1]) |
| σ 归一化 | 离线固化 σ_GRP/σ_Score | 无 σ(直接 δ,幅度天然在 ±1 内) |

- PPO rollout 端:V16 的「4 视角 × 每边界 4 次 GRP 调用」改为「每个半庄 1 条
  prefix 序列,每边界末步 1 次 GRU 前向」;调用量从 4×(T+1)×bs 降到 (T)×bs。
- 注意:V16 GrpRollout 需要 terminal_ranks 做终局 reward,新实现同理(用真实
  排名 + 排列标签时可省略 rank_by_player 传递,直接用 rank_utility)。

## 7. Q-Boosting / SFT KL 背景(供 PPO 配置参考)

- V16 教训:entropy 从 0.229→0.110→0.050→0.025 收缩(熵坍缩);KLSFT 0.203→
  0.26→0.29→0.31 漂移。故本方案:熵系数 0.01→0.005 保守退火;SFT KL 用
  U 形调度 0.002→0.0005→0.001 防漂移但允许偏离;Q-Boosting 减弱
  (q_boost_coef 0.05、T=1.5)防后期熵再次快速塌缩。