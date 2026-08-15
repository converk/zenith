# Contract: V16 GRP 模型、输入与训练

## 1. 模型

```text
Input → Linear 64 → 2-layer GRU(hidden=64) → Linear 64→32 → SiLU
      → Linear 32→4 → Rank Softmax
```

- 总参数 50–70K。
- 输出:P(rank=1..4)。
- 生命周期:离线训练后完全冻结,PPO 不更新;只在每个小局边界执行一次。

## 2. 输入

一条 GRP sequence 对应完整一个半庄;每个玩家按 SELF/RIGHT/ACROSS/LEFT 旋转到
统一视角(每半庄生成 4 个 player-relative samples)。

每个小局 boundary 输入 = 当前比赛状态 + 上一小局结果:

| 当前比赛状态 | 编码 |
|--------------|------|
| 四家当前点数 | 连续归一化 |
| SELF 与其余三家分差 | 连续归一化 |
| SELF 当前排名 | 类别 embedding |
| 场风 / 局数 | 类别 embedding |
| 庄家相对位置 | 类别 embedding |
| honba | 连续归一化 |
| riichi sticks | 连续归一化 |

| 上一小局结果 | 编码 |
|--------------|------|
| 四家 score delta | 连续归一化 |
| 结果类型 | 类别 embedding |
| winner seat | 类别 embedding(流局 N/A) |
| deal-in seat | 类别 embedding(流局/自摸 N/A) |
| 流局 tenpai mask | 类别 embedding |
| 庄家是否连庄 | 类别/位 |

第一小局的 previous-result 固定为 START。最终拼接约 40–60 维后投影到 64;禁止
加入完整手牌、牌河、牌山等高维信息。

## 3. 训练

- 数据:Tenhou 2024+2025,与 SFT 相同的 40% 采样与 train/validation 划分。
- 标签:每个 prefix 监督该视角玩家的**最终排名**。
- 损失:L = (1/K) Σ_k CE(P_φ(rank | s_{0:k}), rank_final)。
- 产物:`checkpoints/train_riichi_v16/grp`(含配置快照);训练后冻结。

