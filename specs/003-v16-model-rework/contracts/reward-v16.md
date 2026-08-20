# Contract: V16 PPO 奖励与 Top-3 Q-boosting

> V17 修订(2026-08-20):PPO 阶段已完全移除 Q 体系(Q loss 与 Top-3
> Q-boosting 蒸馏)。Advantage 改为纯 value-based GAE(γ, λ),见 §4;
> §1–§3 保留为 V16 奖励契约历史记录。

## 1. 排名 utility

```text
rank 1 → +24
rank 2 → +8
rank 3 → -12
rank 4 → -24
```

和为 -4(末位 -24 为重罚第 4 名,不再严格零和)。终局(半庄结束)使用真实最终
排名的 utility,不使用 GRP 预测。

## 2. GRP 期望与小局 delta

```text
V_GRP = 24·P1 + 8·P2 - 12·P3 - 24·P4
R^GRP_k = V_{k+1} - V_k
```

## 3. 归一化与最终奖励

σ_GRP、σ_Score 在训练数据上离线统计一次后固定,训练过程不得动态修改:

```text
R̂_GRP  = clip(R_GRP / σ_GRP, -10, 10)
R_Score = clip(Δscore / 1000, -24, 24)
R̂_Score = clip(R_Score / σ_Score, -10, 10)
R = 0.7·R̂_GRP + 0.3·R̂_Score
```

70/30 只作用于归一化后的量;不得对原始数值直接加权。V16 奖励范围放大:
utility 为历史值的 2 倍,σ_GRP 保持历史离线固化值不变(GRP 分量实际翻倍),
外层归一化 clip 放宽到 ±10、内层分差截断放宽到 ±24 千点。

## 4. Value-based GAE advantage(V17 起,取代 Top-3 Q-boosting)

1. 小局(kyoku)内按决策顺序对 transitions 计算 GAE(γ, λ):
   `δ_t = r_t + γV(s_{t+1}) − V(s_t)`,小局终局的 `V(s_{t+1}) = 0`;
   `A_t = δ_t + γλ·A_{t+1}`,小局之间互不跨越。
2. reward 为纯 GRP 小局 delta,只落在该小局最后一个 transition 上;Value
   由独立 return loss(与 rollout empirical returns 对齐)训练。
3. PPO policy loss 使用归一化后的 GAE advantage;无 Q scorer、无候选集、
   无 Dueling 基线、无 boosting 蒸馏。
4. 模型结构保留 `q_scorer`(checkpoint 契约兼容),但训练与前向均不再使用。
