# Contract: V16 PPO 奖励与 Top-3 Q-boosting

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

## 4. Top-3 Q-boosting

1. Actor 产生 π(a|s),取 Top-1/Top-2/Top-3。
2. Critic Q scorer 只评估候选:输入 [z_critic; h_a],h_a 为 **detach** 的
   动作表示;结构 512→256→SiLU→1,输出原始优势评分 u_i(不再有 241 维
   Action-Q head)。
3. Dueling 约束:候选集合内把 Actor 概率重新归一化为 p_i,计算
   `A_i = u_i - Σ p_j·u_j`,`Q_i = V(s) + A_i`,由构造恒有
   `Σ p_i·Q_i = V(s)`;Value 只负责绝对局面价值,Q 只编码候选间的相对差异。
4. Critic 训练候选 = Top-3 ∪ 实际 rollout 行为动作(最多 4 个);只有行为
   动作的 Q(s,a_t) 回归到与 Value 相同的 rollout return 目标,未执行的
   候选不构造虚假 Q target。
5. Boosting 只使用 Actor Top-3:`p_boost ∝ p_i·exp(λ_q·A_i/T)`,并在 Top-3
   内部重新分配且保持原始总概率质量不变;`p_boost.detach()` 作为 Actor 的
   Top-3 交叉熵辅助蒸馏目标(权重 `q_boost_coef`),不替代 PPO policy loss。
6. Q loss 与 boosting 蒸馏都不得经动作表示直接更新 Actor(detach 保证)。
