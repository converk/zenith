# V15 Q-Boosting 简要设计

## 目标

V15 保留 V14 已验证有效的：

```text
Strong SFT → PPO
Asymmetric Actor-Critic
per-seat trajectory
privileged Critic
```

不重构 full joint trajectory，不给 Critic 新增 query。

核心改动：

```text
V(s) + GAE
↓
Q(s,a) + Seat-local Q-Boosting
```

同时修复 Actor 中 offense query 没有直接进入最终 policy scoring 的问题。

---

## 1. Actor 修改

V14 每个合法 action 有两个 query：

```text
offense_query(a)
defense_query(a)
```

平均合法 action 约：

```text
7.1 ~ 8
```

所以平均约 14\~16 个 Actor query。

V14 最终：

```text
defense_hidden
    ↓
policy_head
    ↓
logit(a)
```

offense 只能通过 attention 间接影响 defense。

V15 改成：

```text
offense_hidden
      ↓
zero-init Linear(192→192)
      │
      + defense_hidden
      ↓
原 policy_head
      ↓
logit(a)
```

即：

[\
h\_{policy}=h\_{def}+W\_{off}h\_{off}\
]

其中：

```text
W_off = zero init
```

这样初始化时 V15 policy 与旧 SFT/V14 policy 完全一致，之后 PPO 再逐渐学习 offense direct path。

---

## 2. Critic 修改

Critic **不增加任何 query**。

继续使用 V14：

```text
public state
+
opponent hidden hands
+
next 5 wall
+
value_query
    ↓
2-layer critic transformer
    ↓
value_query hidden [192]
```

只把：

```text
Linear(192,1) → V(s)
```

改为：

```text
Linear(192,241) → Q(s,a)
```

即一个 centralized state representation 直接预测所有 241 个动作的 Q。

Illegal action 的 Q 不参与任何 loss 或 expectation。

虽然总动作空间 241，但平均 legal action 只有 7.1\~8，所以实际计算：

[\
E\_\pi[Q]\
]

时只考虑这些合法动作。

---

## 3. Seat-local Q-Boosting

当前代码使用：

```text
pending[env][seat]
```

即一个 seat 的 trajectory 为：

```text
自己的 decision_t
↓
其他玩家动作 + 摸牌 + 环境变化
↓
自己的 decision_t+1
```

V15 不改变这个结构。

因此这不是论文完整 VRPO，而是 **Seat-local Q-Boosting**。

它主要消除：

```text
当前玩家未来自己 action sampling 的方差
```

而其他玩家动作和 Nature randomness 仍作为 environment transition stochasticity。

---

## 4. 核心公式

对于 timestep (t)：

[\
Q\_t=Q(s\_t,a\_t)\
]

计算当前 policy 下：

[\
V\_t^Q=\
\sum\_{a\in legal}\
\pi(a|s\_t)Q(s\_t,a)\
]

Expected-SARSA TD residual：

r\_t+\gamma V\_{t+1}^Q-Q\_t\
]

terminal：

[\
V\_{T}^Q=0\
]

反向计算 λ-trace：

\delta\_t^Q\
+\
\gamma\lambda C\_{t+1}\
]

Q target：

[\
Q\_t^{target}=Q\_t+C\_t\
]

Q-Boosting Advantage：

Q\_t^{target}-V\_t^Q\
]

然后直接把这个 Advantage 交给现有 PPO clipped objective。

建议继续：

```yaml
gamma: 0.99
qboost_lambda: 0.95
```

---

## 5. Rollout 数据

Model forward 输出：

```text
policy_logits [B,241]
q_values      [B,241]
```

在 GPU 上直接计算：

```text
chosen_action
old_logprob
Q_taken = Q(s,a_taken)

expected_Q =
Σ π(a|s)Q(s,a)
```

然后 Worker 只保存：

```text
action
logprob
q_taken
expected_q
reward
critic_features
```

不要把完整 241 Q-values 传回 CPU。

这样 rollout 每个 decision 只比 V14 多保存一个左右的 scalar，避免增加大量通信开销。

---

## 6. Critic Loss

Learner 中：

```text
q_values [B,241]
      ↓ gather(action)
predicted_q
```

只训练实际执行 action：

[\
L\_Q=\
Huber(\
Q(s\_t,a\_t),\
Q\_t^{target}\
)\
]

继续沿用 V14 的：

```text
Huber loss
batch_std target normalization
critic_public_grad_scale
critic bootstrap
```

第一版不要同时修改这些机制。

---

## 7. Critic Bootstrap

继续保留：

```text
前 40 update：
Actor frozen
Public 不接受 Critic gradient
Q-Critic bootstrap

之后：
Actor + Critic joint PPO
critic_public_grad_scale = 0.25
```

Q-boosting 对 Q accuracy 比普通 GAE 更敏感，因此 bootstrap 仍然有价值。

---

## 8. PPO 部分

以下保持 V14：

```text
PPO ratio clipping
old-new KL
SFT reference KL
entropy
Actor LR
Shared LR
Critic LR
opponent mix
point-delta reward
```

变化只有：

```text
GAE Advantage
↓
Q-Boosting Advantage
```

第一版不要同时加入：

```text
半庄排名 reward
belief auxiliary learning
EMA
Critic replay buffer
full joint trajectory
新的 Critic query
```

---

## 9. 主要代码修改

### `architecture.py`

Actor：

```text
新增 zero-init offense projection
offense + defense → 原 policy_head
```

Critic：

```text
value_head: Linear(192,1)
↓
q_head: Linear(192,241)
```

Critic Transformer 和 `value_query` 保持不变。

### `inference.py`

原：

```text
action
logprob
value
```

改：

```text
action
logprob
q_taken
expected_q
```

### `trajectory.py`

原：

```text
value
return
GAE
```

改：

```text
q_taken
expected_q
q_target
Q-boosting advantage
```

### `learner.py`

Critic loss：

```text
Q(s,a_taken) ↔ q_target
```

Actor PPO 逻辑基本不变。

---

## 10. 建议新增监控指标

至少记录：

```text
q_loss
q_taken_mean/std
q_target_mean/std

expected_q_mean/std

qboost_advantage_std

Q explained variance
```

另外推荐记录：

\sum\_a\
\pi(a)\
(Q(s,a)-E\_\pi[Q])^2\
]

它可以直接观察：

```text
当前 policy 下不同合法动作的 Q 差异有多大
```

如果这个值明显大于 0，说明 Q-boosting 要消除的 future action sampling variance 在麻将中确实存在。

---

## 最终 V15 结构

```text
V14
│
├── Actor:
│   defense-only scoring
│
├── Critic:
│   V(s)
│
└── GAE


V15
│
├── Actor:
│   offense + defense scoring
│   offense path zero-init
│
├── Critic:
│   同一个 value_query
│   Linear(192,241)
│   → Q(s,a)
│
└── Seat-local Q-Boosting
    → PPO
```

核心原则：

> 在不重构 V14 rollout 和 PPO 框架的前提下，用 centralized Q-Critic + seat-local Q-boosting 改善 Advantage 质量，同时修复 Actor offense information 没有直接进入最终 action scoring 的问题。

## 参考论文

V15 Q-Boosting / 不完全信息博弈 RL 主要参考以下三篇论文：

1. **GAE Falls Short in Imperfect-Information Self-Play Reinforcement Learning**\
   Zhiyuan Fan, Gabriele Farina, 2026\
   V15 Q-Boosting 的主要参考论文，提出 **Q-Critic、Q-boosting 和 VRPO**。\
   [https://arxiv.org/abs/2605.19235](https://arxiv.org/abs/2605.19235)
2. **Reevaluating Policy Gradient Methods for Imperfect-Information Games**\
   Max Rudolph et al., 2025\
   系统比较 PPO 等 Policy Gradient 方法与 CFR / NFSP / PSRO 等不完全信息博弈算法，说明经过合理训练的 PPO 在 IIG 中仍然具有很强竞争力。\
   [https://arxiv.org/abs/2502.08938](https://arxiv.org/abs/2502.08938)
3. **A Policy-Gradient Approach to Solving Imperfect-Information Games with Iterate Convergence**\
   Mingyang Liu, Gabriele Farina, Asuman Ozdaglar, 2024\
   从理论角度研究 Q-value based Policy Gradient 在不完全信息博弈中的收敛性质，是 QFR 方法的主要论文。\
   [https://arxiv.org/abs/2408.00751](https://arxiv.org/abs/2408.00751)

其中 V15 实现应**重点参考第 1 篇**。当前方案并非完整复现 VRPO，而是在现有 V14 per-seat trajectory 框架下实现：

```
Centralized Q-Critic
+
Seat-local Expected-SARSA(λ)
+
Q-Boosting Advantage
+
PPO clipped objective

```

第 2、3 篇主要用于支撑继续采用 Policy Gradient / PPO 路线以及 Q-based Policy Gradient 在 imperfect-information games 中的研究背景。
