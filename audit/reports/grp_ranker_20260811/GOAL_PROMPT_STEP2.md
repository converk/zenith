# Goal：验证 Top4 Paired Counterfactual Rollout + GRP V2 是否能改进 Policy 候选排序

## 一、项目背景

当前项目是一个立直麻将 AI。

已经拥有一个较强的 SFT / Policy 模型，以及一个经过验证的 GRP V2（Global Reward Prediction）模型。

当前研究已经不再主要关注从整个合法动作空间搜索动作，而是关注：

> **Policy 已经给出的高质量候选动作内部，能否进一步找到长期收益更高的动作。**

整体研究路线为：

```text
Policy
   ↓
High-Recall Candidate Set
   ↓
Counterfactual Evaluator
   ↓
Candidate Reranking
   ↓
未来蒸馏为轻量 Reranker
```

当前 Goal 是其中的第二阶段：

> **验证昂贵 Counterfactual Teacher 本身是否可靠。**

本阶段不要训练最终 reranker。

---

# 二、已有实验基础

## 2.1 Policy 候选召回能力

使用：

```text
checkpoints/train_riichi_v13_sft/best_heuristic.pt
```

模型：

```text
v13 SFT
isolated_action_query head
```

在完整验证集：

```text
959,045 decisions
```

中以固定随机种子：

```text
seed = 20260811
```

随机无放回抽取：

```text
50,000 decisions
```

进行验证。

推理条件：

```text
CPU inference
softmax probability only normalized over legal actions
```

得到：

```text
Recall@1 = 81.78%
Recall@3 = 97.93%
Recall@4 = 99.06%
```

其中：

```text
Recall@4 - Recall@3 = +1.13 percentage points
```

Top3 miss 共：

```text
1033 / 50000 = 2.07%
```

其中：

```text
54.50% 的 Top3 miss
专家动作恰好是 Policy Rank4
```

因此从 Top3 扩展到 Top4，可以直接覆盖约一半的 Top3 candidate miss。

Top4 miss 只剩：

```text
470 / 50000 = 0.94%
```

因此当前后续 Counterfactual Evaluator 的候选集合优先采用：

```text
Policy Top4
```

---

# 三、Policy confidence 与错误高度相关

当前验证中，Top1−Top2 probability gap 与专家标签一致率有明显关系：

| Top1−Top2 Gap |  样本占比 | Top1 Accuracy | Top3 Accuracy |
| ------------- | ----: | ------------: | ------------: |
| < 0.05        |  3.6% |          ~42% |          ~92% |
| 0.05–0.20     |  9.5% |          ~47% |          ~94% |
| 0.20–0.50     | 16.0% |          ~62% |        ~95.8% |
| 0.50–0.70     | 11.7% |         77.1% |         97.8% |
| 0.70–1.00     | 54.0% |         95.5% |         99.5% |
| = 1.00        |  5.2% |          100% |          100% |

整体：

```text
Top1 probability mean   = 0.8161
Top1 probability median = 0.8964

Top1-Top2 gap mean      = 0.6900
Top1-Top2 gap median    = 0.8215
```

因此可以认为：

> **Policy probability gap 是一个很强的 decision difficulty / confidence signal。**

当前研究假设是：

```text
high gap
→ 大部分决策 Policy Top1 已经足够可靠

low gap
→ Top1 更容易出错
→ 但正确动作大概率仍在 Top4
→ 更值得投入额外计算做 reranking
```

但本阶段必须进一步验证：

> 这种关系在“真实长期价值”意义上是否仍然成立，而不仅仅是对专家动作标签成立。

---

# 四、第一阶段 GRP 验证结果

GRP V2 已完成独立验证。

实验规模：

```text
48 kyoku-ending states
6464 adaptive Monte Carlo continuations
119 a-priori pairs
```

最终结论：

```text
CONDITIONAL GO
```

即：

> **GRP V2 具备可靠的 kyoku-ending pairwise 长期价值排序能力，但应根据 |ΔGRP| 进行置信度门控。**

关键结果：

```text
95% MC CI 可判定 pairs:
51 / 51 排序正确 = 100%

hard pairs:
28 / 28 = 100%

easy sanity pairs:
23 / 23 = 100%
```

分阶段：

```text
early:  8 / 8
middle: 15 / 15
late:   28 / 28
```

相关性：

```text
全部 pair:
Pearson  = 0.989
Spearman = 0.898

MC 可判定子集:
Pearson  = 0.993
Spearman = 0.983
```

同时发现：

```text
|ΔGRP| 越大
→ MC 越容易确定
→ 实际 |ΔMC| 越大
→ 排序可靠性越高
```

High-confidence 子集：

```text
Top 50% |ΔGRP|
threshold ≈ 1.1
47 / 47 correct

Top 25%
30 / 30 correct

Top 10%
12 / 12 correct
```

但：

```text
|ΔGRP| < 0.25
```

区域无法通过当前 MC budget 稳定区分。

因此 GRP 应被理解为：

> **高置信状态差异非常可靠，但 near-tie 应允许拒绝排序。**

需要特别注意：

第一阶段中的：

```text
|ΔGRP| ≈ 1.1
```

只是针对两个独立 kyoku-ending states 的经验阈值。

本 Goal 中经过多 world rollout 后得到的是：

```text
E[ΔGRP]
```

其统计分布不同。

因此：

> **不要直接把 1.1 写死为本阶段 override threshold。**

本阶段需要重新校准。

---

# 五、本 Goal 的核心问题

对于当前一个真实 decision state：

```text
I
```

Policy 给出合法动作概率，并选择：

```text
Top4 = {A, B, C, D}
```

其中：

```text
A = Policy Top1
```

当前 baseline 是：

```text
argmax Policy probability
→ A
```

需要构建 Counterfactual Teacher：

```text
                   current state I
                         │
                     Policy Top4
                A       B       C       D
                 \      |      |      /
                  shared sampled worlds
                         │
             force candidate first action
                         │
                rollout to kyoku end
                         │
                      GRP V2
                         │
              expected long-term value
                         │
               rank A / B / C / D
```

核心研究问题：

> **Top4 paired rollout to kyoku end + GRP V2 是否能比 Policy probability 更准确地识别长期收益最高的候选动作？**

---

# 六、本阶段最重要的原则

本阶段不是为了证明：

```text
rollout 一定比 Policy 好
```

而是客观验证：

```text
昂贵 Teacher 是否值得存在
```

最终允许三种结论：

```text
GO
CONDITIONAL GO
NO-GO
```

如果 Teacher 本身不能稳定优于 Policy，就不要继续训练 reranker。

---

# 七、实验输入状态选择

本阶段不要求大量样本。

优先：

```text
小规模
+
高质量
+
足以判断路线是否成立
```

建议从验证集 / 可恢复真实局面中抽取数百级 decision states。

不要求机械固定数量，可以根据 simulation throughput 和结果稳定程度自行调整。

---

# 八、必须进行 Stratified Sampling

不能均匀随机抽样后让大量 easy state 淹没真正有研究价值的 hard state。

必须按照：

```text
Top1 - Top2 probability gap
```

分层采样。

建议至少覆盖：

```text
< 0.05
0.05 – 0.20
0.20 – 0.50
0.50 – 0.70
>= 0.70
```

采样优先级：

```text
< 0.50
→ 高采样

0.50 – 0.70
→ 中等采样

>= 0.70
→ 少量但必须保留
```

原因：

高置信样本必须用于检测：

```text
confident-but-wrong
```

不能因为预计它们正确就完全不做 Counterfactual Evaluation。

---

# 九、必须记录 Policy 信息

每个 decision 至少保存：

```text
state_id
kyoku index
Policy Top1 / Top2 / Top3 / Top4 action
π1
π2
π3
π4
```

同时建议保存：

```text
logπ1
logπ2
logπ3
logπ4

π1 - π2
π1 - π3
π1 - π4

logπ1 - logπ2
logπ1 - logπ3
logπ1 - logπ4

Policy entropy
Top4 cumulative probability
```

这些字段未来都会成为：

```text
Reranker input
Difficulty Gate input
```

的候选特征。

---

# 十、Hidden World Sampling

当前游戏属于不完全信息环境。

对于当前玩家可见状态：

```text
I
```

需要从与当前 information state 一致的隐藏状态分布中采样 possible worlds：

```text
W1
W2
...
WN
```

至少必须满足所有麻将规则与已知信息约束。

严禁：

```text
使用真实日志中的隐藏牌作为唯一 future world
```

然后把它当作当前玩家可知信息。

如果项目已经有 hidden-state reconstruction / sampling 实现，优先复用。

如果当前 sampler 只是根据剩余未知牌随机分配，也可以先作为 baseline，但必须在报告中明确说明其局限。

本 Goal 不要求为了 belief modeling 进行大规模新模型开发。

---

# 十一、Paired Counterfactual Evaluation

这是本实验最重要的工程原则之一。

同一个 sampled world：

```text
Wi
```

必须同时用于：

```text
A
B
C
D
```

即：

```text
Wi
├── force A
├── force B
├── force C
└── force D
```

随后分别 rollout。

不要：

```text
A 使用一组 worlds
B 使用另一组 worlds
```

因为当前任务关注的是候选动作之间的：

```text
relative value difference
```

共享 sampled worlds 可以显著减少比较中的随机方差。

---

# 十二、Rollout 过程

对于候选动作：

```text
a ∈ {A, B, C, D}
```

在 world `Wi` 中：

1. 强制执行 candidate action；
2. 后续所有玩家按照指定 continuation policy 正常行动；
3. rollout 到当前 kyoku 结束；
4. 获得合法的 kyoku-ending state；
5. 使用 GRP V2 计算该状态的 global value。

记：

```text
G[a, i]
```

为 candidate `a` 在 world `i` 下：

```text
kyoku-end GRP value
```

---

# 十三、Continuation Policy

后续 rollout policy 默认优先复用当前强 Policy。

必须明确记录：

```text
self continuation policy
opponent continuation policy
model checkpoint
sampling / greedy strategy
```

为了降低额外噪声，第一版优先采用确定性或可控随机策略。

如果使用 stochastic policy，必须保证 A/B/C/D 之间尽可能采用 matched / paired randomness。

本 Goal 暂时不要为了 opponent modeling 引入复杂新系统。

---

# 十四、Action Value 估计

对于 candidate `a`：

```text
Q_GRP(a)
=
mean_i G[a, i]
```

但真正应重点统计的是 paired difference。

以 Policy Top1：

```text
A
```

为 reference：

```text
ΔBA_i = G[B,i] - G[A,i]
ΔCA_i = G[C,i] - G[A,i]
ΔDA_i = G[D,i] - G[A,i]
```

计算：

```text
mean Δ
std Δ
standard error
confidence interval
sample count
```

同时也可以计算：

```text
B vs C
B vs D
C vs D
```

用于完整 Top4 ranking analysis。

---

# 十五、允许 Tie / Uncertain

绝对不要强迫每个 decision 得到严格：

```text
A > B > C > D
```

麻将中大量动作的真实长期价值可能非常接近。

如果：

```text
ΔQ ≈ 0
```

并且 confidence interval 无法稳定区分：

```text
candidate X
candidate Y
```

必须允许：

```text
tie / uncertain
```

这类状态未来最合理的策略通常是：

```text
保持 Policy Top1
```

而不是让 noisy evaluator 强行 override。

---

# 十六、Adaptive Rollout Budget

不要求每个 decision 固定使用大量 worlds。

优先实现：

```text
progressive / adaptive sampling
```

例如：

```text
N = 16 / 32
```

开始。

如果：

```text
Top1 与其他候选已经明显分离
```

可以停止。

如果：

```text
A / B 很接近
```

则继续：

```text
64
128
256
...
```

直到：

1. 排序证据足够；
2. 或达到合理计算预算；
3. 或判断为 near-tie / unresolved。

---

# 十七、第一项核心实验：Teacher 是否会改变 Policy Top1

统计：

```text
Teacher Best Action
vs
Policy Top1
```

但不能只统计：

```text
override rate
```

必须同时区分：

```text
Teacher 明确认为 Top1 最好
Teacher 明确认为 Top2/3/4 更好
Teacher 无法确定
```

推荐输出：

```text
keep_top1_rate
override_rate
uncertain_rate
```

---

# 十八、第二项核心实验：Policy Gap 与 Teacher Disagreement

这是本阶段最重要的分析之一。

按照：

```text
Top1 - Top2 probability gap
```

统计：

```text
P(
Teacher confidently says Policy Top1 is not best
)
```

例如最终生成：

| Gap Bucket | Samples | Teacher Determined | Top1 Best | Top1 Wrong | Uncertain |
| ---------- | ------: | -----------------: | --------: | ---------: | --------: |
| <0.05      |         |                    |           |            |           |
| 0.05–0.20  |         |                    |           |            |           |
| 0.20–0.50  |         |                    |           |            |           |
| 0.50–0.70  |         |                    |           |            |           |
| ≥0.70      |         |                    |           |            |           |

最终重点画：

```text
x = Policy Top1-Top2 gap

y = P(Teacher says Top1 is wrong)
```

核心问题：

> **Policy confidence 是否不仅能预测专家标签一致率，也能预测真实长期价值排序错误率？**

如果答案为是，则未来 Selective Reranking Gate 具有很强依据。

---

# 十九、第三项核心实验：Top4 的实际贡献

必须单独分析：

```text
Teacher 最优候选是 Top4
```

的情况。

统计：

```text
Teacher best = Policy Rank1
Teacher best = Policy Rank2
Teacher best = Policy Rank3
Teacher best = Policy Rank4
```

特别关注：

> Policy Rank4 是否真的存在有意义的长期价值翻盘样本。

因为 Recall@4 相比 Recall@3 有：

```text
+1.13pp
```

专家动作召回收益。

但最终是否值得 Top4 reranking，还需要证明：

```text
Rank4 带来的真实 value gain
>
新增 ranking noise
```

---

# 二十、Top3 vs Top4 Teacher Benchmark

同一批数据必须同时比较：

```text
Teacher restricted to Top3
```

与：

```text
Teacher restricted to Top4
```

至少输出：

```text
Top3 candidate best value
Top4 candidate best value
```

以及：

```text
Top4 相比 Top3
有多少 decision 找到了更高 Teacher value
平均提升多少
是否主要集中于 hard states
```

这样无需重新生成 Teacher 数据，就能判断未来 reranker 应采用 Top3 还是 Top4。

---

# 二十一、第四项核心实验：Teacher vs Expert Label

Expert action 不能被认为是绝对 ground truth，但仍然是有价值的外部参照。

对于当前验证集决策：

统计：

```text
Policy Top1 == Expert
Teacher Best == Expert
```

以及分 gap bucket 的结果。

但必须明确：

> 本实验的最终目标不是最大化 Expert Action Accuracy，而是验证 Teacher 是否具有合理的 counterfactual long-term value discrimination。

因此如果：

```text
Teacher != Expert
```

不能自动判定 Teacher 错误。

这类 disagreement 应单独保存，未来可用于更高预算验证或案例分析。

---

# 二十二、第五项核心实验：Teacher Stability

必须验证 rollout budget 增加时 ranking 是否收敛。

对于部分 representative decisions：

```text
N=16
N=32
N=64
N=128
...
```

记录：

```text
best candidate
ΔGRP estimate
CI
```

观察：

> candidate ranking 是否随着 simulation budget 增加趋于稳定。

如果大量 decision 在增加 N 后频繁：

```text
A → B → A → C
```

说明 Teacher 本身不稳定，需要重新检查 sampling / rollout / GRP variance。

---

# 二十三、GRP Confidence 在本阶段的重新校准

第一阶段已经发现：

```text
single kyoku-ending state pair
```

下：

```text
|ΔGRP|
```

与长期 MC reliability 高度相关。

但本实验产生的是：

```text
paired-world mean ΔGRP
```

因此必须重新分析：

```text
|mean ΔGRP|
→ ranking stability
→ CI
→ override reliability
```

不要直接固定使用：

```text
1.1
```

作为 threshold。

需要 sweep 多个 threshold，输出：

```text
threshold
coverage
override rate
confidence / stability
```

为未来 conservative override 提供依据。

---

# 二十四、Success Criteria

本 Goal 不预设 Teacher 一定成功。

最终至少回答：

## Q1

```text
Top4 paired rollout to kyoku end + GRP V2
```

是否能够稳定产生具有统计意义的候选 value difference？

---

## Q2

Teacher 是否在一部分 decision 上稳定认为：

```text
Policy Top2 / Top3 / Top4
```

比 Policy Top1 有更高长期价值？

---

## Q3

这种 disagreement 是否主要集中于：

```text
low Policy gap / high uncertainty
```

状态？

---

## Q4

在：

```text
high Policy gap
```

区域，Teacher 是否大多数情况下确认 Policy Top1？

如果不是，需要明确报告：

```text
confident-but-wrong rate
```

---

## Q5

Top4 是否相比 Top3 带来具有实际意义的额外长期价值提升？

---

## Q6

是否能够找到一个：

```text
coverage
vs
Teacher confidence
```

之间合理的区域，用于未来 high-precision override？

---

# 二十五、Go / No-Go 标准

最终给出三种结论之一。

## GO

如果：

```text
Teacher ranking 稳定
+
低 gap 区域能找到明显 Policy 排序错误
+
高 gap 区域大部分确认 Policy
+
存在可靠的 high-confidence override region
```

则：

```text
GO
```

进入下一阶段：

```text
生成更大规模 Teacher labels
→ 训练 lightweight reranker
```

---

## CONDITIONAL GO

如果 Teacher：

```text
仅特定 game stage
仅特定 Policy gap
仅较大 ΔGRP
```

时可靠，则：

```text
CONDITIONAL GO
```

明确限定未来 reranker 的 intervention region。

---

## NO-GO

如果：

```text
paired rollout ranking 高度不稳定
或
Teacher 无法明显优于 Policy probability
或
大量 override 在增加 simulation budget 后翻转
```

则：

```text
NO-GO
```

不要继续蒸馏 reranker。

优先分析：

```text
hidden world sampling
continuation policy
rollout variance
GRP propagation
```

等问题。

---

# 二十六、不要做的事情

本 Goal 明确不要：

* 训练最终 reranker；
* 新增复杂 MCTS；
* 训练新的 Policy；
* 重新训练 GRP V2；
* 扩到 Top10；
* 做大规模 self-play policy improvement；
* 实现最终线上 override 系统；
* 为了实验进行无关架构重构。

当前唯一目标：

> **验证昂贵 Teacher 是否真的具有 Top4 candidate reranking 能力。**

---

# 二十七、工程实现要求

优先阅读并复用当前项目已有：

```text
Policy inference
MJAI / Mahjong environment
state clone / restore
hidden state reconstruction
GRP V2 inference
half-game state representation
kyoku termination
continuation simulation
reward utilities
```

需要特别检查：

1. 是否可以安全 clone 当前完整 simulator state；
2. 是否能够在 clone 后强制执行不同合法 candidate action；
3. 四个 branch 是否能够共享同一 hidden world；
4. 后续 rollout agent 是否严格只使用其合法可见信息；
5. GRP 输入是否与第一阶段验证完全一致；
6. 是否能 batch 处理大量 active rollout states。

---

# 二十八、性能原则

当前硬件：

```text
2 × NVIDIA L20
```

Policy 模型规模约：

```text
3.5M parameters
```

因此优先：

```text
大量环境并行
+
batched Policy inference
```

不要使用：

```text
一个 rollout
→ 一次 GPU forward
```

这种低吞吐方式。

建议维护大量 active trajectories：

```text
active rollout states
        ↓
collect action requests
        ↓
batch model inference
        ↓
scatter actions
        ↓
environment step
```

本实验必须记录：

```text
decision evaluations / second
rollouts / second
model inference batch size
GPU utilization
CPU utilization
平均一个 decision 使用的 worlds 数量
```

但性能优化不能牺牲实验正确性。

---

# 二十九、可复现性

保存：

```text
random seed
sampled decision IDs
Policy checkpoint
GRP checkpoint
experiment config
world sampling seed
continuation policy config
```

对于每个 decision 保存：

```text
Policy Top4
π1~π4
gap
每个 sampled world
每个 candidate 的 GRP result
paired ΔGRP
mean
std
SE
CI
sample count
最终 Teacher ranking
whether determined / uncertain
Expert action
```

保证后续无需重新 rollout 就能完成更多统计分析。

---

# 三十、最终输出文件

推荐至少产生：

```text
decision_summary.csv
candidate_values.csv
paired_rollout_results.parquet
gap_bucket_summary.csv
top3_vs_top4_summary.csv
experiment_summary.json
report.md
```

文件名可以结合当前项目结构调整。

---

# 三十一、最终报告必须回答

最终报告不要只输出大量数字。

必须明确回答：

### 1.

```text
昂贵 rollout + GRP Teacher
是否值得继续？
```

### 2.

```text
Policy gap 是否能有效定位真正需要 rerank 的状态？
```

### 3.

```text
Top4 是否真的比 Top3 值得？
```

### 4.

```text
哪些 decision 可以安全直接使用 Policy Top1？
```

### 5.

```text
哪些 decision 有较高 Policy Improvement 潜力？
```

### 6.

```text
Teacher 需要多少 rollout budget 才趋于稳定？
```

### 7.

```text
是否已经具备生成 reranker 训练数据的条件？
```

最后给出：

```text
GO
CONDITIONAL GO
NO-GO
```

以及明确原因。

---

# 三十二、最重要的执行原则

本阶段仍然遵循：

```text
先证明 Teacher
再训练 Student
```

不要因为最终目标是 reranker，就提前开始训练 reranker。

如果：

```text
paired rollout + GRP
```

本身不能稳定优于：

```text
Policy probability ranking
```

那么训练 Student 没有意义。

反之，如果 Teacher 能在 Policy 不确定区域稳定发现具有明显长期价值优势的 Top2 / Top3 / Top4 动作，同时在高置信区域大多数时候确认 Policy Top1，则证明：

```text
Selective Candidate Reranking
```

路线成立。

当前 Goal 到此结束。

---

# 附：上一 Goal（GRP V2 独立验证）使用的模型与硬件信息（2026-08-11/12）

本文件为 Step 2 目标提示词。执行时请优先参考上一 Goal（`GOAL_PROMPT.md`，即
Top4 paired counterfactual rollout 之前已完成的 GRP V2 独立验证）中实际使用并验证过的
模型、环境与硬件配置：

## 模型路径

- **SFT 策略模型**：
  `checkpoints/train_riichi_v13_sft/best_heuristic.pt`
  （v13 SFT，`isolated_action_query` head；上一步中作为策略候选之一，可用于候选召回与
  continuation policy。）
- **PPO / RL 策略模型**：
  `checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt`
  （PPO v2 checkpoint，`ppo_format_version=2`；上一步实测略优于 SFT，最终被选为状态生成
  与 MC continuation 的主策略，采用温度 1 的 softmax 采样。）
- **GRP V2**：
  `checkpoints/train_riichi_ppo_goal_grp_v2/grp_rank_predictor.pt`
  （MLP 20→256→128→4；`val_rank_accuracy=53.96%`、`val_top2_accuracy=78.02%`、
  `ECE=1.13%`；输入为小局初始/结束分数 + chang/ju/honba/riichi sticks + player one-hot，
  global reward 权重 `(10, 4, -4, -10)`，与 `riichi_ppo_v1/grp/model.py` 一致。）

## 硬件与显卡情况

- Conda 环境：`Mahjong-AI`。
- 本机共 5 张 GPU（`nvidia-smi` 物理编号）：
  - 物理 GPU 0：NVIDIA L20 44GB
  - 物理 GPU 1：NVIDIA L20 44GB
  - 物理 GPU 2：NVIDIA T400 4GB（**不要用于训练/采样**）
  - 物理 GPU 3：NVIDIA L20 44GB
  - 物理 GPU 4：NVIDIA L20 44GB
- 项目 CUDA 编号约定（与 AGENTS.md 一致）：
  - `CUDA_DEVICE=0` ↔ 物理 GPU 0
  - `CUDA_DEVICE=1` ↔ 物理 GPU 1
  - `CUDA_DEVICE=2` ↔ 物理 GPU 3
  - `CUDA_DEVICE=3` ↔ 物理 GPU 4
- 上一 Goal 实际使用：shard0 跑在 `CUDA=0`（物理 GPU 0），shard1 跑在 `CUDA=3`
  （物理 GPU 4，L20）；两卡并行完成全部 MC continuation。
- 上一步任务说明中“当前可用 CUDA=0 和 CUDA=2（分别对应物理 GPU 0 和 GPU 3）”为推荐默认；
  物理 GPU 4 同为 L20 且实测可用，但物理 GPU 1/4 上存在其他用户的常驻任务，使用前应先用
  `nvidia-smi` 确认占用。
- 上一步实测吞吐（供 Step 2 预算参考）：每批 96 个 continuation、两卡并行时聚合约
  632 policy decisions/s；48 个状态 × 6464 次 continuation 约 56 分钟完成。

## 上一步产物（Step 2 可直接复用）

- 实验代码：`audit/reports/grp_ranker_20260811/grp_ranker_experiment.py`
- 完整报告：`audit/reports/grp_ranker_20260811/REPORT.md`
- 结果：`audit/reports/grp_ranker_20260811/results/`（`state_values.csv`、
  `pairwise_results.csv`、`summary.json`、`mc_continuations.csv` 等）
