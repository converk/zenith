# Goal：Step 2.5 — High-Budget Teacher Override Audit

## 一、背景

当前项目正在研究立直麻将 AI 的 Selective Candidate Reranking。

整体路线为：

```text
Policy
  ↓
High-Recall TopK Candidates
  ↓
Expensive Counterfactual Teacher
  ↓
High-Quality Ranking Labels
  ↓
Future Lightweight Reranker
```

目前已经完成：

```text
Step 1:
GRP V2 kyoku-ending pairwise value validation
        ↓
CONDITIONAL GO

Step 2:
Top4 Paired Counterfactual Rollout
→ kyoku end
→ GRP V2
        ↓
CONDITIONAL GO
```

当前 Goal 为：

```text
Step 2.5:
High-Budget Teacher Override Audit
```

本阶段的唯一核心问题是：

> **Step 2 中 rollout + GRP Teacher 认为应该推翻 Policy Top1 的高置信 decision，经过更高预算、直接模拟到整个半庄结束的真实 final utility 验证后，到底有多少是真正正确的？**

也就是验证：

```text
Teacher Override Precision
```

如果这一指标足够高，则说明当前昂贵 Teacher 可以进入下一阶段：

```text
批量生成 ranking labels
→ 训练 RAG / Learning-to-Rank 风格的轻量 Reranker
```

如果 Teacher override 本身不可靠，则不要进入 Reranker 训练。

---

# 二、已有 Step 2 实验

Step 2 使用：

## Candidate Policy

```text
checkpoints/train_riichi_v13_sft/best_heuristic.pt
```

v13 SFT，`isolated_action_query`。

候选集合：

```text
Policy Top4
```

---

## Continuation Policy

```text
checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt
```

PPO v2。

采用：

```text
greedy / argmax
```

进行 deterministic continuation。

---

## GRP Teacher

```text
checkpoints/train_riichi_ppo_goal_grp_v2/grp_rank_predictor.pt
```

GRP V2：

```text
20 → 256 → 128 → 4
```

最终排名 utility：

```text
rank1 = +10
rank2 = +4
rank3 = -4
rank4 = -10
```

GRP 输出最终排名概率后转换为 expected utility。

---

# 三、Step 2 的核心结果

Step 2 共验证：

```text
420 decisions
```

使用 Top4 paired worlds rollout 到当前 kyoku 结束，再由 GRP V2 估值。

最终判定规则（verdict95）下：

```text
final_verdict_resolved95 = 173 / 420 = 41.2%
```

其中：

```text
conservative_keep = 143
high_confidence_override = 30
uncertain = 247
```

即：

```text
conservative_keep        = 34.0%
high_confidence_override = 7.1%
uncertain                = 58.8%
```

另外，独立统计量：

```text
all_pairwise_resolved95 = 19 / 420
```

表示 B/C/D 三个 challenger-vs-Top1 的 95% CI 全部不跨 0、三个比较全部得到统计方向判定；
它与 `final_verdict_resolved95` 是两种不同概念，禁止混用裸字段名 `determined95`。

30 个高置信 override 中，Teacher Best 分布：

```text
Policy Rank2 = 11
Policy Rank3 = 9
Policy Rank4 = 10
```

因此当前已经证明：

> rollout + GRP Teacher 确实会在一小部分 decision 上高置信地认为 Policy Top2 / Top3 / Top4 比 Policy Top1 具有更高长期价值。

但是：

> **这些 override 目前仍然最终依赖 GRP leaf value。**

因此在进入 Reranker 训练之前，需要使用一个更强、尽量独立于 GRP 的 ground-truth approximation 对这些 override 进行最终审计。

---

# 四、本 Goal 的核心思想

Step 2 Teacher：

```text
candidate action
    ↓
paired rollout
    ↓
当前 kyoku 结束
    ↓
GRP V2
    ↓
estimated global utility
```

Step 2.5 不再在当前 kyoku 结束后使用 GRP。

而是：

```text
candidate action
    ↓
paired continuation
    ↓
当前 kyoku 结束
    ↓
继续模拟之后所有 kyoku
    ↓
整个半庄真正结束
    ↓
actual final rank
    ↓
actual final utility
```

也就是说：

> **使用 Monte Carlo full-hanchan continuation 的最终真实 reward，直接验证 Teacher override。**

---

# 五、主要验证对象

本阶段优先验证 Step 2 中：

```text
30 个 high_confidence_override decisions（final_verdict_resolved95 且 verdict95 为 override_top2/3/4）
```

对于每个 decision：

```text
A = Policy Top1
B = Step2 Teacher Best
```

其中 `B` 可能是：

```text
Policy Rank2
Policy Rank3
Policy Rank4
```

核心比较：

```text
Policy Top1 A
vs
Teacher Override Action B
```

不需要重新对 Top4 全部动作进行同等预算的完整搜索。

当前最重要的是：

> **验证 Teacher 提出的具体 correction 是否正确。**

---

# 六、增加 Matched Keep Controls

除了 30 个 override，还需要抽取一批 matched keep controls。

建议：

```text
约 30 个
```

不要求机械固定为 30，如果匹配条件导致样本不足，可以合理调整。

这些 control 来自：

```text
Step2 final_verdict_resolved95（verdict95 = conservative_keep）
且 Teacher 明确支持 Policy Top1
```

优先按照以下变量进行匹配：

```text
Policy Top1-Top2 gap
kyoku stage
Policy Top1 probability
Step2 |mean ΔGRP|
simulation budget
decision/action type
```

目标是形成：

```text
Override group
vs
Matched Keep group
```

以判断：

> Teacher 不仅能发现 correction，也是否能正确识别“不应该改”的状态。

---

# 七、不要使用 Expert Action 作为 Ground Truth

顶尖玩家 expert action 可以保留作为外部参照，但：

```text
Expert Action
```

不是本阶段 ground truth。

本阶段 ground truth approximation 是：

```text
high-budget paired full-hanchan Monte Carlo expected utility
```

因此：

```text
Teacher != Expert
```

不能直接判 Teacher 错误。

反之：

```text
Teacher == Expert
```

也不能直接判 Teacher 正确。

最终必须根据 full-hanchan expected utility 判断。

---

# 八、Full-Hanchan Counterfactual Evaluation

对于一个 override decision：

```text
A = Policy Top1
B = Teacher Best
```

从当前 information state：

```text
I
```

采样：

```text
W1
W2
...
WN
```

个合法 hidden worlds。

对于每个 world `Wi`：

```text
Wi
├── force A
│      ↓
│   continuation
│      ↓
│   entire hanchan ends
│      ↓
│   R_Ai
│
└── force B
       ↓
    continuation
       ↓
    entire hanchan ends
       ↓
    R_Bi
```

必须优先采用：

```text
paired worlds
+
matched randomness where appropriate
```

计算：

```text
D_i = R_Bi - R_Ai
```

最终：

```text
ΔFull = mean(D_i)
```

以及：

```text
std
SE
confidence interval
sample count
```

---

# 九、最终 Reward

必须直接使用完整半庄最终结果。

例如当前系统定义：

```text
1st = +10
2nd = +4
3rd = -4
4th = -10
```

则：

```text
R_Ai
R_Bi
```

直接根据最终实际 ranking 计算。

不要在 full-hanchan 结束之前使用 GRP V2 bootstrap。

本阶段的目的正是：

> **绕开 GRP leaf approximation。**

---

# 十、Continuation Policy

默认优先沿用 Step 2：

```text
checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt
```

作为后续所有玩家的 continuation policy。

第一版保持：

```text
greedy / argmax
```

以降低 policy sampling noise。

必须确保：

```text
A branch
B branch
```

除了首个强制 candidate action 外：

> 后续 continuation policy 完全一致。

如果未来实际部署 Policy 与这里不同，在报告中明确说明。

---

# 十一、World Sampling

优先复用 Step 2 已通过审计的 baseline sampler：

```text
从当前玩家可见信息重建剩余未知牌
→ 均匀随机分配三家隐藏手牌和牌山
```

必须保持：

```text
牌种守恒
合法手牌数
dora 槽位一致
MJAI / environment state consistency
```

本 Goal 暂时不要开发新的 belief model。

但是必须明确报告：

> 当前 sampler 是 uniform hidden-world baseline，不包含基于弃牌、立直、副露、手切/摸切等行为信息的 posterior inference。

这一点属于实验 limitation，而不是本 Goal 的阻塞项。

---

# 十二、Paired Evaluation 必须优先比较差值

不要主要比较：

```text
mean(R_A)
mean(R_B)
```

而应该直接保存：

```text
D_i = R_Bi - R_Ai
```

然后估计：

```text
mean(D)
SE(D)
CI(D)
```

核心问题是：

```text
E[R_B - R_A] > 0 ?
```

而不是分别高精度估计两个绝对 value。

---

# 十三、Adaptive Sampling

本阶段不要求每个 decision 固定使用相同 simulation budget。

优先使用：

```text
progressive / adaptive sampling
```

例如：

```text
64
→ 128
→ 256
→ 512
→ ...
```

根据实际吞吐和 uncertainty 决定上限。

---

## 提前停止条件

如果：

```text
ΔFull 的 CI 已经稳定完全 > 0
```

说明：

```text
Teacher override supported
```

可以停止。

如果：

```text
CI 完全 < 0
```

说明：

```text
Teacher override rejected / harmful
```

可以停止。

如果：

```text
CI 长期包含 0
```

并且效应量很小：

```text
|ΔFull| ≈ 0
```

则标记：

```text
near-tie / unresolved
```

不要为了强制得到二元结论无限增加预算。

---

# 十四、Near-Tie 必须单独处理

如果：

```text
A 与 B 的长期 expected utility 极其接近
```

则即使 Teacher 排序方向和 MC 最终 sample mean 不一致，也不能简单作为严重错误。

因此每个 override 最终分成至少：

```text
SUPPORTED
REJECTED
UNRESOLVED / TIE
```

---

# 十五、Multiple Comparisons

本阶段主要验证：

```text
Policy Top1
vs
Teacher chosen override action
```

因此每个 decision 优先只有一个主要 hypothesis：

```text
H:
Q(TeacherBest) > Q(PolicyTop1)
```

不要重新对 Top2/Top3/Top4 同时进行大量无约束 pairwise significance tests。

这样可以减少：

```text
multiple-comparison false positives
```

如果确实需要在 secondary analysis 中比较多个候选，必须明确使用：

```text
Holm correction
```

或其他合理 family-wise error control。

但主 Override Precision 必须基于预先指定：

```text
Teacher Best vs Policy Top1
```

的 comparison。

---

# 十六、核心指标 1：Override Precision

这是本阶段最重要的指标。

对于 MC 可判定的 override：

```text
Override Precision
=
SUPPORTED
/
(SUPPORTED + REJECTED)
```

其中：

```text
SUPPORTED:
full-hanchan MC 认为 TeacherBest > PolicyTop1

REJECTED:
full-hanchan MC 认为 TeacherBest < PolicyTop1
```

`UNRESOLVED` 不直接作为错误计入，但必须单独报告。

必须输出：

```text
n_total_override
n_supported
n_rejected
n_unresolved

override_precision
Wilson / bootstrap confidence interval
```

---

# 十七、核心指标 2：Override Gain

只统计方向正确还不够。

需要计算：

```text
mean ΔFull
median ΔFull
```

以及：

```text
SUPPORTED overrides 的 mean ΔFull
```

目的：

> Teacher correction 不仅应该“方向正确”，还应该带来有意义的实际 expected utility gain。

---

# 十八、核心指标 3：Harmful Override

必须重点报告：

```text
ΔFull < 0
```

的 high-confidence Teacher override。

统计：

```text
harmful_override_count
harmful_override_rate
mean harmful loss
max harmful loss
```

由于最终系统目标是：

```text
high-precision override
```

因此 harmful override 是比 override rate 更重要的风险指标。

---

# 十九、核心指标 4：Matched Keep Accuracy

对于 matched keep controls：

Step 2 Teacher 明确认为：

```text
Policy Top1 应保持
```

使用 full-hanchan continuation 比较：

```text
Policy Top1
vs
最有竞争力的 Step2 alternative
```

或者直接使用 Step2 中对应的 strongest challenger。

统计：

```text
keep_supported
keep_rejected
keep_unresolved
```

这能判断 Teacher 是否同时具备：

```text
纠正错误
+
避免不必要 override
```

两种能力。

---

# 二十、按 Teacher Rank 分析

30 个 override 中：

```text
Rank2 = 11
Rank3 = 9
Rank4 = 10
```

必须分别统计：

```text
Teacher Best = Rank2
Teacher Best = Rank3
Teacher Best = Rank4
```

对应的：

```text
SUPPORTED
REJECTED
UNRESOLVED
mean ΔFull
```

这尤其用于回答：

> **Top4 是否真的值得保留。**

如果 Rank4 的 10 个 high-confidence override 在 full-hanchan MC 中大多数成立，则 Top4 的价值获得非常强的支持。

如果 Rank4 override 大量失败，则需要重新评估：

```text
Top3 vs Top4
```

---

# 二十一、按 Policy Gap 分析

必须继续保留：

```text
Top1 - Top2 probability gap
```

分桶：

```text
<0.05
0.05–0.20
0.20–0.50
0.50–0.70
>=0.70
```

对于 override cases 统计：

```text
样本数
SUPPORTED
REJECTED
UNRESOLVED
Override Precision
mean ΔFull
```

但不要因为当前样本较少而过度解读显著性。

当前目的主要是观察：

> high-confidence Policy 状态中的 rare override 是否真的属于 confident-but-wrong。

---

# 二十二、按 Step2 Teacher Confidence 分析

需要保存 Step2 中：

```text
mean ΔGRP
SE
CI
|mean ΔGRP|
world count
```

然后分析：

```text
|mean ΔGRP|
        ↓
Full-hanchan Override Precision
```

建议做 threshold sweep，例如：

```text
>= 0.5
>= 1.0
>= 1.5
>= 2.0
>= 4.0
...
```

具体阈值根据实际数据分布调整。

输出：

```text
threshold
coverage
n
supported
rejected
unresolved
override_precision
mean ΔFull
```

这将直接用于未来：

```text
Conservative Override Gate
```

的设计。

---

# 二十三、最重要的 Calibration 分析

重点分析：

```text
Step2 predicted ΔGRP
vs
Step2.5 observed ΔFull
```

至少输出：

```text
Pearson correlation
Spearman correlation
```

以及分桶：

```text
predicted ΔGRP bucket
n
mean predicted ΔGRP
mean full-hanchan Δ
supported fraction
```

核心问题：

> **Step2 Teacher 不仅能不能判断方向，它预测的优势大小是否也对应真实 full-hanchan expected gain？**

---

# 二十四、不要把单次最终排名作为独立样本

统计单位必须是：

```text
decision
```

而不是：

```text
每一次 rollout trajectory
```

每个 decision 的大量 rollout 只是用于估计：

```text
Q(action)
```

最终 confidence interval / precision 分析必须以 decision-level comparison 为基础。

避免将大量相关 rollout 当成大量独立决策样本，从而虚假提高统计显著性。

---

# 二十五、Policy Probability 不参与 Full-MC Ground Truth

必须继续保持：

```text
Policy prior
```

和：

```text
Teacher / MC label
```

的独立性。

Policy probability：

```text
π1
π2
π3
π4
gap
entropy
```

可以保存并用于：

```text
分析
future reranker feature
gate design
```

但不得参与：

```text
ΔFull ground truth
```

的计算。

不要使用：

```text
ΔFull + λ log π
```

作为本阶段 label。

---

# 二十六、Expert Label 作为 Secondary Analysis

保留：

```text
expert action
```

并统计：

```text
Policy Top1 == Expert?
Teacher Best == Expert?
Full-MC preferred action == Expert?
```

但明确：

```text
Expert != ground truth
```

如果：

```text
Teacher != Expert
但 Full-MC 支持 Teacher
```

这反而可能是非常有价值的案例。

建议将此类 decision 单独保存进行人工案例分析。

---

# 二十七、优先复用 Step 2 数据

不要重新做 Step2 的 420 个 decision 全套实验。

优先读取已有：

```text
decision_summary
candidate_values
paired_rollout_results
teacher_vs_expert
grp_threshold_sweep
```

等结果。

从中直接选取：

```text
30 high_confidence_override decisions
+
matched conservative_keep controls
```

然后进行 full-hanchan continuation。

不要重复已经完成的 kyoku-ending rollout。

---

# 二十八、统计字段口径（已确认，无需再审计）

Step 2 统计字段口径已确认：

```text
all_pairwise_resolved95 = 19
    表示 B/C/D 三个 challenger-vs-Top1 比较全部在 95% CI 下得到方向判定
    （三个 comparison 的 |ΔGRP| 均 > 1.96 × SE，方向为 better 或 worse）。

final_verdict_resolved95 = 173
    表示最终 verdict95 != uncertain，即最终决策规则已给出
    conservative_keep 或 high_confidence_override（override_top2/3/4）。
    其中 conservative_keep = 143，high_confidence_override = 30。
```

两者为**不同统计概念**，后续禁止使用含糊的裸字段名 `determined95`。

`conservative_keep` 定义（最终报告必须复述）：

```text
当前 Teacher 没有足够统计证据证明任何 challenger 优于 Policy Top1，
因此默认保持 Policy Top1；
它不表示已经统计证明 Top1 是 Top4 中的最优动作。
```

`high_confidence_override` 定义：

```text
存在唯一 challenger 的 95% CI 完全高于 Policy Top1，且该 challenger 是
Teacher Best（raw mean 最高的候选），因此最终决策规则输出 override_topN。
```

---

# 二十九、State Reconstruction Fidelity

Step2 audit 中：

```text
8 个 decision
Top4 set consistency = 87.5%
```

不一致仅发生在：

```text
gap < 0.01
```

的 near-tie 排序交换。

本阶段涉及的每一个 override decision 在投入高预算 full-hanchan MC 前，应至少确认：

```text
legal action set 一致
Policy Top1 action 一致
Teacher Best action 合法
candidate action identity 无歧义
```

如果因为状态重建导致：

```text
Policy Top1 / Teacher Best action 发生变化
```

则该 decision 不应直接进入审计，应标记并单独处理。

---

# 三十、Full-Hanchan Runtime Correctness

由于 continuation 长度从：

```text
到当前 kyoku 结束
```

扩展为：

```text
到整个 hanchan 结束
```

必须增加 correctness audit。

至少检查：

```text
连庄
流局
本场
供托
南入/终局逻辑
点数更新
rank settlement
终局条件
```

确保 full-hanchan continuation 完整遵循当前 `4p-red-half` 环境规则。

---

# 三十一、性能原则

当前硬件约：

```text
2 × NVIDIA L20
```

Policy 模型约：

```text
3.5M parameters
```

Full-hanchan continuation 比 Step2 昂贵得多。

因此继续优先：

```text
大量 active trajectories
+
batched inference
+
environment parallelism
```

不要采用：

```text
一个 trajectory
→ 一次小 batch GPU forward
```

的方式。

记录：

```text
rollouts / second
policy decisions / second
average batch size
GPU utilization
CPU utilization
worlds per audited decision
total simulation count
```

但正确性优先于性能。

---

# 三十二、建议的初始 Budget

不要提前写死最终 sample 数。

建议从：

```text
N = 64
```

或根据现有实现吞吐选择类似量级开始。

然后逐级：

```text
64
128
256
512
...
```

使用 adaptive stopping。

对于已经：

```text
CI 明确远离 0
```

的 decision，立即停止追加。

将更多计算预算集中于：

```text
可能影响 Override Precision 判断的边界案例
```

---

# 三十三、Success Criteria

本阶段的最终目标不是追求一个人为指定的好结果。

但需要明确回答以下问题。

## Q1

Step2 的 30 个 high-confidence Teacher override 中：

> 有多少在 full-hanchan expected utility 意义上得到支持？

---

## Q2

Teacher Override Precision 是否足够高，可以作为未来 Reranker 的监督信号？

---

## Q3

Teacher override 带来的：

```text
mean expected utility gain
```

是否具有实际意义？

---

## Q4

是否存在 harmful override？

如果存在：

```text
数量
比例
平均损失
最大损失
共同特征
```

是什么？

---

## Q5

Step2 的：

```text
|mean ΔGRP|
```

是否能够预测 full-hanchan override reliability？

---

## Q6

Policy gap 是否仍然具有 gate value？

尤其：

```text
high-gap 中的 rare override
```

是否真的属于 confident-but-wrong？

---

## Q7

Rank4 override 是否能够通过 full-hanchan audit？

即：

> Top4 是否值得成为未来 Reranker 的正式候选集合？

---

## Q8

Matched keep controls 是否大多数确认 Policy Top1？

---

# 三十四、Go / Conditional Go / No-Go

最终给出：

```text
GO
CONDITIONAL GO
NO-GO
```

---

## GO

如果 high-confidence Teacher overrides 在 full-hanchan MC 中：

```text
具有很高 Override Precision
+
harmful override 很少
+
平均 expected gain 为正且有实际意义
+
Teacher confidence 与真实 gain 有良好关系
```

则：

```text
GO
```

进入 Step 3：

```text
大规模 Teacher Label Generation
+
RAG / Learning-to-Rank 风格 Reranker Training
```

---

## CONDITIONAL GO

如果：

```text
仅 |ΔGRP| 较大
仅某些 kyoku
仅某些 Policy confidence 区域
```

的 override 足够可靠，则：

```text
CONDITIONAL GO
```

并明确未来 label generation 只覆盖可靠区域。

---

## NO-GO

如果：

```text
Teacher high-confidence override
```

在 full-hanchan MC 中大量失败，或者：

```text
harmful override rate 较高
```

则：

```text
NO-GO
```

不要开始训练 reranker。

优先检查：

```text
GRP leaf bias
uniform world sampling
continuation policy
state reconstruction
simulation variance
multiple-comparison effects
```

---

# 三十五、如果 Step 2.5 通过，下一阶段是什么

如果结果为：

```text
GO
```

或明确可控的：

```text
CONDITIONAL GO
```

下一阶段才开始：

```text
Step 3:
Reranker Dataset Generation + Training
```

届时可以正式参考 RAG / Learning-to-Rank：

```text
Policy        ≈ Retriever
Top4          ≈ Retrieved candidates
Policy logits ≈ Retriever scores
Teacher ΔQ    ≈ relevance / utility labels

Policy Top1 wrong but highly ranked
             ≈ hard negative

Pairwise / Listwise / Relative-Q
             ≈ reranker objectives
```

并让未来 Student Reranker 输入：

```text
state
Top4 actions
π1 / π2 / π3 / π4
logits / ranks / gaps
```

但当前 Goal 不实现这些内容。

---

# 三十六、本 Goal 明确不做

不要：

* 训练 Reranker；
* 生成几十万规模 ranking dataset；
* 重训 GRP；
* 重训 Policy；
* 开发复杂 MCTS；
* 扩到 Top10；
* 开发新的 belief model；
* 修改最终线上策略；
* 用 Policy probability 融入 MC label；
* 因为 expert label 不一致就修改 Teacher；
* 为获得更好结论而改变预先定义的审计样本。

---

# 三十七、最终输出

至少生成：

```text
override_audit_results.csv
keep_control_audit_results.csv
teacher_fullmc_calibration.csv
rank4_override_audit.csv
gap_override_audit.csv
threshold_sweep.csv
experiment_summary.json
report.md
```

具体文件名可根据当前项目风格调整。

每个 audited decision 至少保存：

```text
decision_id

Policy Top1 action
Teacher Best action
Teacher Best original Policy rank

π1
π2
π3
π4
Policy gap

Step2 mean ΔGRP
Step2 SE
Step2 CI
Step2 worlds

Full-MC worlds
mean R_policy_top1
mean R_teacher_best

paired mean ΔFull
std ΔFull
SE ΔFull
CI ΔFull

SUPPORTED / REJECTED / UNRESOLVED

expert action
kyoku stage
```

---

# 三十八、最终报告必须给出的最重要结论

最终报告开头必须直接给出：

```text
Teacher Override Audit:

Total overrides:
Supported:
Rejected:
Unresolved:

Override Precision:
95% CI:

Mean full-hanchan utility gain:
Harmful override rate:

Rank2 precision:
Rank3 precision:
Rank4 precision:
```

并同时给出 Step 2 字段口径与 `conservative_keep` 定义：

```text
all_pairwise_resolved95 = 19
final_verdict_resolved95 = 173
  conservative_keep = 143
  high_confidence_override = 30
uncertain = 247

conservative_keep：
当前 Teacher 没有足够统计证据证明任何 challenger 优于 Policy Top1，
因此默认保持 Policy Top1；
它不表示已经统计证明 Top1 是 Top4 中的最优动作。
```

然后明确回答：

> **当前 rollout-to-kyoku-end + GRP V2 Teacher 是否已经足够可靠，可以作为未来 Reranker 的 Teacher？**

最终只能给：

```text
GO
CONDITIONAL GO
NO-GO
```

之一，并说明依据。

---

# 三十九、最重要的执行原则

本阶段不是继续寻找更多 override。

而是：

> **验证已经发现的 override 到底是不是真的。**

优先追求：

```text
少量样本
+
高 simulation budget
+
独立于 GRP 的 full-hanchan final utility
+
可靠的 Override Precision
```

而不是：

```text
大量 decision
+
低预算
+
噪声很大的结论
```

当前 Step 2.5 到此结束。

如果 Teacher Override Precision 通过验证，再进入 Step 3。

---

# 附 A：Step 2 已完成工作与文件路径（2026-08-12）

Step 2（Top4 Paired Counterfactual Rollout + GRP V2）已实施、验证并提交：
Git commit `c4a19f0`，消息 `step2采样验证`。

## 新增脚本（`audit/reports/grp_ranker_20260811/step2_top4_rollout/`）

```text
select_candidates.py   候选筛选：50k validation decisions（seed=20260811）+ gap 分层抽样 420
paired_rollout.py      Top4 paired rollout 主引擎（自适应 worlds 16→128；支持 --resume/--only-ids/--append）
semantic_audit.py      业务语义审计（重建 fidelity / world 不变量 / rollout 终止）
analyze.py             分析 + 报告生成（gap 分桶、Top3 vs Top4、stability、threshold sweep、结论）
README.md              使用说明
```

## 结果文件（`audit/reports/grp_ranker_20260811/step2_top4_rollout/results/`）

```text
selected_decisions.csv         420 个审计 decision（含 π1-π4、gap、entropy、Top4 累计概率）
policy_candidates.csv          50k policy sweep 完整结果
policy_sweep_summary.json      Recall@1/3/4 与 gap 分桶统计
decision_summary.csv           420 decisions：verdict95/80、paired Δ、SE、CI、worlds
candidate_values.csv           逐 world 逐 candidate GRP（paired_rollout_results.csv 为等价降级版）
paired_rollout_results.csv     parquet 因环境无 pyarrow 自动降级为 CSV
gap_bucket_summary.csv         按 Top1-Top2 gap 分桶的 Teacher 判定表
top3_vs_top4_summary.csv       Top4 相对 Top3 的价值提升分析
grp_threshold_sweep.csv        |mean ΔGRP| 阈值 sweep
stability_summary.csv          相邻 N 的 best-candidate 翻转
stability_convergence.csv      N=16/32/64 与最终判定一致率
teacher_vs_expert.csv          Teacher/expert 对照
experiment_summary.json        结构化摘要 + conclusion
semantic_audit.json            语义审计结果
report.md                      Step 2 完整报告（结论 CONDITIONAL GO）
rollout_config.json            Step 2 运行配置（模型、seed、预算、z 阈值）
rollout_summary_shard{0,1}.json 两卡性能统计
```

## Step 2 核心数字（供 Step 2.5 直接复用）

```text
420 decisions
final_verdict_resolved95 = 173（conservative_keep = 143；high_confidence_override = 30；uncertain = 247）
all_pairwise_resolved95 = 19（独立统计量）
high_confidence_override 分布：Rank2 = 11，Rank3 = 9，Rank4 = 10
平均 worlds/decision ≈ 80.8；两卡各 210 decisions；~109k rollouts；~5.8M policy decisions
```

## Step 2 报告统计口径说明（对应正文第二十八节）

`decision_summary.csv` 的 `determined95` 字段即 `all_pairwise_resolved95`，只有 **19 个 True**，
表示三个 challenger-vs-Top1 comparison 的 95% CI 全部不跨 0。
`gap_bucket_summary.csv` 的 `n_determined95` 统计的正是该字段，因此各分桶合计 = 19。
而报告 overall 的 173 实际指 `final_verdict_resolved95`（`analyze.py` 中按
`verdict95 != "uncertain"` 统计 = conservative_keep 143 + high_confidence_override 30），
是 **final verdict 层面的 resolved**，与 CSV 的 `determined95` 字段是两种不同口径（同名不同义）。
这不是数据错误；Step 2.5 起统一使用以下命名：

```text
all_pairwise_resolved95    CSV 字段 determined95（19）——三个 challenger-vs-Top1 的 CI 全部不跨 0
final_verdict_resolved95   报告 overall 173——verdict95 != uncertain
conservative_keep          verdict95 == keep_top1（143）——无证据证明任何 challenger 优于 Top1
high_confidence_override   verdict95 以 override 开头（30，Rank2/3/4 = 11/9/10）
```

---

# 附 B：硬件与模型路径

## 显卡（`nvidia-smi` 物理编号）

```text
物理 GPU 0：NVIDIA L20 44GB
物理 GPU 1：NVIDIA L20 44GB
物理 GPU 2：NVIDIA T400 4GB（不要用于训练/采样）
物理 GPU 3：NVIDIA L20 44GB
物理 GPU 4：NVIDIA L20 44GB
```

CUDA 编号映射（与 AGENTS.md 一致）：

```text
CUDA_DEVICE=0 ↔ 物理 GPU 0
CUDA_DEVICE=1 ↔ 物理 GPU 1
CUDA_DEVICE=2 ↔ 物理 GPU 3
CUDA_DEVICE=3 ↔ 物理 GPU 4
```

Step 2 实际使用 **物理 GPU 0 与 3**（CUDA_DEVICE=0,2）两卡并行；
物理 GPU 1/4 上可能存在其他用户的常驻任务，使用前先用 `nvidia-smi` 确认占用。

## 模型路径

```text
Candidate Policy（v13 SFT，isolated_action_query）：
  checkpoints/train_riichi_v13_sft/best_heuristic.pt

Continuation Policy（PPO v2）：
  checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt

GRP V2（20→256→128→4）：
  checkpoints/train_riichi_ppo_goal_grp_v2/grp_rank_predictor.pt
```

运行环境：Conda 环境 `Mahjong-AI`；游戏环境 `RiichiEnv`（`4p-red-half`）；
GRP 最终排名 utility `(10, 4, -4, -10)`。
