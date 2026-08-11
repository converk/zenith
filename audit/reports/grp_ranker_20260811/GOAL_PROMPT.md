# Goal：验证 GRP 对 Kyoku-Ending States 的长期价值排序能力

## 补充说明（执行前必读）

1. GRP 模型请使用 **GRP V2**。
2. SFT 模型使用 `checkpoints/train_riichi_v13_sft/best_heuristic.pt`；如果需要使用强化学习训练的模型，使用 `checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt`。
3. 根据我们之前的测试：强化学习训练的模型比 SFT 模型好了一些，但不多。两个模型都提供给你，具体用哪个由你根据实验情况自行判断。
4. 当前可用的 CUDA 设备为 **CUDA=0 和 CUDA=2**，分别对应物理 **GPU 0** 和 **GPU 3**；如果需要多卡或并行实验/采样，使用这两张卡。

---

## 一、背景

当前项目是一个立直麻将 AI。

已经存在一个训练完成的 GRP（Global Reward Prediction）模型，其设计参考 Suphx，用于根据当前半庄进度和局间状态预测最终半庄结果/最终排名概率。

当前 GRP 在验证集上的最终排名预测表现如下：

| 半庄第几局 | 预测最终排名准确率 |
| ----- | --------: |
| 第 1 局 |    34.25% |
| 第 2 局 |    40.03% |
| 第 3 局 |    45.12% |
| 第 4 局 |    50.00% |
| 第 5 局 |    55.01% |
| 第 6 局 |    61.75% |
| 第 7 局 |    71.10% |
| 第 8 局 |    89.54% |

整体指标：

* `val_rank_accuracy = 53.96%`
* `val_top2_accuracy = 78.02%`
* `ECE = 1.13%`
* 随机排名预测基线约为 25%

这里需要注意：

GRP 在半庄早期预测某一次真实最终排名的准确率较低，并不能直接说明 GRP 对长期状态价值的估计较差。

因为真正需要估计的是：

```text
E[final utility | current kyoku-ending state]
```

而不是准确预测某一次具体半庄最终会获得第几名。

当前后续计划是把 GRP 用作 Top3 candidate rollout 的 leaf evaluator：

```text
candidate action
    ↓
rollout 到当前小局结束
    ↓
kyoku-ending state
    ↓
GRP
    ↓
estimated long-term value
```

但在实现完整的 Top3 rollout evaluator 之前，必须先独立验证：

> GRP 是否能够可靠地区分两个相近的 kyoku-ending states 中，哪一个具有更高的长期期望收益。

本 Goal **只完成这一项验证**。

---

# 二、核心研究问题

给定两个处于相同或相近半庄阶段的 kyoku-ending states：

```text
S_A
S_B
```

GRP 分别给出：

```text
V_GRP(S_A)
V_GRP(S_B)
```

其中 GRP 如果输出的是最终排名概率：

```text
P(rank=1)
P(rank=2)
P(rank=3)
P(rank=4)
```

需要按照当前项目实际使用的最终 rank reward / global reward 定义转换为 expected utility：

```text
V_GRP(S)
=
Σ P(rank=r | S) * reward(r)
```

不要使用：

```text
argmax rank
```

作为 state value。

然后从 `S_A`、`S_B` 分别进行多次完整 continuation simulation，直到半庄结束，得到：

```text
V_MC(S_A) = mean(final_reward from S_A)
V_MC(S_B) = mean(final_reward from S_B)
```

最终验证：

```text
sign(
    V_GRP(S_A) - V_GRP(S_B)
)
```

是否与：

```text
sign(
    V_MC(S_A) - V_MC(S_B)
)
```

一致。

核心问题是：

> GRP 是否具备足够可靠的 pairwise long-term value ranking 能力？

---

# 三、任务范围

## 必须完成

实现一个最小但可靠的实验流程，完成：

```text
kyoku-ending states
        ↓
GRP value
        ↓
Monte Carlo continuation value
        ↓
pairwise comparison
        ↓
统计结果
```

需要：

1. 从现有数据、模拟环境或已有轨迹中获得一批合法的 `kyoku-ending states`。

2. 对每个状态计算 GRP expected utility。

3. 从这些状态继续模拟到半庄结束，多次采样未来轨迹，计算 Monte Carlo expected utility。

4. 构造具有实际判别意义的 state pairs。

5. 比较：

   * GRP value difference
   * Monte Carlo value difference

6. 输出足够判断 GRP 是否可作为后续 rollout leaf evaluator 的统计结果。

---

# 四、本 Goal 明确不做

不要扩展任务范围。

当前阶段不要实现：

* PPO Top3 rollout
* candidate action reranking
* Top3 Q model
* pairwise neural ranker
* MCTS / tree search
* 新的 PPO 训练
* GRP 重新训练
* GRP 大规模调参
* ranker distillation
* online override mechanism

本 Goal 唯一目标是：

> **验证现有 GRP 对 kyoku-ending state 的相对长期价值排序是否可靠。**

---

# 五、状态选择

不需要做几十万级别的大规模验证。

目标是：

> 使用尽可能少、但足以得出可信工程结论的数据和 simulation budget。

优先选择有代表性的状态。

至少覆盖多个不同半庄阶段，例如：

```text
第1局结束
第2~3局结束
第4~5局结束
第6~7局结束
```

如果数据允许，可以按具体 kyoku 分别统计。

同时尽量覆盖不同点棒形势：

```text
领先
接近
落后
```

---

# 六、Pair 构造原则

不要主要测试这种 trivial pair：

```text
S_A:
当前玩家 45000 点，大幅第一

S_B:
当前玩家 10000 点，大幅第四
```

这种 pair 即使正确也无法说明 GRP 能够服务未来 Top3 reranking。

真正需要重点测试的是：

> **相对接近、具有一定判别难度的 states。**

因为未来 Top3 candidate rollout 后得到的几个 kyoku-ending states 通常不会完全不同，而是比较接近。

因此 pair 应优先满足：

```text
相同或接近的半庄进度
+
当前玩家整体局势相对接近
+
GRP / MC value 差异不是极端巨大
```

可以同时保留少量 easy pairs 作为 sanity check，但最终结论应主要关注 non-trivial / hard pairs。

---

# 七、Monte Carlo Ground Truth

Monte Carlo continuation 应从指定 kyoku-ending state 一直模拟到整个半庄结束。

最终 reward 必须与当前 PPO / GRP 系统实际使用的 global reward 定义保持一致。

对于状态 `S`：

```text
S
├── continuation 1 → R1
├── continuation 2 → R2
├── ...
└── continuation N → RN
```

计算：

```text
V_MC(S) = mean(R1 ... RN)
```

同时计算：

```text
standard error
confidence interval
```

不要只保存平均值。

---

# 八、Simulation Budget 不要求固定

不要求所有状态机械地运行非常大量 continuation。

优先采用：

```text
progressive / adaptive sampling
```

例如先运行一个较小数量：

```text
N = 32 或 64
```

如果 Monte Carlo value 已经足够稳定，可以停止。

如果两个 state 的 value 很接近、confidence interval 很大，则继续增加：

```text
128
256
512
...
```

直到：

1. value difference 足够稳定；
2. 或达到合理的计算上限。

不要求解决本身无法可靠区分的 near-tie。

---

# 九、Near-Tie 处理

非常重要：

如果：

```text
V_MC(S_A) ≈ V_MC(S_B)
```

并且 Monte Carlo 置信区间不足以可靠区分二者，

不要强制生成：

```text
A > B
```

或者：

```text
B > A
```

应标记为：

```text
uncertain / tie
```

这些 pair 不应简单计入普通 pairwise accuracy 的错误样本。

需要单独报告数量和比例。

最终 evaluator 的目标不是区分数学意义上极其微小的 value difference，而是可靠判断有实际收益差异的 states。

---

# 十、核心指标

至少输出以下指标。

## 1. GRP Pairwise Accuracy

仅统计 MC ground truth 足够确定的 pair：

```text
sign(ΔGRP) == sign(ΔMC)
```

其中：

```text
ΔGRP = V_GRP(S_A) - V_GRP(S_B)

ΔMC  = V_MC(S_A) - V_MC(S_B)
```

得到：

```text
pairwise_accuracy
```

---

## 2. 按半庄阶段统计 Pairwise Accuracy

例如：

```text
kyoku 1 ending: xx%
kyoku 2 ending: xx%
...
```

如果单个 kyoku 样本不足，可以分组：

```text
early
middle
late
```

重点观察：

> 即使 GRP 在半庄前期最终排名 classification accuracy 较低，其 pairwise value accuracy 是否仍然具有明显预测能力。

---

## 3. ΔGRP 与 ΔMC 的相关性

统计：

```text
Pearson correlation
Spearman rank correlation
```

重点关注：

```text
ΔGRP 越大
是否通常意味着 ΔMC 也越大
```

---

## 4. ΔGRP Calibration

将 pair 按：

```text
abs(ΔGRP)
```

或者带符号的 `ΔGRP` 分桶。

例如根据实际数据分布自适应选择区间：

```text
very small
small
medium
large
```

每个 bucket 输出：

```text
样本数量
GRP 预测平均 Δ
MC 实际平均 Δ
pairwise accuracy
```

目的是判断：

> 当 GRP 表示“一个状态明显优于另一个状态”时，这种判断是否明显更加可信。

这一点对于未来 conservative override 非常重要。

---

## 5. MC Uncertain Rate

输出：

```text
多少 pair 因为 Monte Carlo uncertainty 太大而无法可靠确定顺序
```

不要隐藏这些样本。

---

# 十一、特别建议增加一个 High-Confidence 子集指标

除了整体 pairwise accuracy，再定义：

```text
GRP high-confidence pairs
```

即 `|ΔGRP|` 较大的 pair。

阈值不要拍脑袋固定，应结合实际 `ΔGRP` 分布选择若干 percentile 或候选 threshold。

例如观察：

```text
Top 50% |ΔGRP|
Top 25% |ΔGRP|
Top 10% |ΔGRP|
```

分别统计：

```text
coverage
pairwise accuracy
mean ΔMC
```

这可以直接回答未来非常重要的问题：

> 如果只在 GRP 非常有把握的时候相信它，排序精度能达到多少？

---

# 十二、成功标准

本实验不是为了证明一个人为指定的理论结论。

需要通过数据判断。

但最终至少应该能够明确回答下面三个问题：

### Q1

GRP 对 non-trivial kyoku-ending state pairs 是否具有显著超过随机 50% 的长期价值排序能力？

### Q2

当 `|ΔGRP|` 增大时，pairwise accuracy 是否明显提高？

### Q3

是否存在一个具有实际 coverage 的区域，使 GRP pairwise accuracy 足够高，可以合理作为后续：

```text
Top3 rollout to kyoku end
+
GRP leaf evaluation
```

的基础？

如果上述结论成立，则实验最终结论：

```text
GRP suitable as kyoku-ending leaf evaluator
```

如果只有 late-game 状态可靠，也必须明确报告：

```text
GRP suitable only in specific game stages
```

如果 pairwise value prediction 本身接近随机或者没有稳定 calibration，则明确判定：

```text
GRP currently unsuitable as Top3 rollout leaf evaluator
```

不要为了得到正面结论而修改评价标准。

---

# 十三、工程原则

优先复用项目中现有：

* 麻将 simulator / environment
* GRP inference
* policy inference
* reward definition
* state serialization
* 半庄 continuation 逻辑

不要为了实验重新实现整套麻将环境。

实验代码尽量独立，例如：

```text
scripts/
evaluation/
experiments/
```

具体目录根据当前仓库结构自行判断。

---

# 十四、性能要求

当前硬件大约为：

```text
2 × NVIDIA L20
```

Policy 模型规模约：

```text
3.5M parameters
```

因此实验应尽量：

* batch policy inference；
* 并行运行多个 continuation；
* 避免逐状态、逐 simulation 的低效 GPU 调用；
* 记录实际 rollout throughput；
* 优先保证实验正确性，其次再优化速度。

本 Goal 不要求为了性能进行大规模架构重构。

---

# 十五、可复现性

实验必须：

* 固定并记录随机种子；
* 保存实验配置；
* 保存选取的 state IDs 或可重建信息；
* 保存每个 state 的 GRP value；
* 保存 MC mean / std / SE / sample count；
* 保存 pair 的 `ΔGRP` / `ΔMC`；
* 可以重新运行得到相同或统计上接近的结果。

---

# 十六、最终输出

最终至少生成：

## 1. 实验代码

能够独立运行整个验证流程。

## 2. 原始结果文件

推荐：

```text
state_values.csv
pairwise_results.csv
summary.json
```

实际名称可以根据项目风格调整。

## 3. 简洁实验报告

报告应包含：

```text
实验设置
样本数量
simulation budget
不同阶段结果
overall pairwise accuracy
high-confidence pairwise accuracy
Pearson / Spearman correlation
ΔGRP calibration
MC uncertain rate
主要发现
限制
最终结论
```

重点回答：

> **现有 GRP 是否足够可靠，可以进入下一阶段的 Top3 paired rollout to kyoku end + GRP expected utility 实验？**

---

# 十七、执行方式

先阅读当前仓库，确认：

1. GRP 模型的输入输出形式；
2. global reward / rank reward 的真实定义；
3. simulator 如何从任意 kyoku-ending state 继续执行；
4. 当前 policy 如何完成剩余半庄；
5. 是否已有保存/恢复完整半庄状态的实现；
6. 哪些现有 evaluation 工具可以直接复用。

然后制定最小实现方案并直接执行。

不要在发现已有实现可以复用时重复造轮子。

如果发现当前架构无法直接从任意 kyoku-ending state continuation，则只实现完成本实验所需的最小状态恢复能力，不进行无关重构。

---

# 十八、最重要的原则

本次实验追求的是：

```text
小规模
+
高质量 Monte Carlo ground truth
+
能够明确判断 GRP 是否可用
```

而不是：

```text
大规模跑分
```

宁愿测试较少、但 MC value 足够稳定的 state pairs，也不要测试大量、但 ground truth 本身噪声巨大的 pair。

最终必须给出明确的 Go / Conditional Go / No-Go 结论：

```text
GO:
GRP 足以作为 kyoku-ending leaf evaluator

CONDITIONAL GO:
GRP 仅在特定阶段或 |ΔGRP| 达到一定程度时可靠

NO-GO:
GRP 对相近状态的长期价值排序能力不足
```

当前 Goal 到这里结束。

不要继续实现 Top3 candidate rollout 或 reranker。
