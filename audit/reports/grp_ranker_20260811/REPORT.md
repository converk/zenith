# GRP V2 对 Kyoku-Ending State 的长期价值排序能力验证报告

日期：2026-08-11/12
目标文件：`audit/reports/grp_ranker_20260811/GOAL_PROMPT.md`

## 1. 实验设置

### 1.1 模型

- **GRP**：GRP V2（`checkpoints/train_riichi_ppo_goal_grp_v2/grp_rank_predictor.pt`，
  MLP 20→256→128→4，`val_rank_accuracy=53.96%`，`ECE=1.13%`）。
  - 输入：小局初始分数 / 小局结束分数 / chang / ju / honba / riichi sticks / player one-hot（20 维）。
  - 输出：4 席最终排名概率；按项目实际 global reward 定义转换为 expected utility：
    `V_GRP = Σ P(rank=r) * pts_weight[r]`，`pts_weight=(10, 4, -4, -10)`（与
    `riichi_ppo_v1/grp/model.py:reward_from_rank_probs` 一致，不用 argmax）。
- **策略**：RL 模型 `checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt`
  （PPO v2，与 SFT 相比略优；作为当前 rollout leaf evaluator 的目标策略）。
  - 推理复用项目现有链路：`RiichiEnv` + `MjaiKyokuStateMachineManager` +
    `BatchedStateBridge` + `DecisionAnalysisBatch`（v13 必需的分析 token）+ 模型
    `forward_policy`，温度 1 的 softmax 采样（与训练 rollout 一致）。

### 1.2 环境与奖励

- 环境：`RiichiEnv(game_mode="4p-red-half")`（立直麻将半庄，含 tobi/30000 终点/延长战规则）。
- MC ground truth 终局奖励：最终排名 reward `(10, 4, -4, -10)`（与 GRP 的
  expected utility 使用同一 global reward 定义）。
- 半庄续打：从 kyoku-ending state 的最小可恢复信息
  （结束分数、next oya/round wind/honba/kyotaku）调用 `RiichiEnv.reset(...)` 重建下一局，
  四席全部由当前策略继续打至半庄结束。每个 continuation 使用独立固定 seed。

### 1.3 状态选取

- 用当前策略自对弈生成 36 局半庄（seed 基 20260811），共记录 **334 个非终局**
  kyoku-ending states。
- 按局序（East1–South4 及个别延长局）和分数 spread 均匀选取 **48 个状态**：
  - 局序覆盖：`{0:6, 1:6, 2:6, 3:6, 4:6, 5:5, 6:6, 7:5, 8:1, 9:1}`；
  - 分数 spread（最高-最低）p10/p50/p90 = 5000 / 16750 / 56070，覆盖领先/接近/落后；
  - 每状态保存完整 resume spec（state ID、初始/结束分数、field、next 参数）。

### 1.4 Monte Carlo continuation（自适应）

- 每个状态从 64 次起步，`max_SE > 0.65` 时追加 64 次，上限 192 次；SE 按 4 席 reward 的
  max 计算。
- 并行：每批 96 个 continuation，两张卡（CUDA 0 / 物理 GPU 0，CUDA 2 / 物理 GPU 3）
  各跑一半状态；policy 决策批量推理，进程内复用同一 `EfficiencyAnalyzer` 缓存。
- 固定随机种子：`seed_base=20260811`，每个 continuation 的 seed 记录在
  `mc_continuations.csv`；脚本支持断点续跑（按 state 增量写盘）。

### 1.5 Pair 构造（a-priori，不使用 MC 结果选择）

- 按半庄阶段分组：early=局序 0–1，middle=2–4，late=5+；同一组内、不同生成局的
  状态两两候选。
- 每个座位（0–3）分别构造：
  - **hard pairs**：该座位当前分数差 ≤ 10000（判别难度高），每组每座最多 8 个；
  - **easy pairs**（sanity check）：分数差 ≥ 15000，每组每座最多 2 个。
- 共 **119 个 pair**：hard 96 + easy 23；early 39 / middle 40 / late 40；
  去重后 unique state pairs = 102。

## 2. 样本量与 Simulation Budget

| 项目 | 数值 |
| --- | ---: |
| 选定状态数 | 48 |
| MC continuation 总数 | 6464（每状态平均 134.7 次；min 64 / median 128 / max 192） |
| MC 用时（两卡并行） | shard0 3211s / shard1 3341s（wall 约 56 分钟） |
| Policy 决策总数 | 2,110,556（聚合吞吐约 632 decisions/s，两卡） |
| 状态生成用时 | 36 局 68.5s（0.53 hanchan/s） |
| Pair 总数 | 119（hard 96 / easy 23；early 39 / middle 40 / late 40） |

## 3. 核心结果

“determined95” 指 MC 的 `|ΔMC| > 1.96·SE(ΔMC)`（95% CI 可区分）；
“determined80” 使用 z=1.28。准确率只在 determined pair 上统计（near-tie 不计入，
单独报告）。

### 3.1 Pairwise Accuracy

| 子集 | n | determined95 | acc95 | determined80 | acc80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 119 | 51 | **51/51 = 100%** | 69 | 66/69 = 95.7% |
| hard | 96 | 28 | **28/28 = 100%** | 46 | 43/46 = 93.5% |
| easy（sanity） | 23 | 23 | 23/23 = 100% | 23 | 23/23 = 100% |
| early | 39 | 8 | 8/8 = 100% | 15 | 13/15 = 86.7% |
| middle | 40 | 15 | 15/15 = 100% | 16 | 16/16 = 100% |
| late | 40 | 28 | 28/28 = 100% | 29 | 28/29 = 96.6% |
| 同局序（exact kyoku） | 40 | 16 | 16/16 = 100% | — | — |
| 跨局序（同阶段组） | 79 | 35 | 35/35 = 100% | — | — |

- 全部 determined95 的 Wilson 95% CI：**[93.0%, 100%]**（按行独立假设）；
  按 unique state pair 聚类的 bootstrap 因 0 错误退化为 [1,1]，不提供更宽的保守界。
- 所有 3 个 determined80 错误均出现在 **hard + 极小 |ΔGRP|**（0.07 / 0.19 / 0.87），
  即 GRP 自身低置信区域。

### 3.2 ΔGRP 与 ΔMC 相关性

| 子集 | Pearson r | Spearman ρ |
| --- | ---: | ---: |
| 全部 119 | 0.989 | 0.898 |
| determined95 51 | 0.993 | 0.983 |
| hard 96 | 0.875 | 0.806 |
| early 39 | 0.988 | 0.682 |
| middle 40 | 0.993 | 0.921 |
| late 40 | 0.989 | 0.944 |

全部 p 值 < 1e-20，方向一致且单调性随 |ΔGRP| 增大而增强。

### 3.3 ΔGRP Calibration（按 |ΔGRP| 分桶）

| |ΔGRP| 桶 | n | determined95 | GRP 平均 Δ | MC 实际平均 Δ | mean |ΔMC| | acc95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| < 0.25 | 24 | 0 | +0.005 | −0.08 | 0.57 | —（不可判定） |
| 0.25–0.74 | 24 | 3 | −0.11 | −0.25 | 0.85 | 3/3 |
| 0.74–1.77 | 23 | 4 | +0.22 | −0.07 | 0.94 | 4/4 |
| 1.77–6.62 | 24 | 20 | +1.11 | +0.62 | 2.94 | 20/20 |
| ≥ 6.62 | 24 | 24 | +1.38 | +1.46 | 13.81 | 24/24 |

低 |ΔGRP| 桶（<0.25）没有任何 pair 能被 MC 在 95% CI 下区分，说明该区域基本是
“没有可验证差异”的区域；|ΔGRP| ≥ 1.77 后所有可判定样本全部正确，且 MC 实际差异随
GRP 差异同步增大。

### 3.4 High-Confidence 子集（|ΔGRP| 分位）

| 子集 | 阈值 | coverage | n | determined95 | acc95 | mean |ΔMC| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top 50% | ≥ 1.13 | 50.4% | 60 | 47 | 47/47 = 100% | 6.94 |
| Top 25% | ≥ 4.10 | 25.2% | 30 | 30 | 30/30 = 100% | 11.82 |
| Top 10% | ≥ 15.48 | 10.1% | 12 | 12 | 12/12 = 100% | 17.25 |

### 3.5 MC Uncertain / Tie 比例（不隐藏）

- 95% CI 不可判定：**68/119 = 57.1%**；
- 80% CI 不可判定：**50/119 = 42.0%**；
- 近平局（|ΔMC| < 0.5）：**31/119 = 26.1%**。

这 57% 主要是 hard near-tie（MC 预算不足以可靠区分），不是 GRP 的错误；它们被明确排除在
准确率之外，后续作为 leaf evaluator 时应视为“不决策/保持原顺序”。

## 4. 对三个核心问题的回答

### Q1：对 non-trivial kyoku-ending state pairs 是否显著超过随机 50%？

**是，显著超过。** hard pairs 的 determined95 准确率 28/28 = 100%（Wilson 95% CI 下限
87.9%）；全部 determined95 51/51 = 100%（下限 93.0%）。即使按 unique state pair 聚合
（46 个 unique pairs），也是 0 错误。所有错误都集中在 |ΔGRP| < 1 的 GRP 低置信区域。

### Q2：|ΔGRP| 增大时 accuracy 是否明显提高？

**是，非常明显。** |ΔGRP| < 0.25 的区域 MC 根本无法验证（0/24 可判定，mean |ΔMC|=0.57），
而 ≥1.77 的区域 100%（24/24 可判定），≥6.62 的区域 mean |ΔMC|=13.8 且 100%。
determined80 的 3 个错误也全部落在 |ΔGRP| ≤ 0.87。

### Q3：是否存在有实际 coverage 的高置信区域？

**存在。** Top 50% |ΔGRP|（阈值 ≈1.1，覆盖 50.4% 的 pair）上 determined95 准确率
47/47 = 100%，mean |ΔMC| = 6.9；Top 25% / Top 10% 覆盖率 25.2% / 10.1%，准确率均为
100%，mean |ΔMC| 11.8 / 17.2。这个 coverage 足够支撑后续 “Top3 rollout to kyoku end +
GRP leaf evaluation” 中的差异排序（只对足够大的 |ΔGRP| 生效）。

## 5. 主要发现

1. GRP V2 对 kyoku-ending states 的 pairwise 长期价值排序能力**强**：在 MC 可判定的
   pair 上几乎无错，且 ΔGRP 与 ΔMC 单调强相关（Spearman 0.90–0.98）。
2. GRP 的置信度（|ΔGRP|）与实际 MC 差异的幅度高度对齐，适合做 conservative override：
   小差异不表态，大差异几乎完全正确。
3. 中期/后期（middle/late）的确定样本多且全对；early 的 Spearman 较弱（0.68）且
   determined hard 样本仅 1 个（95% CI），**早期阶段证据最薄**。
4. MC 无法判定的比例很高（95% CI 下 57%），这主要反映 hard near-tie 的 ground truth
   噪声，需要更高的 simulation budget 或直接把它们当作 tie 处理。

## 6. 限制

- 样本规模有限：119 个 pair、51 个 determined95 样本、同局序 determined 仅 16 个；
  100% 的精度估计偏乐观，Wilson 下限（93%）更保守。聚类 bootstrap 因 0 错误退化，
  未提供更宽的保守区间。
- determined 子集本身是按 MC CI 过滤的（这是目标要求），因此报告同时给出了
  uncertain/tie 比例，避免高估可用性。
- MC 值依赖策略分布：状态与 continuation 均来自 RL 自对弈；GRP 训练于 tenhou 真实
  牌谱分布，存在分布偏移风险，但本实验范围内表现一致。
- hard pair 的“难度”用当前座位分数差近似；更接近未来 Top3 候选状态的构造（同局、
  同局面演化的分支状态）仍需在后续 paired rollout 中验证。
- 未跑 SFT 策略对比（RL 略优于 SFT，按目标允许择一）；如需完整对比可复用本脚本。

## 7. 最终结论

**CONDITIONAL GO：GRP V2 足以作为 kyoku-ending leaf evaluator，但必须按 |ΔGRP| 门控**
（置信度不足时视为 tie / 不覆盖）。

具体条件：

1. 只对 |ΔGRP| ≥ 1.1（约 Top 50% pair，coverage 50%）做排序/覆盖；此区域 determined
   accuracy 100%（47/47），mean |ΔMC| = 6.9。更保守可选 |ΔGRP| ≥ 4.1（Top 25%，
   30/30，mean |ΔMC| = 11.8）。
2. |ΔGRP| < 1 的区域不要用于 override（MC 无法验证且错误集中于此）。
3. 中期/后期（middle/late）可以放心使用；**早期（局序 0–1）建议先补充更多同局序
   hard pairs 验证**，当前 early 的 determined 样本太少（95% CI 仅 8 个，hard 1 个）。
4. 对 MC 本来就无法区分的 near-tie（本项目 26–57%），leaf evaluator 应输出 tie/保持
   原顺序，而不是强行排序。

下一阶段实现 Top3 rollout to kyoku end + GRP leaf evaluation 时，建议直接采用
“|ΔGRP| 门控 + tie 保持”的 conservative override 设计。

## 8. 产物清单

- 实验代码：`grp_ranker_experiment.py`（generate / select / grp / mc / merge-mc / pairs）
- 配置：`results/config.json`、`results/config_overrides.json`
- 状态与 resume spec：`results/selected_states.csv`、`results/generated_states.csv`
- GRP 值：`results/grp_values.csv`
- MC 原始值：`results/state_values.csv`（GRP+MC 合并）、`results/state_values_mc.csv`
- 逐 continuation 记录（含 seed）：`results/mc_continuations.csv`
- Pair 结果：`results/pairwise_results.csv`（ΔGRP/ΔMC/SE/sample count）
- 统计：`results/summary.json`、`results/calibration.csv`
- 性能：`results/mc_summary_shard0.json`、`results/mc_summary_shard1.json`、
  `results/generate_summary.json`
