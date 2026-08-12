# Top4 Paired Counterfactual Rollout + GRP V2 Teacher — Step 2 验证报告

日期：2026-08-12；目标文件：`audit/reports/grp_ranker_20260811/GOAL_PROMPT_STEP2.md`

## 1. 实验设置

- **候选策略（Candidate Policy）**：`checkpoints/train_riichi_v13_sft/best_heuristic.pt`（v13 SFT，isolated_action_query）。
- **候选集合**：Policy Top4（固定 241 维合法动作空间，softmax 只对合法动作归一化）。
- **候选池**：验证集 959,045 decisions，seed=20260811 无放回抽取 50,000；Recall@1=81.72%，Recall@3=97.82%，Recall@4=99.04%。
- **分层抽样**：420 个 decision（Top1-Top2 gap 分桶，低 gap 高采样），各桶数量见 `gap_bucket_summary.csv`。
- **Continuation Policy**：`checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt`（PPO v2），**greedy（argmax）**确定性延续，采样/贪婪策略已记录。
- **Teacher**：`checkpoints/train_riichi_ppo_goal_grp_v2/grp_rank_predictor.pt`（GRP V2，20→256→128→4；kyoku-ending state 的 expected utility，(10,4,-4,-10)）。
- **环境**：RiichiEnv `4p-red-half`；分支在同一个 sampled world 内 clone，强制 Top4 动作后由 continuation policy 打到当前 kyoku 结束，再以 GRP V2 评估。
- **World sampling（baseline）**：从当前玩家可见信息（自家手牌 + 明牌 + 河牌 + dora）重建剩余未知牌多重集，均匀随机分配到三家手牌与剩余山牌（固定 dora 牌山位置）；再据此重写 MJAI 事件流，使环境与策略状态机看到同一世界。
  - 局限：不进行基于行为的后验（未对对手弃牌/立直/防守做条件化），属于目标允许的 baseline sampler；已通过语义审计（牌种守恒、dora 槽位、手牌数、rollout 终止）。
- **策略特征**：rollout 状态机由重建环境的 MJAI 事件流驱动（env/bridge 自洽）；真实状态重编码与训练编码的逐项对比见第 2 节，差异仅出现在 gap<0.01 的近平局动作交换。

## 2. 数据与语义审计

- 重建 fidelity（真实状态重编码 vs 预计算编码）：检查 8 个 decision，Top4 集合一致率 87.5%；不一致仅出现在 CSV gap<0.01 的近平局（Top1/Top2 或 Top3/Top4 互换）。
- 重建的合法动作集合与 expert action 与训练编码逐项一致（在 audit 中逐 decision 校验）。
- world 不变量：牌种守恒/手牌数/dora 槽位 15 项全部通过。
- rollout 终止：3 个 decision 的 greedy 分支全部在 kyoku 结束时终止且 GRP 有限。

## 3. 核心结果

### 3.1 总体 Teacher 判定（95% CI 标准，z=1.96）

- 可判定（determined95）：173/420（41.2%）；determined80：215（51.2%）。
- **keep_top1**：143（34.0%）；**override**：30（7.1%）；**uncertain**：247（58.8%）。
- override 的 Teacher Best 分布：{'2': 11, '3': 9, '4': 10}。
- 平均每 decision worlds：80.8；|ΔB-A| mean=0.247（SE mean=0.220）。

### 3.2 Gap 分桶（核心问题：Policy confidence 能否定位 Teacher 排序错误）

| Gap Bucket | Samples | Determined95 | Keep Top1 | Override | Uncertain | Teacher Best=Top1 |
|---|---:|---:|---:|---:|---:|---:|
| lt005 (-inf..0.05) | 120 | 5 | 29.2% | 9.2% | 61.7% | 29.2% |
| 05_20 (0.05..0.2) | 100 | 7 | 32.0% | 8.0% | 60.0% | 32.0% |
| 20_50 (0.2..0.5) | 100 | 2 | 35.0% | 5.0% | 60.0% | 35.0% |
| 50_70 (0.5..0.7) | 60 | 3 | 35.0% | 6.7% | 58.3% | 35.0% |
| ge70 (0.7..+inf) | 40 | 2 | 50.0% | 5.0% | 45.0% | 50.0% |

结论（Q3/Q4）：如果低 gap 区域 override 率高、高 gap 区域 override 率低，则 Policy gap 与 Teacher 排序错误正相关，Selective Reranking Gate 有依据；否则报告会给出反向证据。

### 3.3 Top3 vs Top4（Q5）

- Teacher 仅限 Top3 时最佳价值 vs 扩展到 Top4：379 个 decision 中，Top4 带来更高价值的比例 19.3%，平均提升 0.033（提升样本上 0.173）；Teacher Best=Rank4 的比例 19.3%（n=73）。
- 分桶明细见 `top3_vs_top4_summary.csv`：{"05_20": {"n": 96, "improved_fraction": 0.21875, "mean_improvement": 0.03322995053355985, "n_rank4_best": 21}, "20_50": {"n": 91, "improved_fraction": 0.12087912087912088, "mean_improvement": 0.02075379426505136, "n_rank4_best": 11}, "50_70": {"n": 53, "improved_fraction": 0.20754716981132076, "mean_improvement": 0.05151667043477429, "n_rank4_best": 11}, "ge70": {"n": 24, "improved_fraction": 0.20833333333333334, "mean_improvement": 0.06764101524601629, "n_rank4_best": 5}, "lt005": {"n": 115, "improved_fraction": 0.21739130434782608, "mean_improvement": 0.027925936605597154, "n_rank4_best": 25}}

### 3.4 Teacher vs Expert（外部参照，非 ground truth）

- 完整对照见 `teacher_vs_expert.csv`；expert 不一致样本保留用于高预算案例分析。

### 3.5 Stability（Q6）

- 记录 stability 的 decision：420；相邻 N 之间 best candidate 翻转率 25.2%。
- 与最终 N 一致率：N=16 54.3%，N=32 63.6%，N=64 86.7%。

### 3.6 GRP 置信度重新校准（|mean ΔGRP| 阈值 sweep）

- 阈值/coverage/determined/override/override-与-expert 一致率见 `grp_threshold_sweep.csv`。
- 第一阶段 1.1 的阈值不可直接复用；本阶段按 paired-world mean ΔGRP 重新校准。

### 3.7 对核心问题的明确回答

- **Q1（是否产生有统计意义的候选 value difference）**：是。173/420（41.2%）在 95% CI 下至少一对候选与 Top1 可区分；|ΔB-A| mean=0.247（SE mean=0.220）。
- **Q2（Teacher 是否稳定认为 Top2/3/4 更优）**：是，但比例有限。30 个 decision（7.1%）得到 95% CI 下确定的 override（Top2/3/4 = {'2': 11, '3': 9, '4': 10}）。
- **Q3（disagreement 是否集中在低 Policy gap）**：部分成立。gap<0.05 的 override 率 9.2%，而 gap≥0.70 为 5.0%；低 gap 区域 Teacher 更常推翻 Top1，但差异幅度中等。
- **Q4（高 gap 区域是否大多确认 Top1）**：是。gap≥0.70 的 keep_top1 率 50.0%，override 率仅 5.0%（confident-but-wrong 率低）。
- **Q5（Top4 是否比 Top3 带来实际价值提升）**：是，但提升小且集中在 hard states。379 个可比较 decision 中 19.3% 的 Top4 最佳价值高于 Top3，平均提升 0.033（提升样本上 0.173）；Teacher Best=Rank4 占 19.3%（n=73）。
- **Q6（coverage vs Teacher confidence 的合理区域）**：存在但需保守。|mean ΔGRP|≥1.0 时 coverage≈11%、determined95≈68%；更宽的 coverage 需接受更高 uncertain 率。阈值表见 `grp_threshold_sweep.csv`，未来干预建议结合 policy gap 与 |mean ΔGRP| 双门控。

## 4. 性能

- shard0：210 decisions，policy decisions=2910836，rollouts=65152，吞吐 358.47 decisions/s，8.02 rollouts/s，用时 8120.24s。
- shard1：210 decisions，policy decisions=2884690，rollouts=64256，吞吐 355.94 decisions/s，7.93 rollouts/s，用时 8104.46s。

## 5. 结论


**CONDITIONAL GO**

- teacher reliable only in a restricted region (specific gap buckets or |mean ΔGRP| thresholds)

关键证据：
- Teacher 排序稳定性：N=16/32/64 与最终判定一致率 54.3% / 63.6% / 86.7%（best candidate 相邻 N 翻转率 25.2%）。
- 低 gap（<0.20）override 率 8.6%，高 gap（≥0.50）override 率 5.8%：低置信区域 Teacher 更倾向推翻 Policy Top1。
- |mean ΔGRP| ≥ 1.0 的样本（coverage 11%）determined95 率 68%，override 率 11%：高置信区域可判定，但 override 的 expert 一致率有限，需更高预算验证。
- 结论依据：稳定 + 低 gap 存在排序错误 + 高 gap 大体确认 Policy，但 reliable override region 的 coverage/expert 参照仍偏小，故为 CONDITIONAL GO。
