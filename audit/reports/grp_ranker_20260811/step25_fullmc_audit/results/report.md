# Step 2.5 — High-Budget Teacher Override Audit (Full-Hanchan MC)

## Teacher Override Audit

```text
Total overrides:   30
Supported:        6
Rejected:         2
Unresolved:       22

Override Precision: 0.750
95% CI: [0.409, 0.929]
Mean full-hanchan utility gain: 0.438
Harmful override rate: 0.067
Rank2 precision: 0.667
Rank3 precision: 0.800
Rank4 precision: n/a
```

## Step 2 字段口径（复用，禁止混用）

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

## 方法

- 审计集：Step 2 全部 30 个 `high_confidence_override` decision（Teacher Best = Rank2/3/4 = 11/9/10）+ 30 个按 Policy gap bucket、kyoku stage、π1、Step2 |ΔGRP|、world budget、action type 匹配的 conservative-keep control。
- 每个 decision：A = Policy Top1，B = Teacher Best（override）或 Step2 最强 challenger（keep control）；同一 hidden world 的两个分支只强制首个动作不同，后续全部玩家用 PPO v2 greedy/argmax 继续到整个半庄结束。
- Hidden world 采样沿用 Step 2 的 uniform baseline（牌种守恒、合法手牌、dora 槽位一致），并对每个 world 固定 wall seed，使分支后续 kyoku 的洗牌随机数匹配。
- 最终 reward 直接使用目标座位的实际最终排名：1st=+10, 2nd=+4, 3rd=-4, 4th=-10；全程不使用 GRP V2 leaf value。
- Paired 统计量 D_i = R_B - R_A；adaptive budget 64→128→256（override 与 keep control 上限均为 256），z=1.96 的 CI 连续两个 wave 排除 0 即停。

## 结果

### Override Precision 与 Gain

| 指标 | 值 |
|---|---|
| n_total_override | 30 |
| n_supported | 6 |
| n_rejected | 2 |
| n_unresolved | 22 |
| Override Precision | 0.750 |
| Precision 95% CI | [0.409, 0.929] |
| mean ΔFull | 0.438 |
| median ΔFull | 0.250 |
| SUPPORTED 的 mean ΔFull | 1.986 |

### Harmful Override

| 指标 | 值 |
|---|---|
| harmful_override_count | 11 |
| harmful_override_rate（全部 override） | 0.367 |
| harmful among resolved rate | 0.067 |
| mean harmful loss | -0.365 |
| max harmful loss | -0.844 |

### Matched Keep Controls

- 本运行的样本匹配复用了 4 个 keep decision（同一 decision 的 A/B 审计完全相同），统计时按 decision_id 去重：30 audit rows → 26 unique decisions。

| 指标 | 值 |
|---|---|
| n | 26 |
| keep_supported | 1 |
| keep_rejected | 1 |
| keep_unresolved | 24 |
| keep accuracy（resolved 内） | 0.500 |
| keep 95% CI | [0.095, 0.905] |

### 按 Teacher Rank

| Rank | n | SUPPORTED | REJECTED | UNRESOLVED | Precision | mean ΔFull |
|---|---|---|---|---|---|---|
| 2 | 11 | 2 | 1 | 8 | 0.667 | 0.299 |
| 3 | 9 | 4 | 1 | 4 | 0.800 | 1.072 |
| 4 | 10 | 0 | 0 | 10 | n/a | 0.020 |

### 按 Policy Gap

| Bucket | n | SUPPORTED | REJECTED | UNRESOLVED | Precision | mean ΔFull |
|---|---|---|---|---|---|---|
| lt005 | 11 | 2 | 0 | 9 | 1.000 | 0.104 |
| 05_20 | 8 | 3 | 2 | 3 | 0.600 | 1.183 |
| 20_50 | 5 | 1 | 0 | 4 | 1.000 | 0.175 |
| 50_70 | 4 | 0 | 0 | 4 | n/a | 0.543 |
| ge70 | 2 | 0 | 0 | 2 | n/a | -0.262 |

### Step2 |ΔGRP| Threshold Sweep

| threshold | coverage | n | supported | rejected | unresolved | precision | mean ΔFull |
|---|---|---|---|---|---|---|---|
| 0 | 1.000 | 30 | 6 | 2 | 22 | 0.750 | 0.438 |
| 0.2 | 0.867 | 26 | 6 | 1 | 19 | 0.857 | 0.513 |
| 0.3 | 0.767 | 23 | 6 | 0 | 17 | 1.000 | 0.617 |
| 0.4 | 0.600 | 18 | 6 | 0 | 12 | 1.000 | 0.717 |
| 0.5 | 0.367 | 11 | 4 | 0 | 7 | 1.000 | 0.896 |
| 0.6 | 0.200 | 6 | 2 | 0 | 4 | 1.000 | 1.492 |
| 0.8 | 0.100 | 3 | 2 | 0 | 1 | 1.000 | 2.859 |
| 1 | 0.067 | 2 | 2 | 0 | 0 | 1.000 | 4.086 |
| 1.5 | 0.067 | 2 | 2 | 0 | 0 | 1.000 | 4.086 |
| 2 | 0.033 | 1 | 1 | 0 | 0 | 1.000 | 5.656 |
| 3 | 0.033 | 1 | 1 | 0 | 0 | 1.000 | 5.656 |
| 4 | 0.033 | 1 | 1 | 0 | 0 | 1.000 | 5.656 |
| 6 | 0.033 | 1 | 1 | 0 | 0 | 1.000 | 5.656 |

### Calibration：Step2 ΔGRP vs Full-Hanchan ΔFull

| 范围 | Pearson r (p) | Spearman r (p) | n |
|---|---|---|---|
| override only | 0.891 (0.000) | 0.370 (0.044) | 30 |
| override + keep | 0.845 (0.000) | 0.435 (0.001) | 56 |

| predicted |ΔGRP| bucket | n | mean predicted | mean ΔFull | supported fraction |
|---|---|---|---|---|
| 0-0.2 | 4 | 0.115 | -0.047 | 0.000 |
| 0.2-0.4 | 8 | 0.324 | 0.053 | 0.000 |
| 0.4-0.6 | 12 | 0.492 | 0.329 | 0.333 |
| 0.6-1 | 4 | 0.760 | 0.195 | 0.000 |
| 1-2 | 1 | 1.950 | 2.516 | 1.000 |
| >=2 | 1 | 6.053 | 5.656 | 1.000 |

### Expert Secondary Analysis

- Policy Top1 == Expert: 0.433
- Teacher Best == Expert: 0.233
- Full-MC preferred == Expert: 0.100
- Teacher ≠ Expert 且 Full-MC 支持 Teacher 的案例数: 4

### 性能

- 总 worlds（world-pairs）：14912
- 总 rollouts（分支轨迹）：26752
- 总 policy decisions：9189330
- 总 wave 数：209
- 四 shard 合计运行时间：28291.646s

| shard | decisions | worlds | rollouts/s | policy decisions/s | avg batch | GPU util | CPU core util |
|---|---|---|---|---|---|---|---|
| 0 | 13 | 3712 | 0.81 | 327.04 | 72.23 | 2.9% | 101.1% |
| 1 | 14 | 3840 | 0.74 | 322.28 | 71.91 | 2.6% | 101.1% |
| 2 | 13 | 3712 | 1.24 | 326.27 | 62.77 | 3.4% | 101.0% |
| 3 | 14 | 3648 | 1.21 | 324.66 | 64.04 | 3.4% | 101.0% |

## Success Criteria (Q1–Q8)

**Q1** Step2 的 30 个 high-confidence Teacher override 中，在 full-hanchan expected utility 意义上得到支持的为 6（0.750 precision，unresolved 22 个不计数）。
**Q2** Teacher Override Precision = 0.750 (95% CI [0.409, 0.929])，是否足以作为 Reranker 监督信号见最终结论。
**Q3** Teacher override 的 mean ΔFull = 0.438，median = 0.250；SUPPORTED 子集的 mean = 1.986。
**Q4** harmful override：11 个（全部 override 中 0.367；resolved 中 0.067），mean loss = -0.365，max loss = -0.844。
**Q5** Step2 |ΔGRP| 与 full-hanchan override reliability：Spearman r = 0.370（p=0.044）；threshold sweep 见上表。
**Q6** Policy gap 的 gate value：按 gap bucket 的 precision/mean ΔFull 见上表；high-gap 中 rare override 的行为见 gap_override_audit.csv。
**Q7** Rank4 override 审计：n = 10，precision = n/a，mean ΔFull = 0.020。
**Q8** Matched keep controls：n = 26，keep_supported = 1，keep_rejected = 1，keep_unresolved = 24，resolved 内 keep accuracy = 0.500。

## Correctness Audit

- 8 decisions × [0, 1, 2, 3, 4] worlds
- checks: 520/520 passed
- 结束场分布：wind=0 2 分支；wind=1 77 分支；wind=2 1 分支

## Limitations

- Uniform hidden-world baseline：不包含基于弃牌、立直、副露、手切/摸切等行为信息的 posterior inference。
- 分支在 renchan 路径分叉后，后续 kyoku 的 wall 会按各自已消耗的随机数流继续（同一 world seed + 同一 shuffle seed 流，属于 matched randomness 设计）；只有 public 牌种完全一致时 wall 才逐值相同。
- `4p-red-half` 环境按 30000 目标分规则可能在无人达到 30000 时延入西场（本审计中观察到 1 例），这是环境官方规则，非实现错误。
- Expert action 仅作 secondary analysis，不参与 ground truth。

## 结论

**CONDITIONAL GO**

依据：
- Override Precision = 0.750
- harmful override rate（resolved 内）= 0.067
- mean ΔFull = 0.438
- Step2 |ΔGRP| 与 ΔFull 的 Spearman = 0.370
- matched keep accuracy = 0.500

CONDITIONAL 的具体条件（进入 Step 3 的 label generation 范围）：

1. **|ΔGRP| gate**：只对 Step2 |mean ΔGRP| ≥ 0.3 的 override 生成监督标签。本样本中该区域 resolved precision = 1.000（6 supported / 0 rejected，mean ΔFull = 0.617）；全部 2 个 REJECTED 都落在 |ΔGRP| < 0.3。
2. **候选集合**：Rank4 的 10 个 high-confidence override 在 256 worlds 下全部 UNRESOLVED（mean ΔFull ≈ 0.02），没有证据支持 Rank4 纳入正式候选；建议 Step 3 先使用 Top3 候选集，Rank4 需要单独提高预算复验后再决定。
3. **统计保守性**：30 个 override 中只有 8 个在 256 worlds 内达到统计判定，precision 的 95% Wilson CI 为 [0.409, 0.929]，较宽；正式大规模 label generation 前建议先用本 gate 在小批量上复验一次。

```json
{
  "override_precision": 0.75,
  "harmful_among_resolved_rate": 0.06666666666666667,
  "mean_full_hanchan_gain": 0.43793402777777773,
  "spearman_predicted_vs_observed": 0.36982644440291285,
  "keep_accuracy": 0.5,
  "go_thresholds": {
    "precision>=": 0.8,
    "harmful_rate<=": 0.1,
    "mean_gain>=": 0.5,
    "spearman>=": 0.3
  },
  "conditional_thresholds": {
    "precision>=": 0.6,
    "harmful_rate<=": 0.2,
    "mean_gain>": 0.0
  }
}
```
