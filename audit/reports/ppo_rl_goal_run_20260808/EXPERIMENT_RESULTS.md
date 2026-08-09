# PPO 优化 Goal 实验结论（最终报告占位）

> 最终结论以外部评测为唯一真相（v13 SFT 2v2 240 半庄 `team_win_rate` 与
> `team_point_diff_paired_bootstrap_ci95`；固定启发式 96 半庄一位率/分差）。
> 训练 reward 不作为成功依据。本节在 E0–E4（→E5）完成后逐项填写。

## 总览

| 实验 | 假设 | 改动 | u200 2v2 vs SFT | 启发式一位率 | EV | 判定 |
|---|---|---|---|---|---|---|
| E0 基线 | 复现现状 | 无 | 待填 | 待填 | 待填 | 待填 |
| E1 EV 门控 | critic 噪声主导 | learner adv_estimator=ev_gate | 待填 | 待填 | 待填 | 待填 |
| E2 对手混合 | competitive overfitting | worker 80/20 SFT 混合 | 待填 | 待填 | 待填 | 待填 |
| E3 GRP A/B | 点差≠排名收益 | worker GRP/混合 reward | 待填 | 待填 | 待填 | 待填 |
| E4 密集奖励 | 信号稀疏 | dense_efficiency_weight | 待填 | 待填 | 待填 | 待填 |

## 逐实验结论

### E0 基线复现

- 假设 / 改动：无算法改动。
- 指标表：待填。
- 判定：待填。

### E1 EV 门控 / 四席组内基线

- 假设 / 改动：待填。
- 指标表：待填。
- 判定：待填。

### E2 对手混合 80/20

- 假设 / 改动：待填。
- 指标表：待填。
- 判定：待填。

### E3 离线 GRP 验证 + PPO A/B

- 离线验证：见 `e3_grp/grp_validation.json`（val rank acc 53.4%、ECE 0.025、Spearman 0.377）。
- PPO 三组：待填。

### E4 密集效率奖励

- 假设 / 改动：待填。
- 指标表：待填。
- 判定：待填。

## 组合最优结果

待填（最优 checkpoint 路径 + 2v2 对比表）。

## 剩余风险与下一步

待填。
