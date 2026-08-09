# PPO 优化调研：项目证据与引用文件

> 本文档汇总了 Deep Research 提示词中引用的本地证据，供主代理或 Deep Research 快速核对。数据均来自当前工作区，未修改。

## 1. SFT 模型验证指标

来源：`checkpoints/train_riichi_v13_sft/metrics.json`

| 指标 | 数值 |
|---|---:|
| validation/policy_ce | 0.4805 |
| validation/top1 | 0.8166 |
| validation/top3 | 0.9791 |
| validation/action_group_top1 | 0.9375 |
| validation/optimal_shanten_rate | 0.8861 |
| validation/optimal_ukeire_rate | 0.8706 |
| validation/top1_discard | 0.7804 |
| validation/top3_discard | 0.9732 |
| validation/top1_chi | 0.7581 |
| validation/top3_chi | 1.0000 |
| validation/top1_pon | 0.8304 |
| validation/top3_pon | 1.0000 |
| validation/top1_reach | 0.8359 |
| validation/top3_reach | 0.9951 |
| validation/top1_kan | 0.8755 |
| validation/top3_kan | 0.9901 |
| validation/call_pass_accuracy | 0.9454 |

## 2. SFT v11 vs v13 2v2 对战

来源：`checkpoints/train_riichi_v13_sft/v11_vs_v13_2v2_detailed/sft_v11_vs_v13.json`

- 模式：`4p-red-half`，greedy 推理，320 半庄，8 桌并发。
- v11 SFT：队伍胜率 39.84%，平均分差 -8426.25，CI [-12239.41, -4658.70]。
- v13 SFT：队伍胜率 60.16%，平均分差 +8426.25，CI [4658.70, 12239.41]。
- v13 和牌率 23.20%（v11 20.80%），放铳率 7.90%（v11 9.66%），被飞率 10.94%（v11 21.25%）。

## 3. PPO vs SFT 2v2 对战

来源：`checkpoints/train_riichi_ppo/ppo_vs_sft_2v2_detailed/`

| Checkpoint | PPO 队伍胜率 | SFT 队伍胜率 | 平均分差 | 95% CI |
|---|---:|---:|---:|---|
| checkpoint_00050 | 47.34% | 52.66% | -1392.50 | [-5467.50, 2725.02] |
| checkpoint_00100 | 49.69% | 50.31% | -436.88 | [-4783.73, 3861.89] |

两组 CI 均跨 0，不能判定显著差于 SFT，但也没有显著优于 SFT。checkpoint_00100 的立直机会利用率为 72.61%，SFT 对手为 31.43%。

## 4. PPO vs 固定启发式对手（96 半庄）

来源：`checkpoints/train_riichi_ppo/evaluation.jsonl`

| update | mean_rank | first_place | last_place | point_delta_mean | riichi_opportunity_accept |
|---:|---:|---:|---:|---:|---:|
| 15 | 2.094 | 36.5% | 11.5% | +5561.5 | 30.6% |
| 60 | 2.062 | 38.5% | 12.5% | +5844.8 | 36.0% |
| 75 | 1.844 | 52.1% | 10.4% | +10763.5 | 42.1% |
| 90 | 1.979 | 40.6% | 10.4% | +8470.8 | 54.5% |
| 105 | 1.854 | 43.8% | 5.2% | +9221.9 | 57.3% |
| 150 | 2.094 | 41.7% | 14.6% | +5390.6 | 73.5% |
| 180 | 2.177 | 33.3% | 15.6% | +2863.5 | 73.4% |
| 300 | 2.104 | 39.6% | 13.5% | +4713.5 | 58.9% |
| 540 | 2.375 | 32.3% | 26.0% | +1915.6 | 60.3% |
| 720 | 2.188 | 37.5% | 16.7% | +4516.7 | 49.7% |
| 945 | 2.219 | 32.3% | 13.5% | +3196.9 | 63.8% |
| 960 | 2.052 | 34.4% | 5.2% | +5424.0 | 55.0% |

峰值出现在 update 75–105 附近，之后波动且整体没有稳定上升。

## 5. PPO 训练曲线采样

来源：`checkpoints/train_riichi_ppo/metrics.jsonl`

| update | approx_kl | entropy | policy_loss | value_loss | sft_reference_kl | reward/total_mean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00000 | 0.596 | -0.00001 | 0.328 | 0.000 | -0.0021 |
| 50 | 0.00009 | 0.610 | -0.00099 | 0.273 | 0.001 | -0.0004 |
| 100 | 0.00106 | 0.563 | -0.00487 | 0.219 | 0.040 | -0.0005 |
| 150 | 0.00198 | 0.580 | -0.00756 | 0.160 | 0.092 | -0.0004 |
| 200 | 0.00178 | 0.600 | -0.00809 | 0.140 | 0.128 | -0.0012 |
| 300 | 0.00235 | 0.580 | -0.00811 | 0.124 | 0.169 | -0.0002 |
| 500 | 0.00224 | 0.589 | -0.00820 | 0.117 | 0.183 | -0.0005 |
| 700 | 0.00237 | 0.540 | -0.00799 | 0.121 | 0.206 | -0.0005 |
| 950 | 0.00255 | 0.594 | -0.00811 | 0.139 | 0.226 | 0.0001 |

早期 value explained variance 约 0.04，value loss 明显高于 policy loss。

## 6. 当前 reward 实现事实

- `riichi_ppo_v1/training/worker.py`：每个小局结束时对四个座位分别计算 `terminal_kyoku_reward(scores[seat] - start_scores[seat], 24000)`，即 `clip(delta_score / 1000, ±24)`，只写入该小局最后一个 transition 的 `reward`。
- `riichi_ppo_v1/training/worker.py` 中 `record_efficiency(reward=0.0, ...)`：shanten/ukeire 效率信号只做统计，不进入 transition reward。
- `riichi_ppo_v1/training/learner.py`：标准 PPO loss = policy loss + value_coef * value loss - entropy_coef * entropy + sft_kl_coef * KL(policy || frozen SFT reference)。
- 配置文件：`riichi_ppo_v1/configs/training.yaml`，关键超参见提示词。

## 7. 训练数据

来源：`datasets/tenhou_sft_2024_2025/manifest.json`

- games: 363,312
- kyokus: 3,846,384
- decisions: 237,204,275
- train decisions: 234,834,558
- validation decisions: 2,369,717
- raw mjai 源：`datasets/tenhou-to-mjai/2024.zip`、`2025.zip`（内容为 JSON Lines，扩展名 `.mjson`）。

## 8. 关键代码路径

- `riichi_ppo_v1/configs/training.yaml`
- `riichi_ppo_v1/training/worker.py`
- `riichi_ppo_v1/training/learner.py`
- `riichi_ppo_v1/training/train.py`
- `riichi_ppo_v1/training/rewards/terminal.py`
- `riichi_ppo_v1/training/rewards/efficiency.py`
- `riichi_ppo_v1/training/rewards/decision.py`
- `riichi_ppo_v1/training/rewards/public_state.py`
- `riichi_ppo_v1/model/architecture.py`
- `riichi_ppo_v1/model/critic_features.py`
