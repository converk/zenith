# V19 SFT（2 epochs）训练完成检查

> 检查时间：2026-09-07。数据来源：`logs/v19/sft_train_v19_2ep.log`、
> `checkpoints/train_riichi_v19/sft/metrics.json`、
> `checkpoints/train_riichi_v19/sft/tensorboard/`。

## 1. 运行概况

| 项 | 值 |
|---|---|
| 配置 | `riichi_ppo_v1/configs/v19_sft.yaml`（checkpoint 内嵌配置确认 `epochs=2`） |
| 数据 | `datasets/tenhou_sft_2024_2025_encoded_60pct_v19` |
| 训练步数 | 275,210（2 epochs × 137,605） |
| 验证点数 | 92（每 3000 步 + 最终） |
| 耗时 | 36,597.5 s ≈ 10.17 h |
| 吞吐 | 末尾窗口 ~5,000 samples/s、~526k tokens/s、step ~0.102 s |
| 产物 | `best.pt`（step 270000）、`latest.pt`（step 275210） |

注意：工作区当前 `configs/v19_sft.yaml` 存在未提交改动 `epochs: 2 → 1`，
但运行与 checkpoint 内嵌配置均为 `epochs=2`，请确认该改动是否为有意为之。

## 2. 策略仿射（BC）学习效果

| 指标 | step 3000 | step 90000 | step 150000 | step 270000 | step 275210 |
|---|---:|---:|---:|---:|---:|
| train loss | 4.059 | 2.802 | 2.772 | 2.752 | 2.765 |
| train policy CE | 0.920 | 0.469 | 0.453 | 0.444 | 0.445 |
| val policy CE | 0.743 | 0.472 | 0.460 | 0.4489 | 0.4489 |
| val top1 | 0.731 | 0.818 | 0.822 | 0.825 | 0.825 |
| val top3 | 0.942 | 0.980 | 0.981 | 0.982 | 0.982 |

- 总损失从 4.06 降到 2.75，政策 CE 从 0.92 降到 0.445，Top-1 从 0.670/0.731
  升到 0.828/0.825（train/val），Top-3 升到 0.983/0.982。
- train–val 差距极小（CE 差 ~0.004、top1 差 ~0.002），没有过拟合迹象。
- 曲线在 ~150k 步后进入平台期：150k→270k 验证 CE 仅再降 0.011，说明 2 epochs
  对 BC 已基本收敛；继续加步数的边际收益很小。

动作类型 Top-1（最终训练窗口）：

| pass | discard | reach | chi | pon | kan | hora | ryukyoku |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.975 | 0.793 | 0.857 | 0.770 | 0.862 | 0.925 | 0.9996 | 0.978 |

其中 chi/pon/kan 从早期很低（~0.29/0.58/0.29）上升到 0.77/0.86/0.92，
说明稀有动作也被学到了。

## 3. 信念头评测（验证集）

### 3.1 损失

| 头 | step 3000 | step 90000 | step 150000 | step 275210 |
|---|---:|---:|---:|---:|
| hand loss | 0.739 | 0.701 | 0.696 | 0.691 |
| shanten loss | 1.219 | 1.163 | 1.157 | 1.151 |
| wait loss（tenpai 二判） | 0.254 | 0.236 | 0.234 | 0.232 |
| danger loss | 0.116 | 0.0999 | 0.0975 | 0.0957 |
| loss loss | 0.00410 | 0.00351 | 0.00339 | 0.00331 |
| belief_loss_weighted | 2.551 | 2.364 | 2.339 | 2.319 |
| belief_loss_total | 2.551 | — | — | 2.319 |

五头损失全部单调下降，加权总损失从 2.55 降到 2.32。

### 3.2 质量指标与基线对照

验证数据集统计基线（97 个 validation shard 全量统计）：

| 基线 | 值 |
|---|---|
| hand 全零 argmax 精度 | 0.70999 |
| shanten 多数类（2 向听）精度 | 0.30276 |
| 非听牌多数类占比 | 0.84920 |
| danger 正例占比（=随机 top-k 召回期望） | 0.00710 |
| wait Precision@2 随机期望（tenpai 行，2/34） | 0.05882 |
| wait 条件 AUC 随机期望 | 0.500 |

| 指标 | step 3000 | step 90000 | step 275210 | 基线 | 结论 |
|---|---:|---:|---:|---:|---|
| hand_acc | 0.7100 | 0.7115 | 0.7126 | 0.70999 | 仅 +0.26pp，接近全零基线 |
| shanten_top1 | 0.4589 | 0.4802 | 0.4831 | 0.30276 | +18.0pp，显著有效 |
| wait_tenpai_acc | 0.8829 | 0.8920 | 0.8939 | 0.84920 | +4.47pp，有效 |
| wait_precision@2 | 0.0660 | 0.0484 | 0.0645 | 0.05882 | 仅 +0.006，基本随机 |
| wait_conditional_auc | 0.5258 | 0.5236 | 0.5606 | 0.500 | +0.061，弱但非随机 |
| danger_auc | 0.8711 | 0.9201 | 0.9290 | 0.500 | 强 | 
| danger_recall@topk | 0.0117 | 0.0224 | 0.0259 | 0.00710 | 约 3.6× 随机，但绝对召回低 |
| loss_mae | 0.0303 | 0.0206 | 0.0195 | — | 好 |
| loss_conditional_mae | 0.1473 | 0.1344 | 0.1292 | — | 好（真值均值 ~5459 点） |

### 3.3 加权贡献（最终，验证）

| 头 | 原始 loss | 权重 | 加权贡献 |
|---|---:|---:|---:|
| hand | 0.6914 | 0.7 | 0.484 |
| shanten | 1.1510 | 0.8 | 0.921 |
| wait（tenpai） | 0.2319 | 1.8 | 0.417 |
| danger | 0.0957 | 5.0 | 0.478 |
| loss | 0.00330 | 5.0 | 0.017 |
| wait_danger 软约束 | 0.0275 | 0.05 | 0.001 |

五头加权贡献已同量级（shanten 0.92 略高），阶段 16 的调权目标基本达成。

## 4. 结论

**整体有效**：BC 策略头（val CE 0.743→0.449、top1 0.731→0.825）与信念头的
向听/听牌二判/危险 AUC/打点 MAE 都明显受益；train–val 差距小，2 epochs
收敛合理。

**五头逐个看**：

- ✅ shanten：进步最大（48.3% top1，基线 30.3%）。
- ✅ wait（tenpai 二判）：89.4%，比非听多数类高 4.5pp。
- ✅ danger：AUC 0.929，排名能力好。
- ✅ loss：全局/条件 MAE 均下降。
- ⚠️ hand：top1 只比"永远预测 0"高 0.26pp；当前指标无法体现计数 MAE，
  若决策高度依赖手牌计数，需要补测 MAE 或逐家全对率。
- ⚠️ wait 的牌种级指标仍接近随机（precision@2=0.064、条件 AUC=0.561），
  与 `belief_wait_tile_weight=0.0` 关闭 tile BCE 一致；若下游使用 tile 级
  wait 信息，需要决策是否重开/换监督方式，或从读出里移除 tile 级特征。
- ⚠️ `wait_danger_violation ≈ 0.993` 且全程几乎不变：当前实现
  `danger_probs > wait_probs * danger_labels` 对全部 34×3 位置取均值，
  危险正例仅 0.71%，模型任一非危险位置有正概率即判违反，指标基本恒为 1，
  不建议作为有效信号（可考虑只对真危险/wait 位置统计）。

## 5. 遗留事项

- **未发现 96 半庄最终评测产物**：`audit/reports/v19/eval/` 现有文件为本次
  分析生成，SFT 契约要求的最终 96 半庄信念面评测记录缺失，建议补跑。
- `best.pt`（step 270000，val CE 0.448892）与 `latest.pt`（step 275210，
  val CE 0.448926）几乎等价；若下一步 PPO 从 SFT 初始化，建议用 `best.pt`
  或按 `init_model` 配置引用。
- 当前 `configs/v19_sft.yaml` 的 `epochs: 1` 未提交改动与 checkpoint 内嵌
  `epochs=2` 不一致，请确认。

## 6. 本次分析产物

- `audit/reports/v19/eval/v19_sft_2ep_validation_table.csv`：92 个验证点全量表。
- `audit/reports/v19/eval/v19_sft_2ep_trends.png`：损失/精度/信念趋势图。
- `audit/reports/v19/eval/v19_sft_2ep_final.png`：最终损失贡献/动作精度/信念指标图。
- `audit/reports/v19/scripts/plot_v19_sft_metrics.py`：可复现脚本。
