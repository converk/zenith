# V17 进度记录

## 2026-08-19 V1run1 归档(update=100 全量完成)

- 本次双卡 PPO 训练(2026-08-19 02:39 → 16:58,100 iterations,含 u060 起
  resume 续跑)产物已整体归档到 `archive_20260819_V1run1`:
  - checkpoint/metrics/performance/tensorboard:
    `checkpoints/train_riichi_v17/archive_20260819_V1run1/ppo/`(20 个定期
    checkpoint + latest.pt)
  - 训练日志:`logs/v17/archive_20260819_V1run1/ppo_train.log`
  - 1v3 例行评测(20 个更新点 u005..u100,vs V16 SFT)+ 中间 shards:
    `checkpoints/train_riichi_v17/archive_20260819_V1run1/eval/`
- 1v3 vs SFT 全程轨迹(u005→u100):first_place_rate 0.2568→0.3013,峰值
  update=60(0.3035,top2 0.5555);中后段(u065→u100)在 0.2848–0.3013 区间;
  mean_rank 2.48→2.37,point_diff_mean +452.6→+2990.4。
- 训练侧:entropy 0.480→0.106(退火中),value_loss 0.400→0.314,
  1439→1383 sps;全局累计 38.45M 决策 / 58.8 万半庄。
- GRP checkpoint 与 GRP 训练日志保留原位
  (`checkpoints/train_riichi_v17/grp/`、`logs/v17/grp_train.log`),供后续
  冻结复用。

## 2026-08-19 梯度 SNR 诊断

- 完成 7 个 checkpoint(u010/u030/u050/u060/u070/u090/u100,含 1v3 最佳
  u060)× 8 seed = 56 次独立 512 半庄 self-play rollout 的 policy 梯度诊断;
- 全程只读 checkpoint + 采样 + backward,不执行 `optimizer.step()`;梯度仅含
  PPO Actor policy loss(共享主干 + Actor 分支);
- 结果:SNR 0.39–0.56,先降(u010→u050)后升(u060→u090 平台)末尾回落
  (u100);u060 为 1v3 胜率峰值且是 SNR 回升转折点;u100 出现个体梯度继续
  增大、平均梯度停滞、SNR 回落的早期信号弱化迹象;
- 报告:`audit/reports/v17/report/gradient_snr_report.md`;指标与图:
  `audit/reports/v17/report/gradient_snr/`;原始逐 seed 梯度:
  `logs/v17/gradient_snr/`。
