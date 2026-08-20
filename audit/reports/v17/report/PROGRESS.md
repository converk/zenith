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

## 2026-08-20 update=5

- reward_mean=-1.6115e-10 value_loss=0.34057 q_loss=0.34508 entropy=0.46228 actor_grad_norm=0.16239 critic_grad_norm=4.7515 shared_grad_norm=0.26372
- rollout_wall_s=478.64 update_wall_s=335.21 sps=1850.4 grp_calls=1924.8 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2545 top2_rate=0.5123 mean_rank=2.480 point_diff_mean=+367.9 ci95=[-113.97333333333344, 862.1808333333328]

## 评测失败记录

- update=10：1v3 shard 子进程失败，训练已中止；已尝试路径与证据见下方失败详情。
  - shard 5: returncode=1            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;   File "/mnt/disk1/hubowen/zenith/riichi_ppo_v1/model/architecture.py", line 363, in forward;     x = x + self.down(F.silu(gate) * value);                       ~~~~~~~~~~~~~^~~~~~~; torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 50.00 MiB. GPU 0 has a total capacity of 44.42 GiB of which 24.69 MiB is free. Process 795630 has 33.69 GiB memory in use. Process 796037 has 0 bytes memory in use. Process 1937219 has 1.78 GiB memory in use. Process 1937288 has 860.00 MiB memory in use. Process 1937384 has 2.51 GiB memory in use. Process 1937463 has 2.23 GiB memory in use. Process 1938155 has 862.00 MiB memory in use. Including non-PyTorch memory, this process has 802.00 MiB memory in use. Of the allocated memory 316.39 MiB is allocated by PyTorch, and 137.61 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
  - shard 9: returncode=1            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;   File "/mnt/disk1/hubowen/zenith/riichi_ppo_v1/model/architecture.py", line 363, in forward;     x = x + self.down(F.silu(gate) * value);                       ~~~~~~~~~~~~~^~~~~~~; torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 50.00 MiB. GPU 0 has a total capacity of 44.42 GiB of which 46.69 MiB is free. Process 795631 has 32.56 GiB memory in use. Process 796036 has 0 bytes memory in use. Process 1940142 has 5.57 GiB memory in use. Process 1941907 has 1.12 GiB memory in use. Process 1944310 has 1.32 GiB memory in use. Including non-PyTorch memory, this process has 772.00 MiB memory in use. Process 1950655 has 830.00 MiB memory in use. Process 1953569 has 1.05 GiB memory in use. Of the allocated memory 287.70 MiB is allocated by PyTorch, and 136.30 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## 2026-08-20 update=10 评测补齐与恢复

- 根因:12 个 1v3 分片与两个 learner 共用物理 GPU 0,1(配置
  `eval1v3_devices: ["0","1"]`);每个分片峰值占用约 6.7 GiB(6 分片约 40 GiB),
  叠加上 learner 的 33.7 GiB 后远超单卡 44.4 GiB,shard 5/9 前向 OOM,训练中止。
- 处置:评测改放到训练期间空闲的物理 GPU 3,4(`CUDA_DEVICE=2,3`,
  `eval1v3_devices: ["2","3"]`),与 learner 不争显存;已用
  `riichi_ppo_v1/evaluation/rerun_1v3_eval_update.py` 独立补齐 update=10
  全部 12 分片 × 500 = 6000 半庄(`audit/reports/v17/eval/vs_sft_u010.json`)。
- update=10 1v3 vs SFT:first_place_rate=0.2457 top2_rate=0.5040
  mean_rank=2.499 point_diff_mean=+158.3
  ci95=[-332.9680555555555, 646.3274999999999]
- 恢复:自 checkpoint_00010.pt 续跑 u011..u200,配置为
  `riichi_ppo_v1/configs/v17_ppo_v2run1_resume_u010.yaml`(完整自包含副本,
  仅 resume/init_model/eval1v3_devices 与基线不同);v17_ppo.yaml 的
  eval1v3_devices 同步改为 ["2","3"],避免后续全新训练复现该 OOM。

## 2026-08-20 V17(V1run1 best)vs V16(best PPO)同半庄 1v3 对比

- 目的:判断 V17 是否强于 V16。base/对手 = V16 best PPO
  (`checkpoints/train_riichi_v16/archive_20260818_run4/ppo/checkpoint_00150.pt`,
  run4 例行 1v3 vs SFT 峰值 update=150);V17 候选 =
  `checkpoints/train_riichi_v17/archive_20260819_V1run1/ppo/checkpoint_00060.pt`
  (V1run1 例行 1v3 峰值 update=60)。
- 协议:每候选 4000 个互不相交半庄(10 进程 × 400,CUDA 2/3 各 5 进程,
  seed_base=2026082000),两候选共享完全相同的牌山与候选座位轮转;对手均为
  贪心策略。工具:`riichi_ppo_v1/evaluation/run_1v3_compare.py`;产物:
  `audit/reports/v17/eval/vs_v16_ppo_base_1v3_4000/`。
- 结果(4000 半庄,同一批牌):

  | 候选 | first | top2 | mean_rank | point_diff(95% CI) |
  |---|---|---|---|---|
  | V16 best PPO u150(自打基线) | 0.2545 | 0.5050 | 2.486 | +345.9 [-265.8, 958.7] |
  | V17 V1run1 u060 | 0.2495 | 0.5375 | 2.428 | +1496.7 [908.5, 2096.4] |

- 结论:V17 一位率与 V16 基线基本持平(0.2495 vs 0.2545,噪声内),但 top2
  高 3.3pp、平均名次高 0.058、点差 +1497(95% CI 不含 0,较基线 +346 显著
  高出约 +1150/半庄)。整体判断:V17 比 V16 best PPO 更强,优势主要体现在
  少输分/名次稳定,而非一位率。

## 2026-08-20 update=15

- reward_mean=-1.4704e-10 value_loss=0.31523 q_loss=0.33996 entropy=0.37733 actor_grad_norm=0.17831 critic_grad_norm=2.0963 shared_grad_norm=0.26589
- rollout_wall_s=584.69 update_wall_s=340.29 sps=1654.7 grp_calls=1890.6 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2557 top2_rate=0.5105 mean_rank=2.483 point_diff_mean=+724.4 ci95=[213.67388888888885, 1208.753888888888]

## 2026-08-20 update=20

- reward_mean=-1.9045e-10 value_loss=0.31026 q_loss=0.33785 entropy=0.3444 actor_grad_norm=0.19018 critic_grad_norm=1.5088 shared_grad_norm=0.2831
- rollout_wall_s=492.49 update_wall_s=353.47 sps=1774.9 grp_calls=1843.8 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2605 top2_rate=0.5057 mean_rank=2.483 point_diff_mean=+750.0 ci95=[273.31527777777774, 1254.4327777777776]
