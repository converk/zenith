# V17 进度记录

## 2026-08-25 Rust Action Query 融合性能验收

- 新增 `riichienv.prepare_v16_compact_facts` 与 `riichi.encode_v16_batch`,保持
  `riichi` 不依赖 `riichienv`;完整 34 种摸牌 improving-mask 扫描与全部 action
  row 均保留,旧 Python batch/逐动作路径保留为 oracle。
- 正确性:全部 action kind 合成回归、真实 env bridge 全字段、golden 15 数组
  69,733 元素及边界 JSON 全等;unit 164 passed、集成 11 passed、Rust 10 passed、
  RiichiEnv 定向 14 passed。
- 标准 512g4e 三轮(第 1 轮预热)后两轮 rollout 71.339/74.204s,均值
  72.772s;相对 129.374s 基线为 1.78×。update/total 均值 151.021/225.044s,
  相对基线为 1.39×/1.51×;epochs 4/4、minibatches 全部执行。
- 真实 2048 配置:exit 0,2,092 games、1,640,549 transitions、2/2 epochs、
  2,140/2,140 minibatches、3,281,098 executed samples;rollout/update/total
  275.312/287.917/563.269s,未启动长期训练。
- snapshot/critic 继续使用现有批路径并保持 oracle 全等;本轮未合入风险更高的
  双缓冲/direct step,因为 action-query 融合已超过 rollout 1.20× 验收目标。

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

## 2026-08-20 update=5

- reward_mean=-1.8368e-10 value_loss=0.32061 entropy=0.48277 actor_grad_norm=0.17414 critic_grad_norm=2.5391 shared_grad_norm=0.26086
- rollout_wall_s=445.41 update_wall_s=506.23 sps=1548.3 grp_calls=1905.6 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2587 top2_rate=0.5018 mean_rank=2.489 point_diff_mean=+395.2 ci95=[-198.01666666666665, 1008.167083333333]

## 2026-08-20 update=10

- reward_mean=-1.7249e-10 value_loss=0.3118 entropy=0.48077 actor_grad_norm=0.18746 critic_grad_norm=1.7493 shared_grad_norm=0.27943
- rollout_wall_s=449.93 update_wall_s=496.93 sps=1530.8 grp_calls=1869 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2717 top2_rate=0.5178 mean_rank=2.463 point_diff_mean=+1056.2 ci95=[430.9558333333333, 1646.9249999999995]

## 2026-08-21 update=15

- reward_mean=-1.7931e-10 value_loss=0.30865 entropy=0.48757 actor_grad_norm=0.19547 critic_grad_norm=1.2048 shared_grad_norm=0.28972
- rollout_wall_s=441.79 update_wall_s=490.21 sps=1544.8 grp_calls=1877.4 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2823 top2_rate=0.5335 mean_rank=2.422 point_diff_mean=+1879.7 ci95=[1250.6466666666668, 2513.3624999999997]

## 2026-08-21 update=20

- reward_mean=-1.7463e-10 value_loss=0.30335 entropy=0.50394 actor_grad_norm=0.20548 critic_grad_norm=1.1364 shared_grad_norm=0.30687
- rollout_wall_s=448.11 update_wall_s=488.79 sps=1529.9 grp_calls=1886.8 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3058 top2_rate=0.5677 mean_rank=2.348 point_diff_mean=+3030.0 ci95=[2430.794583333334, 3677.4083333333333]

## 2026-08-21 update=25

- reward_mean=-1.9614e-10 value_loss=0.30032 entropy=0.50726 actor_grad_norm=0.21244 critic_grad_norm=0.97655 shared_grad_norm=0.31509
- rollout_wall_s=410 update_wall_s=466.72 sps=1557.5 grp_calls=1818.6 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3038 top2_rate=0.5613 mean_rank=2.355 point_diff_mean=+3090.7 ci95=[2489.3758333333335, 3714.8287499999997]

## 2026-08-21 update=30

- reward_mean=-1.9185e-10 value_loss=0.30209 entropy=0.51591 actor_grad_norm=0.21879 critic_grad_norm=1.0598 shared_grad_norm=0.32629
- rollout_wall_s=430.84 update_wall_s=482.91 sps=1508.6 grp_calls=1842.8 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3115 top2_rate=0.5665 mean_rank=2.350 point_diff_mean=+3100.5 ci95=[2499.0491666666667, 3765.2487499999997]

## 2026-08-21 update=35

- reward_mean=-1.5859e-10 value_loss=0.3003 entropy=0.51506 actor_grad_norm=0.22784 critic_grad_norm=1.017 shared_grad_norm=0.33848
- rollout_wall_s=416.99 update_wall_s=461.61 sps=1540.9 grp_calls=1822.5 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3250 top2_rate=0.5645 mean_rank=2.340 point_diff_mean=+3124.1 ci95=[2492.5825, 3736.5591666666664]

## 2026-08-21 update=40

- reward_mean=-1.896e-10 value_loss=0.29685 entropy=0.52196 actor_grad_norm=0.23894 critic_grad_norm=0.91514 shared_grad_norm=0.35429
- rollout_wall_s=415.43 update_wall_s=467.86 sps=1555.5 grp_calls=1850.6 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3177 top2_rate=0.5673 mean_rank=2.335 point_diff_mean=+3430.2 ci95=[2828.8954166666667, 4087.4725]

## 2026-08-21 update=45

- reward_mean=-1.5894e-10 value_loss=0.29407 entropy=0.52756 actor_grad_norm=0.24662 critic_grad_norm=0.90136 shared_grad_norm=0.3666
- rollout_wall_s=418.14 update_wall_s=457.13 sps=1542.8 grp_calls=1812.7 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3290 top2_rate=0.5877 mean_rank=2.297 point_diff_mean=+3951.4 ci95=[3295.4775, 4586.2891666666665]

## 2026-08-21 update=50

- reward_mean=-1.9585e-10 value_loss=0.29768 entropy=0.53217 actor_grad_norm=0.25193 critic_grad_norm=0.91369 shared_grad_norm=0.37348
- rollout_wall_s=403.86 update_wall_s=466.62 sps=1569.3 grp_calls=1838.2 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3207 top2_rate=0.5737 mean_rank=2.326 point_diff_mean=+3412.3 ci95=[2820.1833333333334, 4064.0779166666666]

## 2026-08-21 update=55

- reward_mean=-1.5769e-10 value_loss=0.29477 entropy=0.54146 actor_grad_norm=0.26009 critic_grad_norm=0.83277 shared_grad_norm=0.38617
- rollout_wall_s=414.58 update_wall_s=458.7 sps=1551.9 grp_calls=1831.3 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3315 top2_rate=0.5747 mean_rank=2.303 point_diff_mean=+3748.2 ci95=[3103.7495833333337, 4359.220416666667]

## 2026-08-21 update=60

- reward_mean=-1.597e-10 value_loss=0.29454 entropy=0.53471 actor_grad_norm=0.26158 critic_grad_norm=0.74499 shared_grad_norm=0.38648
- rollout_wall_s=391.63 update_wall_s=453.39 sps=1576.8 grp_calls=1794.4 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3170 top2_rate=0.5715 mean_rank=2.328 point_diff_mean=+3593.7 ci95=[2957.8525000000004, 4170.57125]

## 2026-08-21 update=65

- reward_mean=-1.9287e-10 value_loss=0.29274 entropy=0.5306 actor_grad_norm=0.27513 critic_grad_norm=0.72037 shared_grad_norm=0.40704
- rollout_wall_s=397.61 update_wall_s=452.14 sps=1566 grp_calls=1803.1 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3317 top2_rate=0.5830 mean_rank=2.296 point_diff_mean=+4142.8 ci95=[3491.720833333333, 4839.2804166666665]

## 2026-08-21 update=70

- reward_mean=-1.723e-10 value_loss=0.29495 entropy=0.52591 actor_grad_norm=0.27432 critic_grad_norm=0.79853 shared_grad_norm=0.40688
- rollout_wall_s=417.09 update_wall_s=472.25 sps=1553.8 grp_calls=1866 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3260 top2_rate=0.5710 mean_rank=2.315 point_diff_mean=+3845.6 ci95=[3219.449583333333, 4505.68625]

## 2026-08-21 update=75

- reward_mean=-1.7783e-10 value_loss=0.29103 entropy=0.51698 actor_grad_norm=0.2796 critic_grad_norm=0.72192 shared_grad_norm=0.41651
- rollout_wall_s=420.93 update_wall_s=461.52 sps=1530.1 grp_calls=1832.6 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3217 top2_rate=0.5703 mean_rank=2.312 point_diff_mean=+3956.8 ci95=[3358.5475, 4550.295416666666]

## 2026-08-21 update=80

- reward_mean=-1.5664e-10 value_loss=0.29201 entropy=0.5268 actor_grad_norm=0.28971 critic_grad_norm=0.69581 shared_grad_norm=0.43009
- rollout_wall_s=428.91 update_wall_s=457.83 sps=1512 grp_calls=1809.1 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3215 top2_rate=0.5683 mean_rank=2.315 point_diff_mean=+3980.6 ci95=[3408.7633333333333, 4595.2179166666665]

## 2026-08-21 update=85

- reward_mean=-1.7775e-10 value_loss=0.29267 entropy=0.52109 actor_grad_norm=0.29447 critic_grad_norm=0.67171 shared_grad_norm=0.43485
- rollout_wall_s=403.17 update_wall_s=465.05 sps=1564.3 grp_calls=1822.3 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3222 top2_rate=0.5777 mean_rank=2.307 point_diff_mean=+3953.2 ci95=[3333.442916666667, 4589.3375]

## 2026-08-21 update=90

- reward_mean=-1.5644e-10 value_loss=0.2907 entropy=0.50987 actor_grad_norm=0.30041 critic_grad_norm=0.70413 shared_grad_norm=0.44848
- rollout_wall_s=412.63 update_wall_s=464.78 sps=1552.5 grp_calls=1844.5 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3162 top2_rate=0.5755 mean_rank=2.320 point_diff_mean=+3637.4 ci95=[2992.75625, 4241.887500000001]

## 2026-08-21 update=95

- reward_mean=-1.8443e-10 value_loss=0.28775 entropy=0.50276 actor_grad_norm=0.30553 critic_grad_norm=0.66846 shared_grad_norm=0.45292
- rollout_wall_s=407.54 update_wall_s=457.81 sps=1550.9 grp_calls=1823.8 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3245 top2_rate=0.5697 mean_rank=2.325 point_diff_mean=+3798.0 ci95=[3182.2408333333333, 4438.994166666666]

## 2026-08-21 update=100

- reward_mean=-1.6985e-10 value_loss=0.29177 entropy=0.48978 actor_grad_norm=0.31263 critic_grad_norm=0.63115 shared_grad_norm=0.46369
- rollout_wall_s=413.35 update_wall_s=461.91 sps=1552.5 grp_calls=1846.8 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3252 top2_rate=0.5697 mean_rank=2.322 point_diff_mean=+3799.6 ci95=[3139.669166666666, 4439.8499999999985]

## 2026-08-22 update=105

- reward_mean=-1.8944e-10 value_loss=0.29421 entropy=0.47679 actor_grad_norm=0.32326 critic_grad_norm=0.62542 shared_grad_norm=0.48314
- rollout_wall_s=409.41 update_wall_s=460.19 sps=1560 grp_calls=1862 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3127 top2_rate=0.5630 mean_rank=2.342 point_diff_mean=+3520.4 ci95=[2906.715416666667, 4128.846666666665]

## 2026-08-22 update=110

- reward_mean=-1.6936e-10 value_loss=0.28922 entropy=0.46371 actor_grad_norm=0.324 critic_grad_norm=0.6389 shared_grad_norm=0.48615
- rollout_wall_s=430.84 update_wall_s=488.31 sps=1509.5 grp_calls=1885.8 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3167 top2_rate=0.5717 mean_rank=2.323 point_diff_mean=+3853.8 ci95=[3226.0729166666665, 4464.176249999999]

## 2026-08-22 update=115

- reward_mean=-1.7948e-10 value_loss=0.29048 entropy=0.45592 actor_grad_norm=0.33112 critic_grad_norm=0.60989 shared_grad_norm=0.49149
- rollout_wall_s=406.63 update_wall_s=451.34 sps=1552.4 grp_calls=1815.2 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3177 top2_rate=0.5753 mean_rank=2.311 point_diff_mean=+4110.9 ci95=[3486.08875, 4745.265833333333]

## 2026-08-22 update=120

- reward_mean=-1.9494e-10 value_loss=0.29064 entropy=0.44798 actor_grad_norm=0.3423 critic_grad_norm=0.61085 shared_grad_norm=0.51148
- rollout_wall_s=409.84 update_wall_s=458.24 sps=1557.2 grp_calls=1838.3 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3277 top2_rate=0.5825 mean_rank=2.293 point_diff_mean=+4383.3 ci95=[3767.3091666666664, 5055.199583333334]

## 2026-08-22 update=125

- reward_mean=-2.0522e-10 value_loss=0.2919 entropy=0.4371 actor_grad_norm=0.34103 critic_grad_norm=0.59659 shared_grad_norm=0.51303
- rollout_wall_s=411.16 update_wall_s=461.52 sps=1559.9 grp_calls=1871.8 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3202 top2_rate=0.5703 mean_rank=2.321 point_diff_mean=+4039.4 ci95=[3431.327916666666, 4706.262083333333]

## 2026-08-22 update=130

- reward_mean=-1.8257e-10 value_loss=0.28782 entropy=0.4225 actor_grad_norm=0.35291 critic_grad_norm=0.56358 shared_grad_norm=0.53081
- rollout_wall_s=404.88 update_wall_s=455.65 sps=1562 grp_calls=1832.5 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3182 top2_rate=0.5720 mean_rank=2.318 point_diff_mean=+3964.4 ci95=[3305.9024999999997, 4584.18375]

## 2026-08-22 update=135

- reward_mean=-1.6936e-10 value_loss=0.28832 entropy=0.40978 actor_grad_norm=0.35749 critic_grad_norm=0.55966 shared_grad_norm=0.53795
- rollout_wall_s=404.27 update_wall_s=464.99 sps=1559.5 grp_calls=1870.8 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3125 top2_rate=0.5770 mean_rank=2.314 point_diff_mean=+3746.5 ci95=[3137.1791666666672, 4393.332499999999]

## 2026-08-22 update=140

- reward_mean=-1.7814e-10 value_loss=0.29064 entropy=0.39709 actor_grad_norm=0.36712 critic_grad_norm=0.57108 shared_grad_norm=0.55177
- rollout_wall_s=409.23 update_wall_s=450.76 sps=1547.6 grp_calls=1832.6 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3207 top2_rate=0.5800 mean_rank=2.302 point_diff_mean=+4064.0 ci95=[3461.492916666666, 4687.348333333332]

## 2026-08-22 update=145

- reward_mean=-1.7338e-10 value_loss=0.29197 entropy=0.3946 actor_grad_norm=0.37272 critic_grad_norm=0.55329 shared_grad_norm=0.55964
- rollout_wall_s=421.07 update_wall_s=467.06 sps=1546.3 grp_calls=1859.5 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3145 top2_rate=0.5753 mean_rank=2.320 point_diff_mean=+4020.3 ci95=[3412.4616666666666, 4645.577499999999]

## 2026-08-22 update=150

- reward_mean=-2.0886e-10 value_loss=0.28805 entropy=0.38977 actor_grad_norm=0.37979 critic_grad_norm=0.53991 shared_grad_norm=0.57268
- rollout_wall_s=405.76 update_wall_s=457.36 sps=1557.6 grp_calls=1835.7 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3267 top2_rate=0.5785 mean_rank=2.301 point_diff_mean=+4185.8 ci95=[3544.3104166666667, 4828.5183333333325]
