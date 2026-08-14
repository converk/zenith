# V14 PPO 进度日志

## 2026-08-12 实现完成（待长训）

- 代码改动全部完成并通过 199 个 riichi_ppo_v1 单测、219 个 RiichiEnv env 测试、
  25 个 riichi_lab_bot 测试（`python -m pytest`）：
  - `RiichiEnv`：`BatchedRiichiEnv.walls()`（每桌剩余墙牌，按摸牌顺序返回）。
  - critic：`SEGMENT_CRITIC_FUTURE_WALL=5` 未来 5 张 live wall 有序 token，
    仅进入 `critic_factors`；bridge/worker 接线并每次 step/reset 刷新。
  - semantic validation：critic 放行未来牌行并校验位置/种类/red5；actor 仍拒绝 segment 5。
  - 对手分布：70/20/10 current/SFT/historical，历史池 u60+ 且滞后 ≥60；
    池空回退 70/30；inference 按 `history:uNNN` 懒加载冻结 checkpoint。
  - checkpoint 原子保存（tmp + `os.replace`）；`v14_ppo.yaml`；1v3 分片同步评测
    （10 进程 × 160 半庄，CUDA_DEVICE=0/2），汇总 JSON + bootstrap CI。
- 重要实测修正：原生 `wall.tiles` 的下一张正常摸牌在**尾部**（`_deal_next` 用
  `pop()`，实测 step 后 `walls()[0][0]` 若不反转不会前移）。因此 `walls()` 按
  摸牌顺序返回（前 69 张 live、后 14 张 dead），`tiles[:5]` 即接下来 5 张。
- 决策记录：长训 `kyokus_per_worker=16`（AGENTS.md/§2 硬约束；§3.6 YAML 中的 1
  按冒烟覆盖处理），冒烟用 `--kyokus 1`。

### 冒烟性能（CUDA_DEVICE=0，v14_ppo.yaml，kyokus=1，3 次取第 2–3 次）

| 指标 | run2 | run3 | e0 基线 |
| --- | ---: | ---: | ---: |
| sps | 45.87 | 41.72 | 45.29 |
| rollout_wall_s | 1.752 | 1.566 | 1.538 |
| update_wall_s | 0.337 | 0.372 | 0.336 |
| executed padding fraction | 0.164 | 0.164 | - |

结论：新增 5 个 critic token 无明显 >10% 系统性劣化（单局噪声范围内），可启动长训。
评测 CLI 探测：2 半庄 `point_diff_samples` 正常输出。

## 2026-08-13 异常记录：u120 历史池崩溃与修复

- 现象：update 120 首次启用历史池（`history:u060`）时，RolloutInferenceActor
  抛 `malformed history namespace 'history:u060'`，训练中止（u120 checkpoint 未保存）。
- 根因：`_history_model` 解析 namespace 时只去掉 `history:` 前缀，而 `history_namespace`
  按提示词生成带 `u` 的标签（`history:u060`），`"u060".isdigit()` 为 False。
- 修复：`parse_history_namespace()` 同时去掉可选 `u` 前缀；新增单测
  （`history:u060`/`history:u780` 解析正确、非法格式拒绝），12 个相关测试通过。
- 恢复：从 u90 checkpoint 精确 resume（`v14_ppo_resume.yaml`，
  `--training-config v14_ppo.yaml --config v14_ppo_resume.yaml`），
  于 01:14 重启，iteration=91 起继续，u91–u119 重算。

## 2026-08-13 SFT 全席基准（u000）

- 用户确认需要 SFT vs SFT 的候选席基准：四席全为 best_heuristic.pt，
  与每个 V14 checkpoint 完全相同的 1600 半庄协议（10 进程 × 160，
  seed_base=20260812+shard，候选席 i%4 轮转，greedy）。
- 为避免与占满双卡的长训抢资源（§7 要求评测期间不同时跑重任务），
  已创建 `run_sft_baseline.sh` watcher：训练进程退出后自动执行，
  输出 `eval/vs_sft_u000.json`（update=0）与 10 个 shard 原文。
- 应继续执行：经用户确认，先暂停训练（最近完整 checkpoint u600），
  立即运行基准后恢复（resume u600 → iteration 601）。基准结果：
  first_place_rate=0.2537、top2=0.5025、fourth=0.2288、mean_rank=2.501、
  point_diff=-7.5（CI [-1011.1, +996.7]）、win=0.2369、deal_in=0.1731、
  tsumo_loss=0.2025、riichi=0.0127、draw_tenpai=0.2911。
- 校验：u000 基准与 u30 评测逐项完全一致——u30 处于 critic bootstrap、
  actor 为 SFT 冻结权重，同种子/同席位协议下应与四 SFT 基准逐位相同，
  证明基准协议与曲线口径一致。

## 2026-08-12

- 状态：goal 提示词已生成（GOAL_PROMPT.md），尚未开始实现与训练。
- 已确认事实：`checkpoints/train_riichi_v13_sft/best_heuristic.pt` 包含完整 critic 权重
  （51 个 model key，含 critic_embedding/critic_backbone/value_head），可直接 strict load 后外挂训练；
  `BatchedRiichiEnv` 目前不暴露 wall，需在 `riichienv-python/src/env.rs` 增加 `walls()`；
  单环境实测 `wall` 初始 83 张（69 live + 14 dead），`wall[0:5]` 即接下来 5 张 live wall 牌。
- 决策（用户已确认）：总迭代 800；checkpoint 与 1v3 评测每 30 update；评测同步阻塞；
  测试/训练/评测统一 CUDA_DEVICE=0,2（物理 GPU0+GPU3）；历史池 = u60 起、滞后 ≥60 的
  30 间隔 checkpoint，池空退化为 70/30 current/SFT。
- 下一步：启动 goal 后先实现 RiichiEnv `walls()` 并重装扩展，再实现 critic 编码与语义校验、
  对手分布/历史池、1v3 shard 评测，全部单测 + smoke 通过后启动长训。

## 2026-08-13 update=60

- reward_mean=-0.0026399 value_loss=n/a sft_reference_kl=n/a actor_grad_norm=n/a critic_grad_norm=n/a shared_grad_norm=n/a
- rollout_wall_s=18.439 update_wall_s=27.472 sps=712.96 history_seats=0 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2725 top2_rate=0.4919 mean_rank=2.481 point_diff_mean=+473.3 ci95=[-622.6312500000001, 1510.192708333333]

## 2026-08-13 update=120

- reward_mean=-0.011738 value_loss=0.15027 sft_reference_kl=0.093926 actor_grad_norm=0.39674 critic_grad_norm=0.7494 shared_grad_norm=0.9425
- rollout_wall_s=19.718 update_wall_s=27.409 sps=701.67 history_seats=4.3333 history_pool_size=1
- 1v3 vs SFT: first_place_rate=0.3531 top2_rate=0.5663 mean_rank=2.309 point_diff_mean=+4914.3 ci95=[3814.519791666667, 6006.556249999998]

## 2026-08-13 update=180

- reward_mean=-0.008231 value_loss=0.13765 sft_reference_kl=0.13396 actor_grad_norm=0.45036 critic_grad_norm=0.75892 shared_grad_norm=1.0791
- rollout_wall_s=19.912 update_wall_s=27.424 sps=698.82 history_seats=3.25 history_pool_size=3
- 1v3 vs SFT: first_place_rate=0.2838 top2_rate=0.5119 mean_rank=2.462 point_diff_mean=+688.5 ci95=[-392.4614583333334, 1688.5124999999998]

## 2026-08-13 update=240

- reward_mean=0.0040086 value_loss=0.14335 sft_reference_kl=0.16767 actor_grad_norm=0.47677 critic_grad_norm=0.84129 shared_grad_norm=1.142
- rollout_wall_s=19.677 update_wall_s=26.2 sps=700.15 history_seats=2.3333 history_pool_size=5
- 1v3 vs SFT: first_place_rate=0.2938 top2_rate=0.5419 mean_rank=2.389 point_diff_mean=+3552.6 ci95=[2472.7760416666674, 4609.078124999999]

## 2026-08-13 update=300

- reward_mean=0.01299 value_loss=0.14469 sft_reference_kl=0.17062 actor_grad_norm=0.48321 critic_grad_norm=0.91191 shared_grad_norm=1.1403
- rollout_wall_s=19.356 update_wall_s=25.943 sps=703.87 history_seats=3.3333 history_pool_size=7
- 1v3 vs SFT: first_place_rate=0.2681 top2_rate=0.4719 mean_rank=2.538 point_diff_mean=-101.6 ci95=[-1150.2333333333336, 982.4687500000001]

## 2026-08-13 update=360

- reward_mean=5.7817e-05 value_loss=0.1402 sft_reference_kl=0.19272 actor_grad_norm=0.54297 critic_grad_norm=1.1575 shared_grad_norm=1.3134
- rollout_wall_s=22.495 update_wall_s=26.367 sps=651.58 history_seats=3.5 history_pool_size=9
- 1v3 vs SFT: first_place_rate=0.3056 top2_rate=0.5369 mean_rank=2.376 point_diff_mean=+2753.4 ci95=[1647.39375, 3807.4187500000003]

## 2026-08-13 update=420

- reward_mean=0.0060181 value_loss=0.16257 sft_reference_kl=0.20048 actor_grad_norm=0.51968 critic_grad_norm=1.5238 shared_grad_norm=1.2866
- rollout_wall_s=19.09 update_wall_s=26.288 sps=699.98 history_seats=2.9167 history_pool_size=11
- 1v3 vs SFT: first_place_rate=0.2656 top2_rate=0.4819 mean_rank=2.466 point_diff_mean=+1545.2 ci95=[538.169791666667, 2568.3802083333335]

## 2026-08-13 update=480

- reward_mean=-0.0020354 value_loss=0.16969 sft_reference_kl=0.22696 actor_grad_norm=0.56729 critic_grad_norm=1.6206 shared_grad_norm=1.3762
- rollout_wall_s=18.706 update_wall_s=26.751 sps=712.7 history_seats=2.8333 history_pool_size=13
- 1v3 vs SFT: first_place_rate=0.2769 top2_rate=0.4900 mean_rank=2.504 point_diff_mean=+430.5 ci95=[-600.94375, 1468.7031249999995]

## 2026-08-13 update=540

- reward_mean=-0.00082871 value_loss=0.18787 sft_reference_kl=0.23227 actor_grad_norm=0.59848 critic_grad_norm=1.7804 shared_grad_norm=1.4841
- rollout_wall_s=20.684 update_wall_s=25.123 sps=673.04 history_seats=3.5 history_pool_size=15
- 1v3 vs SFT: first_place_rate=0.2662 top2_rate=0.5394 mean_rank=2.413 point_diff_mean=+2833.1 ci95=[1807.8520833333334, 3945.310416666666]

## 2026-08-13 update=600

- reward_mean=-0.0019009 value_loss=0.18451 sft_reference_kl=0.23927 actor_grad_norm=0.61767 critic_grad_norm=1.7405 shared_grad_norm=1.4998
- rollout_wall_s=20.175 update_wall_s=25.669 sps=694.72 history_seats=3.4167 history_pool_size=17
- 1v3 vs SFT: first_place_rate=0.2794 top2_rate=0.4988 mean_rank=2.434 point_diff_mean=+2247.8 ci95=[1153.3666666666668, 3350.4197916666653]

## 2026-08-13 update=660

- reward_mean=-0.0052496 value_loss=0.20757 sft_reference_kl=0.25282 actor_grad_norm=0.62172 critic_grad_norm=1.6955 shared_grad_norm=1.5388
- rollout_wall_s=18.675 update_wall_s=25.375 sps=705.7 history_seats=2.9167 history_pool_size=19
- 1v3 vs SFT: first_place_rate=0.3106 top2_rate=0.5637 mean_rank=2.337 point_diff_mean=+5113.1 ci95=[4076.1072916666667, 6203.765625]

## 2026-08-13 update=720

- reward_mean=-0.0071365 value_loss=0.24484 sft_reference_kl=0.25519 actor_grad_norm=0.69916 critic_grad_norm=1.7197 shared_grad_norm=1.749
- rollout_wall_s=19.506 update_wall_s=25.12 sps=707.06 history_seats=3.25 history_pool_size=21
- 1v3 vs SFT: first_place_rate=0.3106 top2_rate=0.5381 mean_rank=2.399 point_diff_mean=+3377.8 ci95=[2280.498958333333, 4411.528125]

## 2026-08-13 update=780

- reward_mean=-0.0045367 value_loss=0.29059 sft_reference_kl=0.25979 actor_grad_norm=0.71829 critic_grad_norm=1.2363 shared_grad_norm=1.7504
- rollout_wall_s=21.525 update_wall_s=25.956 sps=666.39 history_seats=3.25 history_pool_size=23
- 1v3 vs SFT: first_place_rate=0.2831 top2_rate=0.5475 mean_rank=2.401 point_diff_mean=+3014.1 ci95=[1978.5822916666664, 3991.0906250000007]

## 2026-08-13 训练完成（800 updates）

- 800 updates 全部完成：26 个 checkpoint（u30–u780）+ latest.pt 已保存，
  27 份 1600 半庄评测（u000 SFT 全席基准 + u030–u780）全部落盘，
  train.log / eval1v3.jsonl / metrics.jsonl / performance.jsonl 齐全。
- 历史池全程正常：u120 起启用，最终 history_pool_size=23（u60–u750），
  后期每 worker 约 3.25 个历史席。
- 训练期间两次暂停/恢复均从原子 checkpoint 精确恢复（u90→u91、
  u600→u601，第二次为插入 SFT 全席基准）。
- 最优 checkpoint 结论见 EXPERIMENT_RESULTS.md：
  **u120**（分差 +4914，CI [+3815, +6007]，一位率 0.3531），
  u660/u510 为强候选。
- 下一步：u120/u660/u510 多邻域与多样化对手验证，见 EXPERIMENT_RESULTS.md §5。

## 2026-08-13 Holdout 2000 半庄验证（seed_base=20260822）

- 按用户要求，用与原始 1600 半庄完全不同的 2000 个半庄（种子 20260822..20260831，
  与原 20260812..20260821 无重叠）重跑 u120/u660/u510 的 1v3 vs SFT，
  并补跑四席 SFT 基准；每实验 10 进程 × 200 半庄，两两并发（同时 20 进程），
  耗时约 16.5 分钟。结果：

| 实验 | 一位率 | 前二率 | 均顺 | 平均分差 | CI | 和牌率 | 放铳率 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| u000 SFT 基准 | 0.2430 | 0.4950 | 2.511 | -52.0 | [-960, +819] | 0.2385 | 0.1580 |
| u120 | 0.3155 | 0.5460 | 2.380 | +3070 | [+2165, +4045] | 0.2615 | 0.1300 |
| u510 | 0.3100 | **0.5740** | **2.330** | **+3777** | [+2914, +4645] | **0.2850** | **0.1070** |
| u660 | 0.2705 | 0.5260 | 2.442 | +2915 | [+1990, +3852] | 0.2675 | 0.1225 |

- 结论：三个候选在全新 2000 半庄上均显著优于 SFT 基准（CI 下界全部高于
  基准 CI 上界）；u510 在 holdout 上最稳（分差/前二/放铳最优），
  u120 一位率最高，u660 相对原曲线回落最大。详细对比与最终推荐见
  EXPERIMENT_RESULTS.md。
