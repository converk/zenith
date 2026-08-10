# 新 goal 提示词：PPO 优化下一阶段（树搜索 / 信号组合）

> 本文件即新 goal 的提示词。复制全文作为新 goal 的输入；新 goal 的实验结果、日志与提示词
> 统一放在 `audit/reports/ppo_rl_next_goal_20260810/`。

## 0. 背景（前一个 goal 的结论）

前一个 goal（`audit/reports/ppo_rl_goal_run_20260808/`）执行了 E0–E4、两个组合臂、两个补充臂与两个收尾臂，
共 11 个训练实验。结论：

- 原总判据「u200 2v2 >55% 且 paired bootstrap 95% CI 不跨 50%，且启发式 u100 后一位率不再持续退化」**未达成**。
- **E3-b 纯 GRP**（`checkpoints/train_riichi_ppo_goal_e3_grp_reward/checkpoint_00200.pt`）是唯一 u200 显著为正的臂：
  55.83%、分差 +5240、CI [1152, 9579]；但其启发式 u120=34.4%、u195=35.4%，u100 后退化。
- 长训（补充B E3-b→u400、收尾D E4→u400）不能修复退化；dense（0.03/0.05）改善中期启发式稳定性但损伤 u200 2v2。
- EV/value：GRP 系列 value_loss ~0.02–0.03，EV 未见长期为负。

详细证据见 `audit/reports/ppo_rl_goal_run_20260808/EXPERIMENT_RESULTS.md` 与 `PROGRESS.md`。
本 goal 按其中 §6 的设计执行下一步实验，并放宽成功/停止条件。

## 1. 本 goal 的目标与停止条件（放宽版）

**目标**：找到比当前最优臂（E3-b u200）更全面、更稳的策略。实验优先级：

1. **树搜索离线可行性验证**（pMCPA/MCTS 增强 vs E3-b u200，不占训练 GPU，约 2–4h）；
2. 若验证通过：**搜索蒸馏**（200 update 左右）；
3. **E5-a：GRP + SFT-KL 锚定**（纯配置）；
4. **E5-b：两阶段奖励课程**（GRP→dense 或 dense→GRP，resume 式，纯配置）。

**停止/合格条件**（满足其一即视为本 goal 达成；判定以 2v2 240 半庄 + 启发式 96 半庄为准，
训练 reward 不算成功依据）：

- **合格线 A（主）**：任一臂满足 u200 2v2 胜率 **≥52%** 且分差 paired bootstrap 95% CI **不跨 0**，
  且启发式 u100–u200 未出现连续两次评测一位率 <40%；
- **合格线 B（次）**：任一臂相对 E3-b u200 在 2v2 上显著更好（CI 不跨 0），或启发式明显更稳
  （u100–u200 无 <40% 的低点）且 u200 2v2 不低于 50%；
- **默认收尾**：预设计划内的实验全部完成（树搜索验证 + 至多 2–3 个训练臂），即使未达合格线，
  也要输出结论报告（说明尝试了什么、证据、为什么不达标、下一步还能做什么）。

EV 不再长期为负作为全程监控项（GRP 系列已满足，不作为单独停止条件）。

## 2. 硬约束

- 所有 Python / 训练 / 评测 / 冒烟测试必须通过 `conda run -n Mahjong-AI`（或显式激活该环境）执行。
- **GPU 映射**：`CUDA_DEVICE=0`→物理 GPU0、`CUDA_DEVICE=1`→物理 GPU1、
  `CUDA_DEVICE=2`→物理 GPU3、`CUDA_DEVICE=3`→物理 GPU4。**启动训练/评测前必须先 `nvidia-smi`
  确认卡是否空闲**；优先使用空闲卡（此前 CUDA=0/CUDA=2 空闲），同一张卡不要同时跑两个训练；
  卡号对不上时以 nvidia-smi 实测为准。你可以每 5 分钟检查一次是否有空闲的 GPU，
  如果有空闲的，可以直接在那个卡上执行可以并行的实验。
- 冒烟/性能测试：`target_kl=0.0`、`update_epochs=4`、`kyokus_per_worker=1`，默认跑 3 次，
  第 1 次视为热身，报告后两次的性能与耗时；长期实验 `kyokus_per_worker=16`。
- 固定 `seed=1`；不改模型 backbone；**每次只改一个变量**；每个实验独立 config 与 checkpoint 目录。
- **边界**：代码只允许修改 `riichi_ppo_v1/`；新配置放 `riichi_ppo_v1/configs/next_goal/`；
  checkpoint 放 `checkpoints/train_riichi_ppo_next_*`；日志/评测/结果统一放
  `audit/reports/ppo_rl_next_goal_20260810/`（模型文件不放这里）。
- 不删除、不覆盖旧 goal 的 checkpoints / 数据集 / 评测结果；可复用旧 checkpoint
  （E3-b u200、GRP 模型 `checkpoints/train_riichi_ppo_goal_grp/grp_rank_predictor.pt`、E4/C 的 checkpoint）。
- 输出与结论用中文；区分「实证支持」与「推测」；训练 reward 不作为成功依据。

## 3. 执行计划

0. 先完整阅读 `audit/reports/ppo_rl_goal_run_20260808/EXPERIMENT_RESULTS.md`（尤其 §6）与 `PROGRESS.md`；
   `nvidia-smi` 确认可用 GPU 后再动手。
1. **树搜索离线验证**（CPU 为主，不占训练 GPU）：用轻量 pMCPA/MCTS（深度 2–4、宽度 8–16，
   以当前 PPO policy + GRP 终局奖励做 rollout）在关键决策上替换 E3-b u200 的动作，96–240 半庄
   A/B 对比；显著优于基线（CI 不跨 0）才进入搜索蒸馏，否则记录结果并跳到信号组合实验。
2. **E5-a：GRP + SFT-KL 锚定**（纯配置，从 `best_heuristic.pt` 初始化）：`grp_mix_lambda=1.0`，
   `sft_kl_coef_start=0.0`、`sft_kl_coef_end` 取 0.05/0.1 两档消融；200 update；评测 u50/u100/u200。
3. **E5-b：两阶段奖励课程**（resume，纯配置）：
   - 臂1「GRP→dense」：resume E3-b u200，`dense_efficiency_weight=0.05`，再训 150–200 update；
   - 臂2「dense→GRP」：resume E4 u200（或收尾C u200），`dense_efficiency_weight=0.0`，再训 150–200 update；
   - 评测 u200 与续训终点（u300/u350/u400 视可用 checkpoint）。
4. **搜索蒸馏**（仅当第 1 步通过且时间允许）：从 E3-b u200 初始化，用搜索策略分布做 KL/监督目标，
   200 update；评测 u50/u100/u200/u400。

并行规则：GPU 有空闲时最多两个独立实验并行（分别占 CUDA=0 与 CUDA=2）；每臂总耗时尽量 ≤5h
（含评测），以先前实测性能估算（单卡约 80–100s/update + 每 15 update 一次启发式评测约 145s）。

## 4. 评测纪律

- 关键 checkpoint（u50/u100/u200、续训终点）跑 2v2 240 半庄：
  ```bash
  CUDA_DEVICE=<空闲卡> conda run --no-capture-output -n Mahjong-AI python -m riichi_ppo_v1.sft.head_to_head \
    --model-a <ppo_ckpt> --model-b checkpoints/train_riichi_v13_sft/best_heuristic.pt \
    --hanchans 240 --parallel-hanchans 24 --output audit/reports/ppo_rl_next_goal_20260810/<实验>/vs_sft_uXXX.json
  ```
- 读取 `model_a.team_win_rate` 与 `model_a.team_point_diff_paired_bootstrap_ci95` 作为唯一真相。
- 启发式：训练内 `evaluation_enabled=true`、`evaluation_interval_updates=15`（或 30 省时），
  记录 u100–u200 一位率与分差，用于判断「是否连续两次 <40%」。
- 每个实验完成后按合格线 A/B 判定，更新 `PROGRESS.md`；全部完成后写 `EXPERIMENT_RESULTS.md`。

## 5. 交付物

- `PROGRESS.md`：每次实验前后更新（日期、实验、状态、关键结论、下一步）。
- `EXPERIMENT_RESULTS.md`：最终结论（每臂指标表、最优 checkpoint、是否达到合格线、风险、下一步）。
- 各实验子目录：config diff、训练日志、评测 JSON/log。
- 如需 git 提交，使用中文提交消息；提交前确认工作区只包含本 goal 相关改动。
