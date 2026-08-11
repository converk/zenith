# PPO 优化实验执行计划（供 Codex /goal 模式执行）

> 本计划依据 `DEEP_RESEARCH_PROMPT.md` 与 `PPO_RL_OPTIMIZATION_REPORT.md` 制定。执行前必须先完整阅读：
> `DEEP_RESEARCH_PROMPT.md`、`PPO_RL_OPTIMIZATION_REPORT.md`、`REFERENCES.md`、`RIICHIENV_ML_GRP_REVIEW.md`、`results/*.json`，并核对相关训练/评测代码。

## 0. 总目标与停止条件

**目标**：让 PPO 从 v13 SFT 初始化后稳定超越 SFT 模型，而不是「打平后在后期退化」。

**总成功判据（唯一真相，训练 reward 不算数）**：

- 对 v13 SFT（`checkpoints/train_riichi_v13_sft/best_heuristic.pt`）2v2 240 半庄胜率 **>55%**，且 paired bootstrap **95% CI 不跨 50%**；
- 对固定启发式对手不再出现 u100 后一位率持续退化（u100–u200 一位率不显著低于 u100 峰值）；
- `ppo/value_explained_variance` 不再长期为负。

**停止条件**：
1. 达到总成功判据；
2. E0–E4 全部完成且未达标（此时输出结论报告）；
3. 被阻塞（见 §12）。

## 1. 硬约束与项目约定

- **不改模型 backbone 结构**；policy/value head 的输入输出用途可以调整。
- 一律从 `checkpoints/train_riichi_v13_sft/best_heuristic.pt` 初始化（`init_model`），固定 `seed=1`。
- **每次只改一个变量**；每个实验使用独立 checkpoint 目录与独立 config overlay，留基线对照。
- 不删除、不覆盖现有 checkpoints、datasets、评测结果；模型文件放 `checkpoints/train_riichi_ppo_goal_*`，代码放 `riichi_ppo_v1/` 下新增模块；**运行日志与相关结果统一放 `audit/reports/ppo_rl_goal_run_20260808/`**（模型 checkpoint 不放这里）。
- **子 agent 使用声明**：如果需要使用子 agent（并行调研、独立实验、代码审查等），必须以「复制全部上下文」的方式启动（把完整背景、本计划、调研报告、本地证据与任务说明一并交给子 agent，例如全上下文 fork）；不允许只发送片段任务让子 agent 在缺失上下文的情况下自行假设。子 agent 的产出（文件、指标、结论）必须回写主线程并由主 agent 审校。
- 使用 Conda 环境 `Mahjong-AI`：**所有 Python / 训练 / 评测 / 冒烟测试命令都必须通过 `conda run -n Mahjong-AI` 执行（或显式激活该环境后运行）**，不允许使用其他 Python 环境。**GPU 优先级：优先单卡 `CUDA_DEVICE=2`（物理 GPU3）、`learner_gpus=1`；只有必须双卡时才用 `CUDA_DEVICE=1,2`（物理 GPU1+GPU3）、`learner_gpus=2`（`CUDA=1` 是补充卡）。** 所有训练、评测、冒烟/性能测试默认按此执行；单卡时耗时估算约为双卡的 1.5–2 倍，以实测为准。
- **冒烟/性能测试**（验证改动跑通时）：`target_kl=0.0`、`update_epochs=4`、`kyokus_per_worker=1`，默认跑 3 次，第 1 次视为热身，报告后两次的 elapsed-time 与全部相关性能指标。**长期实验** `kyokus_per_worker=16`。
- 输出与结论用中文；区分「实证支持」与「推测」；每条建议给出来源或证据。

### 子代理实验方法

如需使用子 agent 执行实验，必须按以下方法进行：

- **使用前提**：只在任务相互独立、且不会争抢同一张 GPU 时并行。示例：GRP 离线训练（CPU）可与 E1 训练并行；E1 与 E2 仅在 GPU 有空闲时并行。**同一张单卡（`CUDA_DEVICE=2`）上不允许同时跑两个训练实验**，否则互相拖慢且可能显存不足。
- **派发方式**：必须以「复制全部上下文」的方式启动：完整背景、本计划、调研报告、本地证据、该实验的完整任务说明与输出路径一并交给子 agent（例如全上下文 fork）；不允许只发送片段任务让子 agent 在缺失上下文的情况下自行假设。
- **职责边界**：每个子 agent 只负责一个实验臂，拥有独立 checkpoint 目录（`checkpoints/train_riichi_ppo_goal_<实验>/`）与结果子目录（`audit/reports/ppo_rl_goal_run_20260808/<实验>/`）；不得修改其他子 agent 负责的文件；`PROGRESS.md` 由主 agent 统一更新。
- **执行规范**：子 agent 必须使用与主 agent 相同的评测命令与判定标准（§10），不得以训练 reward 判定成功；结果必须写回自己的结果子目录（评测 JSON、config diff、指标表）。
- **验收**：子 agent 完成后，主 agent 核对输出字段（`team_win_rate`、`team_point_diff_paired_bootstrap_ci95`、EV 等）与文件落位，再写结论进 `PROGRESS.md` 并决定下一步。
- **失败回退**：若子 agent 无法创建（如线程上限），主 agent 自行完成该实验，不阻塞流程。

## 2. 调研结论 → 实验映射

| 失败原因（按证据排序） | 对应实验 | 主要代码位置 | 参考证据 |
|---|---|---|---|
| critic/value 学习失效（EV 转负） | E1 EV 门控 / 四席组内基线 | `learner.py`、`trajectory.py`、`worker.py` | EVPO (arXiv:2604.19485)、AsyPPO (arXiv:2510.01656) |
| reward 与最终目标错配且稀疏 | E3 GRP、E4 密集效率奖励 | `worker.py`、`training/rewards/` | Suphx (arXiv:2003.13590)、RUDDER (arXiv:1806.07857)、arXiv:2305.04145 |
| 四席 current 自博弈 competitive overfitting | E2 对手混合 80/20 | `worker.py` lineups、`inference.py` | Territory Paint Wars (arXiv:2604.04983)、OpenAI Five |
| SFT KL 累积漂移（0.04→0.226） | 不作为单独实验，全程监控 | `metrics.jsonl` | Catastrophic Goodhart (arXiv:2407.14503) |
| 超参/规模（推测性） | 仅当 E1–E4 无效时做 1 组超参消融 | `training.yaml` | — |

## 3. 执行顺序

1. **E0** 基线复现（必须先做，约 0.5–1 天，含冒烟测试）；
2. **E1**（learner 侧）与 **E2**（worker 侧）代码路径独立，可在 GPU 有空闲时并行，但建议 E1 先出结果；
3. **E3** 必须在 E1/E2 之后做（避免多变量同时改无法归因）；其离线 GRP 训练不占 GPU，可提前启动；
4. **E4** 在「E1/E2/E3 中表现最好的组合」之上叠加并做消融。

每个实验结束后必须按 §9 的评测纪律出指标表，并按该实验的判定标准写结论（成功 / 部分成功 / 失败），更新 `PROGRESS.md`。

**时长口径说明**：各实验标注的「单卡约 0.5–1 天」是**单个 200-update 实验（单臂）**的估算，不是全部实验的总时长。全部实验串行大致为：E0(1 臂) + E1(1 臂) + E2(1 臂) + E3(3 臂) + E4(1 臂) ≈ 7 臂 × 0.5–1 天 ≈ **3.5–7 天**（单卡，以实测为准），另加 GRP 离线训练数小时与各次 2v2/启发式评测开销；E1∥E2 并行、提前达标提前停止、或某实验判定失败直接跳过时，总时长会缩短。

## 4. E0：基线复现

**目标**：确认当前现象可复现，建立对照基线。

**改动点**：无算法改动。先跑通冒烟/性能测试（§1 约定），再用当前 `training.yaml`（长期 `kyokus_per_worker=16`）从 `best_heuristic.pt` 初始化、`seed=1`，跑 **200 update**，输出到 `checkpoints/train_riichi_ppo_goal_e0_baseline/`。

**记录**：
- 每 update：`ppo/value_explained_variance`、`value_prediction_std`、`return_std`、`sft_reference_kl`、`entropy`、`clipfrac`、立直接受率；
- 每 15 update：启发式评测（现有 `evaluation_enabled` 机制）；
- u50/u100/u200：2v2 SFT 240 半庄。

**判定**：能复现「u100 附近平台、对启发式先升后降、EV 转负」即基线有效；复现不出则先解决环境/复现问题，不要开始 E1。

**时长**：1–2 天。

## 5. E1：EV 门控 / 四席组内基线

**假设**：稀疏终局 reward 下 critic 噪声主导，EV<0 时 GAE 优势≈带噪 MC 收益；改用「MC 终局收益 − 同小局组内基线」能恢复有效学习信号。

**改动点**：
1. `Transition` 增加 `kyoku_group`（env_index + kyoku 序号）与 `step_in_kyoku`（该席在小局内的决策序号），在 `worker.py` 构造 transition 时写入；
2. `learner.py` 每 update 计算 batch EV（现成 `ppo/value_explained_variance` 或从 value/return 重算）；
3. 当 EV < 0 时：policy 优势 = 该 transition 的 MC 折扣收益（gamma 衰减，kyoku 内）− 组内基线；组内基线首选「同一小局、同一 `step_in_kyoku` 的四席 leave-one-out 均值」，组内 current 席 <2 或对齐困难时退化为 batch 均值；随后继续走现有 advantage normalize；
4. value loss 保持 GAE return 不变（只替换 policy 优势项）；记录 EV 门控启用频率到 metrics；
5. 通过 config 开关 `adv_estimator: gae|ev_gate` 控制，默认 `ev_gate`，可一键回退。

**配置**：新增 `riichi_ppo_v1/configs/goal/e1_ev_gate.yaml`，其余超参不变。

**期望指标**：EV 回升并维持 ≥0；`policy_loss` 重新下降；`sft_reference_kl` 增速放缓；u200 2v2 明显优于 E0（分差 CI 不跨 0）或直接达标。

**判定**：
- 成功：u200 2v2 >55% 且 CI 不跨 50%；
- 部分成功：EV 转正且 2v2 显著优于 E0（分差 CI 不跨 0）；
- 失败：无明显改善。

**失败处理**：保留开关继续 E2。

**时长**：200 update（单卡约 0.5–1 天，以实测为准）。

## 6. E2：对手混合 80/20

**假设**：四席 current 自博弈导致 competitive overfitting；混入固定对手（冻结 SFT + 少量随机）能恢复泛化。

**改动点**：
1. `worker.py` 的 `set_rollout_context` 改为 config 驱动 lineup：默认 80% envs 全 current；20% envs 把 1 席换成冻结 v13 SFT（可选再拆出 5% 随机席）；
2. 非 current 席推理：`inference.py` 支持第二个 namespace（如 `"sft"`）加载冻结 SFT checkpoint（greedy）；随机席在 worker 内对合法动作均匀采样；
3. 只对 current 席记录 transition（非 current 席 `record=False`）；reward 只发 current 席（现有 lineup 过滤逻辑已存在，需确认覆盖）；
4. 新增 config：`opponent_mix.current_frac=0.8`、`sft_frac=0.2`、`random_frac=0.0`（起步）；先消融 SFT，再加随机。

**期望指标**：启发式一位率不再先升后降；`sft_reference_kl` 更稳；u200 2v2 不差于 E0。

**判定**：
- 成功：2v2 >55% 且 CI 不跨 50%，或至少 2v2 显著优于 E0 且启发式无退化；
- 失败：无改善。

**与 E1 合并注意**：组内基线仅对「同小局 ≥2 个 current 席」启用，否则退化为 batch 均值；E1 与 E2 各自先独立出结果，再合并 best 组合。

**时长**：200 update（单卡约 0.5–1 天，以实测为准）。

## 7. E3：离线 GRP 验证 + PPO A/B

**假设**：小局点差 ≠ 最终排名收益（Suphx 明确反对直接用点差）；GRP 把点差换成「预测最终排名的期望收益」能修复目标错配，降低 critic 学习难度。

### 7.1 离线 GRP 训练与验证（不占 GPU，可与 E1/E2 并行）

1. **数据**：`datasets/tenhou-to-mjai/2024.zip`、`2025.zip`（`.mjson`，用 `zipfile` 流式读取）；先取 10%–20% 子集（约 38 万–77 万小局）。
2. **特征（20 维）**：四家初始分/25000、结束分/25000、本小局分差/12000、场风/3、局数/3、本场/4、立直棒/4、player one-hot。**核对编码**：`chang` = bakaze（东=0 南=1 西=2 北=3）、`ju` = 亲家局数（0 起）。优先复用本地 RiichiEnv 的 `Kyoku.take_grp_features()`；没有则按 riichienv-ml 的 `GrpFeatureEncoder` 复刻。
3. **模型**：MLP `20→128→64→4`，CrossEntropy，`lr=5e-4`、`batch=512`、1 epoch（参考 `smly/RiichiEnv/riichienv-ml`；新代码放 `riichi_ppo_v1/grp/`）。
4. **验证指标**：rank 分类准确率（随机基线 25%，**建议 ≥35% 再进 PPO**）、ECE（建议 ≤0.15）、与 point-delta reward 的 Spearman 相关（**>0.9 则收益有限**，优先改 GRU 或加手牌/牌河特征）。
5. **交付**：GRP checkpoint + `grp_validation.json`（acc/ECE/相关矩阵）。

### 7.2 PPO A/B 三组

- a) point-delta（= E0 基线）；b) GRP reward；c) `λ=0.5` 混合：`r = λ·r_grp + (1−λ)·clip(delta/1000, ±24)`。
- 每组 **200 update**、`seed=1`、独立 checkpoint 目录。
- **worker 集成**：小局结束时（现有写 `terminal_kyoku_reward` 的位置）构造 GRP 特征，按上述公式写该小局最后 transition；GAE/value head 不动；GRP 小模型放 CPU，每小局 4 次 forward，开销可忽略。
- **pts_weight**：默认 `[10,4,-4,-10]`；备选 `[6,4,2,0]`（减均值后等价 `[3,1,-1,-3]`）；softmax 可加温度做 2–3 档扫描。
- 新增 config：`grp_model_path`、`grp_pts_weight`、`grp_mix_lambda`。

**判定**：
- GRP 离线验证不达标（acc 不高或与 point-delta 高度相关）→ 不跑 PPO 组，改为 GRU 版或直接跳到 E4，并记录；
- 成功：混合组 2v2 >55% 且 CI 不跨 50%，且优于 point-delta 组；
- 部分成功：GRP/混合组优于 point-delta 但未达总目标。

**风险**：GRP 训练自人类牌谱，自博弈策略偏离人类分布后外推失真 → 若长期使用需定期用自博弈数据重训；防刷分以 2v2 为唯一真相。

**时长**：GRP 训练数小时；PPO 每组 200 update（单卡约 0.5–1 天，以实测为准）。

## 8. E4：密集效率奖励（小权重）

**假设**：discard 的 shanten/ukeire 效率信号能降低探索方差、给 critic 结构性信号，帮助稳定超越 SFT。

**改动点**：
1. `worker.py` 的 `_model_actions`：discard 决策时 `transition.reward += efficiency_reward(...) × dense_weight`（`dense_weight=0.01–0.05` 起步），非 discard 为 0；终局 terminal reward 仍累加到该小局最后 transition（确认 `finish_kyoku` 按 reward 字段正确累计）；
2. 新增 config `dense_efficiency_weight`；在 best(E1/E2/E3) 组合上叠加并做有/无消融。

**期望指标**：EV 回升；立直接受率回落到 SFT 合理区间（参考 <60%，当前 PPO 72.6%）；放铳率不升。

**判定**：
- 成功：2v2 胜率不下降且行为指标改善；
- 失败：胜率下降 → 立即回滚 dense 权重。

**注意**：效率奖励不是 potential-based shaping（Ng et al. 1999），理论上可能改变最优策略，权重必须小；suji/危险度防守惩罚留到第二阶段。

**时长**：200 update（单卡约 0.5–1 天，以实测为准）。

## 9. 监控指标与评测纪律

**训练监控（每 update）**：`ppo/value_explained_variance`（EV<0 报警）、`value_prediction_std / return_std`、`sft_reference_kl`、`entropy`、`clipfrac`、立直接受率、放铳/和牌/副露率（semantic metrics）。

**固定节拍评测**：
- 每 15 update：固定启发式 96 半庄（现有 `evaluation_enabled: true`）；
- u50/u100/u200/最终：2v2 SFT 240 半庄，命令：

```bash
CUDA_DEVICE=2 conda run -n Mahjong-AI python -m riichi_ppo_v1.sft.head_to_head \
  --model-a <ppo_checkpoint> \
  --model-b checkpoints/train_riichi_v13_sft/best_heuristic.pt \
  --hanchans 240 --parallel-hanchans 24 \
  --output audit/reports/ppo_rl_goal_run_20260808/vs_sft_uXXX.json
```

读取输出 JSON 中的 `model_a.team_win_rate` 与 `model_a.team_point_diff_paired_bootstrap_ci95`。
双卡时把首行改为 `CUDA_DEVICE=1,2`（并按需调整 `--parallel-hanchans`，单卡建议 ≤8–12）。

**纪律**：
- 以外部评测为唯一真相；训练 reward 上升但 2v2 不升 = reward hacking，停止该组并记录；
- 每次实验固定 seed 1、同一评测脚本；config diff 与结果 JSON 存入 `audit/reports/ppo_rl_goal_run_20260808/<实验名>/` 子目录；
- 冒烟/性能测试按 §1 约定执行（3 次、第 1 次热身、报告后两次）。

## 10. 进度与最终交付

- `audit/reports/ppo_rl_goal_run_20260808/PROGRESS.md`：每次实验前后更新（checkpoint、改动、指标表、结论、阻塞）。
- 最终 `audit/reports/ppo_rl_goal_run_20260808/EXPERIMENT_RESULTS.md`：逐实验结论（假设/改动/指标/判定/证据等级）、最优 checkpoint 路径、2v2 结果对比表、剩余风险与下一步。
- 每个实验的评测 JSON、config diff、指标汇总放入 `audit/reports/ppo_rl_goal_run_20260808/<实验名>/` 子目录（e0_baseline/、e1_ev_gate/、e2_opponent_mix/、e3_grp/、e4_dense_reward/）。
- 报告结构：总览 → 每个实验（假设、改动、指标表、判定、结论）→ 组合最优结果 → 风险与建议。

## 11. 受阻停止条件

遇到以下情况停止并报告（不要硬编）：
- GPU/环境不可用（单卡 `CUDA_DEVICE=2` 不可见、双卡 `CUDA_DEVICE=1,2` 不可用、Ray 起不来、Conda 环境缺失）；
- 评测工具跑不通（head_to_head 报错、输出缺 CI 字段）；
- GRP 数据无法读取（zip 损坏、`Kyoku.take_grp_features()` 不存在）；
- 基线无法复现（E0 现象与调研不符）；
- 同一实验连续 3 次针对性调整仍无进展。

报告内容：已尝试路径、证据、阻塞原因、下一步需要的输入。

## 12. 关键参考资料

- Suphx（GRP）：arXiv:2003.13590
- EVPO：arXiv:2604.19485；AsyPPO：arXiv:2510.01656
- Territory Paint Wars：arXiv:2604.04983；OpenAI Five（80/20 对手池）
- DouZero：arXiv:2106.06135；ACH：ICLR 2022
- RUDDER：arXiv:1806.07857；Catastrophic Goodhart：arXiv:2407.14503
- 单机麻将 reward shaping：arXiv:2305.04145
- riichienv-ml GRP：https://github.com/smly/RiichiEnv/tree/main/riichienv-ml（issue #75）
- Mortal（序列 GRP 实践）：https://github.com/Equim-chan/Mortal（讨论 #82、`mortal/reward_calculator.py`）
- 本地证据：`REFERENCES.md`、`RIICHIENV_ML_GRP_REVIEW.md`、`checkpoints/train_riichi_ppo/{metrics,evaluation}.jsonl`
