---

description: "Task list for the Mortal-GRP PPO feature"
---

# Tasks: Mortal 式 GRP 纯奖励 PPO

**Input**: Design documents from `/specs/004-mortal-grp-ppo/`

**Prerequisites**: plan.md (required), spec.md, research.md, data-model.md,
contracts/grp-v17.md

**Organisation**: Tasks are grouped by user story to enable independent
implementation and testing of each story. 实现顺序按 Phase 分组(宪法修订先行)。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: US1=GRP Mortal 重写, US2=纯 GRP reward, US3=512 半庄/3072 batch,
  US4=4000 半庄 1v3 评测, US5=TensorBoard 监控
- 路径均为仓库根相对。

---

## Phase 0: 宪法与文档(Done in spec/plan 阶段)

**Purpose**: 宪法原则 IV(4000/5)与 II(v17)已修订(1.5.0),三件套齐全。

- [x] C001 宪法修订 1.4.0 → 1.5.0:原则 IV 4000 半庄/5 updates、原则 II v17 代
- [x] C002 specs/004 全部设计文档落盘(spec/plan/research/data-model/contracts)

---

## Phase 1: US1 - GRP Mortal 方案(Foundation,先于一切)🎯

**Purpose**: 新的 GRP 模型、数据集与训练入口是本方案奖励信号的地基。

- [x] T001 [P] 重写 `riichi_ppo_v1/model/grp.py`:
  - 删除 `GRP_CATEGORIES/GRP_NUMERIC_FEATURES/GRP_EMBED_DIM/GRP_HEAD_HIDDEN`
    与旧 `GRPModel`;新增常量 `GRP_INPUT_SIZE=7`、`GRP_NUM_CLASSES=24`、
    `GRP_HIDDEN=64`、`GRP_LAYERS=2`、`GRP_UTILITY=(1, 1/3, -1/3, -1)`。
  - `GRPModel`: `nn.GRU(7, 64, 2, batch_first=True)` +
    `fc(128→128 ReLU→24)`;`register_buffer("perms"/"perms_t")`(24,4)/(4,24)。
  - `forward(inputs: Tensor packed) -> logits (N, 24)`(仅末层 hidden concat);
  - `calc_matrix(logits) -> (N, 4, 4)`;`get_label(rank_by_player) -> (N,)`;
    `freeze()`。
- [x] T002 [P] 重写 `riichi_ppo_v1/training/grp/prepare.py`:
  - `parse_hanchan` 提取 StartKyoku 的 `[grand_kyoku, honba, kyotaku,
    s/1e4 x4]`(grand_kyoku: E0..3, S4..7; 4p 最多 S4=7)。
  - `prepare_grp_dataset(source, output, ...)` 产出 v17 数据集
    (`datasets/tenhou_grp_2024_2025_v17/`,train/validation,40% 子集,
    chunk npz:features float32 (ΣT,7)、labels uint8 (N,)、offsets)。
  - 保留 `rank_among` tie-break 语义;数据集中写入 `dataset.json`(划分统计)。
- [x] T003 重写 `riichi_ppo_v1/training/grp/train.py`:
  - `batch_size=512`(默认)、AdamW、`lr=1e-5`(默认);validation 每 N steps 或
    每 epoch 计算 `val_loss`,保存 **val-loss 最低** 的 checkpoint
    (`checkpoints/train_riichi_v17/grp/best.pt`),不覆盖更低者。
  - loss:24 类 prefix CE(每个半庄的全部 prefix 监督最终排列标签)。
  - checkpoint 载荷含 `validation_loss`、`model_config`(7/64/2/24)。
- [x] T004 新增 `riichi_ppo_v1/configs/v17_grp.yaml`(自包含:seed、device、
  batch_size=512、learning_rate=1e-5、epochs、checkpoint_dir、val_interval_steps)。
- [x] T005 新增 `riichi_ppo_v1/tests/unit/test_grp_mortal.py`:
  - 7 维输入 forward 输出 (N,24);`calc_matrix` 行和=1;`get_label` 与
    `rank_among` 一致;`freeze` 后不更新;45K±10K 参数预算;prefix 标签
    监督最终排序;backward 通过。

**Checkpoint**: GRP 可离线训练、冻结、单测通过。

---

## Phase 2: US2 - 纯 GRP Reward

**Purpose**: 删除点差分量,reward 完全由 GRP expected utility δ 构成。

- [x] T006 重写 `riichi_ppo_v1/training/grp/reward.py`:
  - `RANK_UTILITY=(1, 1/3, -1/3, -1)`;删除 `SCORE_REWARD_WEIGHT/REWARD_CLIP/
    SCORE_DELTA_CLIP/SCORE_DELTA_SCALE` 与 `combined_reward/
    normalized_score_reward/load_normalization`。
  - `grp_expected_value(rank_logits)`、`grp_delta(rank_logits)`(按 Mortal:
    V_{k+1} − V_k;终局直接算到真实排名)。
  - 接口:`reward = grp_delta(...)`(无 σ 归一化)。
- [x] T007 更新 `riichi_ppo_v1/training/worker.py` 的 `GrpRollout`:
  - 保存每环境 1 条 7 维 prefix 序列(非 4 视角);每小局边界 1 次 GRU 前向
    (末步 expected utility);非终局 δ、终局真实排名 δ。
  - 删除 `sigma_grp/sigma_score/load_normalization` 依赖;`calls` 统计保留。
  - 移除 `boundary_reward` 的 score_delta 分支。
- [x] T008 新增/更新 `riichi_ppo_v1/tests/unit/test_v17_reward.py`:
  - δ = V_{k+1} − V_k;终局真实排名;无点差分量;调用数 = 边界数(新语义:
    每环境每边界 1 次,4 桌 = 4 次)。
- [x] T009 [P] 同步更新 `riichi_ppo_v1/tests/unit/test_grp.py`(旧契约断言改新
  契约)与 `test_v16_reward.py`(若不再适用则删除,先全仓 rg 零引用)。

**Checkpoint**: 纯 GRP reward 单测通过;worker 不再引用 score 归一化。

---

## Phase 3: US3 - 512 半庄/Update 与 Global 3072

**Purpose**: rollout 停止条件为完整半庄数;DDP 每 rank 1536 minibatch,
global effective 3072;显存不足用 gradient accumulation。

- [x] T010 `riichi_ppo_v1/training/worker.py` `collect()`:
  - 停止条件改为完整半庄数 `games_per_update`(默认 512 的一半?不——总 512
    半庄由全部 worker 分摊:target_games = ceil(512 / num_workers) 每 worker);
    统计 `games`(精确完整半庄数)、`kyokus`、`transitions_per_s`。
  - 保持 drain 冻结语义(先收口在途小局/整局再冻结,统计双核对)。
- [x] T011 `riichi_ppo_v1/training/train.py`:
  - 主循环检查 `sum(worker stats["games"]) >= 512`(可允许小幅超额,与半局
    drain 一致);聚合 `rollout/games`、`rollout/kyokus`。
- [x] T012 [P] `riichi_ppo_v1/training/learner.py`:
  - 支持 `gradient_accumulation_steps`(默认 1;配置 >1 时,每步 backward
    累积,`loss/accum_steps` 后 optimizer.step + zero_grad,指标按步记录)。
  - `minibatch_size` 语义不变(每 GPU 1536 → global 3072)。
- [x] T013 `riichi_ppo_v1/training/learner_ddp.py`:分片对齐适配 rank 本地
  minibatch 1536(minibatch_size 直接作为每 rank minibatch);`partition_learner_
  shards` 按 1536 对齐;文档/校验更新。
- [x] T014 新增 `riichi_ppo_v1/tests/unit/test_v17_ppo_config.py`:
  - v17_ppo.yaml 自包含且含 `games_per_update=512`、`minibatch_size=1536`、
    `update_epochs=2`、`total_updates=100`、`critic_bootstrap_updates=2` 等;
  - gradient accumulation 语义单测(4 步累积 = 1 step,loss 均值)。

**Checkpoint**: 512 半庄收集、双卡 1536/GPU 分片、GA 兜底均可用且测试通过。

---

## Phase 4: US4 - 4000 半庄 1v3 中途评测与 Best 选择

**Purpose**: 每 5 updates checkpoint + 1v3 vs V16 SFT 4000 半庄;u005..u100
全部评测;最终选最佳。

- [x] T015 `riichi_ppo_v1/evaluation/mechanism.py`:常量修订
  `DEFAULT_1V3_HANCHANS_PER_PROCESS=400`、`DEFAULT_1V3_INTERVAL_UPDATES=5`
  (宪法原则 IV 已批准);`TOTAL_1V3_HANCHANS=4000`。
- [x] T016 `riichi_ppo_v1/training/train.py`:
  - checkpoint 间隔 = 5(`checkpoint_interval_updates: 5`);`run_1v3_evaluation`
    复用 shards 机制(10 进程 × 400,devices 由配置给)。
  - `eval1v3_hanchans_per_process=400`(shards validate 放行 400/进程)。
- [x] T017 新增 `riichi_ppo_v1/evaluation/select_best_checkpoint.py`:
  - 读取 `audit/reports/v17/eval/vs_sft_u*.json`,按
    `point_diff_vs_mean_opponent_mean` 排序(max),缺失回退 `mean_rank`(min);
    输出 `best_checkpoint.json`(含 path、update、完整指标)。
- [x] T018 新增 `riichi_ppo_v1/tests/unit/test_select_best_checkpoint.py`:
  - 合成 3 个 eval json,断言最高 point_diff 被选中;回退路径。
- [x] T019 [P] 更新 `riichi_ppo_v1/tests/unit/test_head_to_head_1v3_shards.py`
  与 `test_artifact_conventions.py`:新常量 400/5/4000 断言。

**Checkpoint**: 5-update 评测节奏、4000 半庄 shards、best 选择脚本与测试通过。

---

## Phase 5: US5 - PPO 配置与 TensorBoard 监控

**Purpose**: V17 PPO 自包含配置;监控键延续且可判读。

- [x] T020 新增 `riichi_ppo_v1/configs/v17_ppo.yaml`(自包含):
  - 超参:actor_lr 2e-5、shared 5e-6、critic 4e-5;ppo_clip 0.2;target_kl 0.01;
    max_grad_norm 0.5;entropy_start 0.01、entropy_end 0.005;
    critic_bootstrap_updates 2;total_updates 100。
  - Q-Boosting:q_boost_coef 0.05、q_boost_lambda 1.0、q_temperature 1.5、
    top3_q_candidates 3;qboost_lambda(资格迹)保持 0.95?否——按需求
    q_boost_lambda=1.0 就是 qboost_lambda 槽位,配置只保留一个。
  - SFT KL:sft_kl_coef_start 0.002、middle 0.0005、end 0.001。
  - 对手:opponent_mix disabled(current_frac=1.0 即纯 self-play)。
  - 拓扑:num_workers 12、envs_per_worker 32、games_per_update 512、
    minibatch_size 1536、gradient_accumulation_steps 1(可调)。
  - 评测:checkpoint_interval_updates 5、eval1v3_enabled true、
    eval1v3_model_b = V16 SFT best.pt、eval1v3_devices、eval1v3_output_dir
    audit/reports/v17/eval。
  - `init_model` = `checkpoints/train_riichi_v16/sft/best.pt`。
- [x] T021 验证 TensorBoard 键(reuse 现有 `write_curated_scalars` 与
  `metrics.jsonl`):normalized entropy、approx_kl/clipfrac、sft_reference_kl、
  q_loss/q_prediction/q_explained_variance、reward(GRP)、advantage/return、
  grad_norm_*;补 `rollout/games` 与 `rollout/grp_calls`。
- [x] T022 同步文档:`riichi_ppo_v1/README.md`/docs 的训练命令,规格见
  quickstart.md;`AGENTS.md` 若引用旧机制需同步(容器中的「评测机制」段落)。

---

## Phase 6: 集成验证与收尾

**Purpose**: 全量测试、冒烟、清理。

- [x] T023 全仓单测:`pytest riichi_ppo_v1/tests/unit -x -q` 通过(含新测试)。
- [x] T024 冒烟(512 半庄/1 update 微小规模):`train.py --smoke` 变体或短
  `--iterations 1` + 小 `games_per_update`;产出后删除 smoke 产物。
- [x] T025 冒烟结束删除其日志/结果文件(宪法 Quality Gate)。
- [x] T026 [P] 若涉及删除旧 GRP 相关文件(如旧 test 断言),全仓 `rg` 零引用
  检查后再删。
- [x] T027 报告实现结果(修改文件、最终配置、命令、关键实现位置)。

---

## Done 定义(Definition of Done)

- GRP 新契约(model/grp.py + prepare + train + v17_grp.yaml + tests)全部落地。
- reward 纯 GRP(worker + reward.py + tests)无点差残留。
- rollout 半庄数停止 + DDP 1536/GPU + gradient accumulation 兜底。
- mechanism 常量 400/5/4000;train.py 每 5 updates checkpoint+评测;best 选择
  脚本可用。
- v17_ppo.yaml 自包含且满足 FR-011..017;TensorBoard 键延续。
- 全量单测通过 + 冒烟清理完成 + README/docs 同步。