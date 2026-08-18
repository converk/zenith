# V16 进度记录

**特性**: specs/003-v16-model-rework(V16 模型重构与训练)
**分支**: `V16` | **基线 HEAD**: `4d4bb2a428c7c80322684bd0f68f0b9162f0c5a9`
**开始时间**: 2026-08-16

## 阶段进度

- [x] Phase 1 Setup:产物目录骨架、变更前基线、分支快照
- [x] Phase 2 Foundational:宪法修订 + 协议 v16 常量
- [x] Phase 3 US1:输入契约与语义落地(硬门槛通过;bridge 装配待 US2 联网)
- [x] Phase 4 US2 代码:网络 + SFT 重编码流水线/训练入口(运行任务 T024-T026 未执行)
- [x] Phase 5 US3 代码:GRP 模型/数据集/训练(运行任务 T032 未执行)
- [ ] Phase 6 US4:PPO 集成(算法契约与配置已落地;Ray rollout/learner 全量接线待续)
- [ ] Phase 7 US5:治理闭环(协议文档已同步;删除类任务依赖 v16 全量接线)
- [ ] Phase 8 Polish:全链复跑与一致性收口

## 基线记录

- 变更前测试基线:pytest 243 passed(0 failed);cargo test --workspace 121 passed
  (111 core + 4 agari + 6 riichi state-machine);cargo 需
  `LD_LIBRARY_PATH=<conda env>/lib` 加载 libpython3.12
- 协议版本:新信息编码协议 v16(单一版本)

## 已完成工作记录

- 2026-08-16:宪法 v1.3.0→v1.4.0(Principle II:信息编码协议 v16,活跃实验代 v16,
  Sync Impact Report 已记录)
- 2026-08-16:新增 `model/encoding_protocol.py`(协议 v16 单一来源、20 slot 语义/
  基数/bucket)+ `tests/unit/test_encoding_protocol.py`(通过);`model/schema.py`
  版本常量改为引用协议常量
- 2026-08-16:Rust 内核完成并通过单测:`riichienv-core/offense_analysis.rs`
  (analyze_offense_v16,O0–O6/O8)与 `riichienv-state-machine/analysis.rs`
  (analyze_defense_v16,D0–D9);两个扩展已 maturin develop --release 重装
- 2026-08-16:产物约定测试扩到 v16;`audit/reports/v15/` 两个历史 eval 归档目录
  已按规范归档移动到 `v15/eval/`(只移动未删除)
- 2026-08-16:US1 语义落地:core `offense_analysis.rs` 拆为役/番数内核(O4/O5,
  `YakuAnalysisV16`),state-machine 新增 `analyze_offense_v16`(O0–O3/O6/O8)与
  `analyze_defense_v16`(D0–D9)+ `public_opponent_summary`;Python 侧新增
  `model/snapshot.py` 与 `model/action_query.py`;语义硬门槛
  (test_v16_query_semantics.py 8 项 + test_v16_replay_bridge.py 3 项)全部通过,
  覆盖门清/副露打牌、自摸/荣和终局、pass/吃牌、振听、N/A 与基数边界
- 2026-08-16:测试基线更新:pytest 258 通过;Rust workspace 124 通过

## 本次会话新增(2026-08-16 续)

- T017-T019:V16 网络 preset(d_model=256/Q=16/KV=4/head_dim=16/FFN=1088/
  shared=4+actor=1+critic=2),Offense/Defense 对称融合(concat 512→256→SiLU→
  Policy MLP),无 zero-init、无 241 维 Q head;实测总参数 7,680,002(加 Top-3 Q
  scorer 后 7,811,587)、Actor 推理 5,525,761,均在设计容差内。
- T015:bridge `prepare_v16` 装配 Objective Facts+Snapshot+每动作一对 Query;
  Critic 特权输入只保留三家对手手牌+后 5 牌山。
- T020-T023:SFT V16 编码器(`encode_kyoku_v16`/`precompute_v16`)、单协议
  manifest(`format=riichi-sft-encoded-v16` + 单一 `encoding_protocol_version=16`
  + 契约 sha256)、`train_v16.py` 从零训练入口与 `configs/v16_sft.yaml`。
- T027-T031:GRP 模型(50-70K)、4 视角旋转数据集构造与 σ_Score 固化、
  `configs/v16_grp.yaml`、离线训练入口(train.py 训练后写回 σ_GRP 并冻结)。
- T033-T037/T040:Top-3 Q scorer([z_critic; detach(h_a)]→512→256→SiLU→1)、
  候选集(Top-3∪行为动作≤4)、GRP+分差奖励(70/30、σ 固化、终局真实排名
  utility)、移除独立半庄排名分量(16/8/-8/-16)、`configs/v16_ppo.yaml`。
- T042:新增 `riichi_ppo_v1/docs/v16_input_protocol.md` 并同步
  `KyokuEventTupleProtocol.md` 的现行契约段落。
- 测试基线更新:pytest 286 通过;Rust workspace 124 通过。

## 运行任务结果(2026-08-16 续,仅 GPU 0)

- T024 canary 编码通过:15,505 局(train 15,462 + validation 43),94.9 万决策;
  query answer 越界 0、snapshot 数值越界 0、合法/专家动作组全覆盖,manifest
  契约正确;验证后 canary 目录已删除。
- T029/T032 GRP:按 game_id 聚合完整半庄(修复逐局误编码),train 143,802 半庄
  (575,208 视角样本)、validation 1,461 半庄;σ_Score=4.2656 固化;GPU 0 训练
  30 epochs(269,629 steps),验证集排名准确率 **93.86%**(均匀随机 25%),σ_GRP=
  2.7112 写回 dataset.json,模型冻结并保存
  `checkpoints/train_riichi_v16/grp/best.pt`。
- T025 40% 全量编码进行中:目标 1,538,630 局(1,523,056 train + 15,574
  validation),16 进程;**已完成**:train 93,943,903 决策、validation 959,045
  决策,query answer/snapshot 数值越界均为 0,manifest 契约与动作覆盖率正确。
- v16 SFT 训练循环已在 GPU 0 小数据集冒烟通过(12 steps,checkpoint/metrics/
  tensorboard 落盘),修复了 v16 配置加载与输出变量遮蔽两处缺陷。
- T026 V16 SFT 训练(GPU 0,单卡,60,000 steps ≈ 3,070 万决策):训练 top1/top3
  80.24%/97.60%,**验证集 Recall@3 = 97.55%(top1 80.09%)**;吞吐约 4,620 样本/
  秒。98% 门槛尚未达到(距 0.45 个百分点),checkpoint 保存于
  `checkpoints/train_riichi_v16/sft/{best,latest}.pt`(可直接载入 v16 模型,
  0 missing/0 unexpected),续训即可逼近门槛。
- 编码期修复:杠/副露手牌形状按「3×副露数 + 杠 4-copy」归一化、离线回放同类
  牌物理 id 归一化、reach/dahai 摸牌回退(修复真实数据 5 类崩溃);GRP 按
  game_id 聚合完整半庄(修复逐局误编码)。

## 2026-08-17 V16-small 与 60% 数据修订

- 版本命名保持 V16,输入/输出协议不变;隐藏层调整为 V16-small:
  `d_model=192`、Q/KV=12/3、head_dim=16、FFN=576、3 Shared + 1 Actor +
  2 Critic;实测总参数 **3,081,603**(含 Top-3 Q scorer)、Actor 推理
  **2,043,073**。旧大模型参数不兼容,已按只归档移动原则迁移。
- SFT 数据扩到 60%:复用 `datasets/tenhou_sft_2024_2025_encoded_40pct_v16`
  (remainders 0,1),`precompute_v16 --reuse-encoded` 只追加 remainder=2 的
  20% 新编码,输出 `datasets/tenhou_sft_2024_2025_encoded_60pct_v16`;
  manifest 记录 `reused_encoded_cache`/`reused_counts`。
- GRP 模型、奖励契约、1v3/SFT 节奏键均不修改。
- 归档:旧 V16 大模型 SFT/PPO checkpoint、日志与 1v3 评测结果已移动到
  `checkpoints/train_riichi_v16/archive_20260817/`、
  `logs/v16/archive_20260817/` 与 `audit/reports/v16/eval/archive_20260817/`。

## 关键指标

- 语义正确性:待记录(20 slot 独立 oracle 比对)
- SFT 验证集 Recall@3:**98.02%@完整 epoch(183,485 steps,GPU 0)**,≥98% 门槛
  达标;最终 top1 81.66%、policy_ce 0.4754,checkpoint 在
  `checkpoints/train_riichi_v16/sft/{best,latest}.pt`。
- 奖励契约更新:utility `[24,8,-12,-24]`(末位 -24 非零和)、σ_GRP=2.7112/
  σ_Score=4.2656 固化值不变、外层 clip ±10、内层分差 clip ±24 千点。
- PPO 性能基线(后两轮):待记录
- 宪法修订:待记录(预期 1.3.0→1.4.0 MINOR)
- V16-small 网络:总参数 3,081,603 / Actor 2,043,073(含 Top-3 Q scorer)

## 接续指引(新会话交接)

- 工作目录 `/mnt/disk1/hubowen/zenith`,分支 `V16`,Conda 环境 `Mahjong-AI`。
- 继续执行 `$speckit-implement`,任务清单与勾选状态在
  `specs/003-v16-model-rework/tasks.md`;设计在 `plan.md`、契约在 `contracts/`。
- 已全部完成:Phase 1(T001–T003)、Phase 2(T004–T007)、Phase 3 除 T015 外
  (T008–T014、T016);T015(bridge 装配)随 US2 模型联网一起做。
- **下一个任务:Phase 4(US2)**,从 T017 开始:参数量测试 → T018 `ModelConfig`
  v16 preset(d_model=256/Q=16/KV=4/head_dim=16/FFN=1088/shared=4/actor=1/
  critic=2)+ Offense/Defense 对称融合(删除 `offense_fusion` zero-init 分支与
  241 维 `q_head`)→ T019 Critic 适配 → T020–T023 SFT 编码器/manifest 契约
  (单一 `encoding_protocol_version=16`)→ T024 canary → T025 40% 全量编码 →
  T026 SFT 冒烟(Recall@3)。
- 关键环境事实:
  - `cargo test` 需要 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`;
  - 扩展重装:`cd RiichiEnv && python -m maturin develop --release` 与
    `cd RiichiEnv/riichienv-state-machine && python -m maturin develop --release`
    (需 `PYO3_PYTHON=$CONDA_PREFIX/bin/python`);
  - 赤五物理牌号 = {16, 52, 88};core 的 `calculate_shanten` 对部分手牌会 panic,
    向听一律用 state-machine(`riichi.analyze_hands` / `analyze_offense_v16`);
  - 役/番数内核:`riichienv.analyze_offense_v16`(O4/O5),等待牌取普通五
    (wait*4 为赤五时 +1);
  - 当前基线:pytest 258 通过,Rust workspace 124 通过。
- 测试文件:`tests/integration/test_v16_query_semantics.py`、
  `tests/integration/test_v16_replay_bridge.py`、
  `tests/unit/test_encoding_protocol.py`。

### 本次会话结束时(2026-08-16 续)

- 用户要求只写代码、不执行真实训练/重编码:运行类任务 **T024(canary)、T025
  (40% 全量编码)、T026(SFT 冒烟)、T032(GRP prepare+train)、T041(PPO 性能
  基线)、T051(场景复跑)** 全部未执行,代码与配置已就绪,待用户运行。
- **尚未完成的接线**:T038/T039 的完整 Ray 链路——V16 rollout(worker 的
  `prepare_v16`/GRP 边界奖励与 `RolloutWorker` 主循环)、`training/inference.py`
  的 V16 infer 路径、`PPOLearner` 的 V16 更新(PPO+Top-3 Q loss)、`train.py` 的
  V16 配置分发。算法契约函数已落地(`learner.select_top3_candidates`/
  `candidate_q_loss`、`worker.V16GrpBoundaryTracker`、`model.q_scores_v16`)。
- **尚未执行的删除类任务**:T043-T045(v13 feature_schema/actor_features/
  critic_features 公开汇总/zero-init/241 Q head 残留/效率奖励)、T046/T047/
  T049/T052/T053。这些与 v16 全量接线互为依赖,须在接线完成后逐主题提交。
- 提交记录(每主题一个 commit,基线 4d4bb2a 之后):governance → 协议常量 →
  rust-core → rust-state-machine → snapshot → action_query → 语义测试 →
  audit 骨架 → spec-kit 三件套 → T017-T019 网络 → T015 bridge → T020-T023 SFT
  → T027-T031 GRP → T033-T037/T040 PPO 算法 → T042 文档。

### V16 SFT/PPO 数据语义审计(2026-08-16 续)

- 完成一次抽样深度审计,详见 `report/V16_data_semantics_audit.md`;新增可复现
  脚本位于 `audit/reports/v16/scripts/`(static / sft_dataset / semantic_oracle /
  ppo_bridge),未修改训练代码、checkpoint 或数据集。
- 结果:全量 6,486 chunk、93,943,903 train + 959,045 validation 结构扫描零异常;
  140 局(8,880 决策、74,945 query 对)重编码与存量数据一致;独立语义 oracle
  通过;40 局 PPO 离线桥接(2,419 决策、20,848 action_id 解码)与 SFT 编码逐决策
  一致。
- 待处理发现 F1:`source_seat` 在 chi/pon/daiminkan/ron 恒为 N/A
  (`Observation.last_discard` 只返回牌 id,`_source_seat` 按 (seat, tile) 解包);
  该字段当前不进入 QueryEmbedding,不影响模型输入/训练数值,但需决定修复或明确
  定义为可选审计字段。
- 追加「独立逐 token 解码」扩展审计:`audit_v16_token_decoder.py` 从原始 MJAI
  事件独立重算 history/state/snapshot/query 因子,分层抽样 27 个 shard、27 局、
  1,974 决策,reach/chi/pon/daiminkan/ankan/kakan/dora/hora/ryukyoku 均覆盖,
  逐 token 与 `prepare_v16` 完全一致(0 mismatch)。已把结果补入
  `V16_data_semantics_audit.md` §5.1。

### V16 PPO 双卡运行归档(2026-08-17 晚)

- 运行至 update 59 停止(未见异常 traceback,疑似外部中断),已整体归档到
  `archive_20260817_run2`:checkpoint/metrics/performance/tensorboard 在
  `checkpoints/train_riichi_v16/archive_20260817_run2/ppo/`,运行日志在
  `logs/v16/archive_20260817_run2/ppo/v16_ppo_from_scratch.log`,1v3 评测在
  `audit/reports/v16/eval/archive_20260817_run2/`。
- update=30 1v3 vs SFT:first_place_rate=0.2556,top2_rate=0.4994,
  mean_rank=2.493,point_diff_mean=130.42,ci95=[-841.27, 1050.09]。
- 后续恢复使用 `riichi_ppo_v1/configs/v16_ppo_resume.yaml`(自包含副本,
  resume 指向归档 checkpoint_00030.pt)。

## 2026-08-17 update=60

- reward_mean=0.0020253 value_loss=0.25878 q_loss=0.35978 entropy=0.22854 actor_grad_norm=0.33176 critic_grad_norm=0.71245 shared_grad_norm=0.51978
- rollout_wall_s=12.613 update_wall_s=23.664 sps=916.86 grp_calls=182.67 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2831 top2_rate=0.5162 mean_rank=2.453 point_diff_mean=+1563.0 ci95=[536.6541666666668, 2513.16875]

## 2026-08-17 update=120

- reward_mean=0.010544 value_loss=0.18352 q_loss=0.30773 entropy=0.065727 actor_grad_norm=0.45288 critic_grad_norm=0.66405 shared_grad_norm=0.7105
- rollout_wall_s=13.162 update_wall_s=23.144 sps=911.96 grp_calls=184.67 history_pool_size=1
- 1v3 vs SFT: first_place_rate=0.2975 top2_rate=0.5581 mean_rank=2.368 point_diff_mean=+2880.1 ci95=[1905.4489583333334, 3785.607291666667]

## 2026-08-17 update=180

- reward_mean=0.0084298 value_loss=0.17579 q_loss=0.34276 entropy=0.047421 actor_grad_norm=0.53939 critic_grad_norm=0.66519 shared_grad_norm=0.81934
- rollout_wall_s=12.764 update_wall_s=24.008 sps=908.46 grp_calls=185 history_pool_size=3
- 1v3 vs SFT: first_place_rate=0.3013 top2_rate=0.5350 mean_rank=2.417 point_diff_mean=+2334.0 ci95=[1301.04375, 3333.7979166666664]

## 2026-08-17 update=60

- reward_mean=-0.0058506 value_loss=0.27856 q_loss=0.35705 entropy=0.3748 actor_grad_norm=0.31578 critic_grad_norm=0.66041 shared_grad_norm=0.47589
- rollout_wall_s=14.448 update_wall_s=22.436 sps=857.49 grp_calls=187 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2950 top2_rate=0.5138 mean_rank=2.455 point_diff_mean=+1477.3 ci95=[477.87187499999993, 2493.5489583333324]

## 2026-08-17 update=120

- reward_mean=0.0004091 value_loss=0.23801 q_loss=0.33961 entropy=0.22728 actor_grad_norm=0.42401 critic_grad_norm=0.59585 shared_grad_norm=0.62797
- rollout_wall_s=15.586 update_wall_s=20.854 sps=837.94 grp_calls=184 history_pool_size=1
- 1v3 vs SFT: first_place_rate=0.2869 top2_rate=0.5537 mean_rank=2.388 point_diff_mean=+2820.8 ci95=[1806.0239583333332, 3784.5166666666664]

## 2026-08-18 update=180

- reward_mean=-0.0092821 value_loss=0.23922 q_loss=0.33154 entropy=0.21957 actor_grad_norm=0.57663 critic_grad_norm=0.63016 shared_grad_norm=0.82135
- rollout_wall_s=17.196 update_wall_s=22.225 sps=805.96 grp_calls=193.33 history_pool_size=3
- 1v3 vs SFT: first_place_rate=0.3088 top2_rate=0.5537 mean_rank=2.366 point_diff_mean=+2603.1 ci95=[1621.1041666666667, 3576.2958333333327]

## 2026-08-18 update=240

- reward_mean=0.010473 value_loss=0.20436 q_loss=0.34389 entropy=0.21185 actor_grad_norm=0.60119 critic_grad_norm=0.59529 shared_grad_norm=0.876
- rollout_wall_s=16.383 update_wall_s=22.079 sps=816.59 grp_calls=181.33 history_pool_size=5
- 1v3 vs SFT: first_place_rate=0.2906 top2_rate=0.5350 mean_rank=2.417 point_diff_mean=+1682.5 ci95=[695.0895833333333, 2697.1062499999994]

## 2026-08-18 update=300

- reward_mean=-0.0031699 value_loss=0.21581 q_loss=0.37535 entropy=0.22519 actor_grad_norm=0.64673 critic_grad_norm=0.59039 shared_grad_norm=0.92761
- rollout_wall_s=21.245 update_wall_s=25.116 sps=668.54 grp_calls=185.67 history_pool_size=7
- 1v3 vs SFT: first_place_rate=0.2831 top2_rate=0.5306 mean_rank=2.446 point_diff_mean=+1184.2 ci95=[197.92812500000014, 2155.2364583333333]

## 2026-08-18 update=360

- reward_mean=0.0068551 value_loss=0.19765 q_loss=0.33587 entropy=0.20448 actor_grad_norm=0.67866 critic_grad_norm=0.58806 shared_grad_norm=0.96939
- rollout_wall_s=17.947 update_wall_s=21.691 sps=789.23 grp_calls=185 history_pool_size=9
- 1v3 vs SFT: first_place_rate=0.2981 top2_rate=0.5587 mean_rank=2.390 point_diff_mean=+2222.3 ci95=[1251.4333333333332, 3165.6718749999995]

## 2026-08-18 update=420

- reward_mean=-0.00027551 value_loss=0.23641 q_loss=0.34201 entropy=0.19151 actor_grad_norm=0.77024 critic_grad_norm=0.66964 shared_grad_norm=1.0689
- rollout_wall_s=15.986 update_wall_s=22.214 sps=841.85 grp_calls=190.67 history_pool_size=11
- 1v3 vs SFT: first_place_rate=0.3075 top2_rate=0.5444 mean_rank=2.390 point_diff_mean=+2480.0 ci95=[1460.1052083333334, 3453.926041666666]

## 2026-08-18 update=480

- reward_mean=-0.0082638 value_loss=0.24254 q_loss=0.38005 entropy=0.18075 actor_grad_norm=0.76267 critic_grad_norm=0.64626 shared_grad_norm=1.0749
- rollout_wall_s=16.854 update_wall_s=22.393 sps=824.04 grp_calls=189.67 history_pool_size=13
- 1v3 vs SFT: first_place_rate=0.2725 top2_rate=0.5256 mean_rank=2.459 point_diff_mean=+1065.7 ci95=[64.02604166666656, 2048.9635416666665]

## 2026-08-18 update=540

- reward_mean=0.01322 value_loss=0.24079 q_loss=0.37177 entropy=0.17649 actor_grad_norm=0.83015 critic_grad_norm=0.65566 shared_grad_norm=1.1816
- rollout_wall_s=17.96 update_wall_s=21.43 sps=792.44 grp_calls=185 history_pool_size=15
- 1v3 vs SFT: first_place_rate=0.3025 top2_rate=0.5544 mean_rank=2.375 point_diff_mean=+2584.7 ci95=[1593.0072916666666, 3510.631249999998]

## 2026-08-18 update=600

- reward_mean=-0.0053007 value_loss=0.25092 q_loss=0.35757 entropy=0.16948 actor_grad_norm=0.83149 critic_grad_norm=0.62392 shared_grad_norm=1.1676
- rollout_wall_s=17.01 update_wall_s=22.346 sps=817.37 grp_calls=184 history_pool_size=17
- 1v3 vs SFT: first_place_rate=0.2969 top2_rate=0.5444 mean_rank=2.414 point_diff_mean=+2179.0 ci95=[1131.6833333333336, 3182.2187499999995]

## 2026-08-18 update=660

- reward_mean=-0.004112 value_loss=0.2518 q_loss=0.3595 entropy=0.16211 actor_grad_norm=0.88535 critic_grad_norm=0.67161 shared_grad_norm=1.2458
- rollout_wall_s=16.829 update_wall_s=21.295 sps=816.17 grp_calls=186.33 history_pool_size=19
- 1v3 vs SFT: first_place_rate=0.2800 top2_rate=0.5225 mean_rank=2.446 point_diff_mean=+1524.9 ci95=[518.1270833333334, 2462.3916666666664]

## 2026-08-18 update=720

- reward_mean=-0.0022547 value_loss=0.26395 q_loss=0.36153 entropy=0.14237 actor_grad_norm=1.0213 critic_grad_norm=0.65723 shared_grad_norm=1.4258
- rollout_wall_s=17.176 update_wall_s=21.801 sps=795.4 grp_calls=192 history_pool_size=21
- 1v3 vs SFT: first_place_rate=0.2844 top2_rate=0.5275 mean_rank=2.436 point_diff_mean=+1450.3 ci95=[425.2489583333331, 2427.8927083333324]

## 2026-08-18 update=780

- reward_mean=-0.01276 value_loss=0.2608 q_loss=0.35554 entropy=0.13693 actor_grad_norm=0.92412 critic_grad_norm=0.64802 shared_grad_norm=1.2701
- rollout_wall_s=18.38 update_wall_s=22.405 sps=783.15 grp_calls=193 history_pool_size=23
- 1v3 vs SFT: first_place_rate=0.2712 top2_rate=0.5262 mean_rank=2.446 point_diff_mean=+1125.3 ci95=[159.28124999999983, 2120.872916666665]

## 2026-08-18 update=840

- reward_mean=-0.0059847 value_loss=0.28125 q_loss=0.36147 entropy=0.12714 actor_grad_norm=0.94811 critic_grad_norm=0.77055 shared_grad_norm=1.3153
- rollout_wall_s=16.435 update_wall_s=20.899 sps=815 grp_calls=186.33 history_pool_size=25
- 1v3 vs SFT: first_place_rate=0.2850 top2_rate=0.5256 mean_rank=2.446 point_diff_mean=+988.5 ci95=[35.690624999999955, 1937.2083333333335]

## 2026-08-18 update=900

- reward_mean=0.0087751 value_loss=0.28265 q_loss=0.34671 entropy=0.11062 actor_grad_norm=1.0212 critic_grad_norm=0.66019 shared_grad_norm=1.4094
- rollout_wall_s=17.636 update_wall_s=21.311 sps=799.03 grp_calls=187.67 history_pool_size=27
- 1v3 vs SFT: first_place_rate=0.2762 top2_rate=0.5162 mean_rank=2.461 point_diff_mean=+1116.2 ci95=[128.61666666666665, 2069.5249999999996]

## 2026-08-18 update=960

- reward_mean=-0.00036474 value_loss=0.29505 q_loss=0.36436 entropy=0.11124 actor_grad_norm=1.0375 critic_grad_norm=0.62518 shared_grad_norm=1.4565
- rollout_wall_s=15.887 update_wall_s=21.058 sps=825.46 grp_calls=182.33 history_pool_size=29
- 1v3 vs SFT: first_place_rate=0.2687 top2_rate=0.5212 mean_rank=2.462 point_diff_mean=+1090.4 ci95=[99.1572916666666, 2094.2291666666656]

## 2026-08-18 update=1020

- reward_mean=0.0026342 value_loss=0.30737 q_loss=0.36376 entropy=0.10302 actor_grad_norm=1.0057 critic_grad_norm=0.54377 shared_grad_norm=1.4427
- rollout_wall_s=17.242 update_wall_s=21.216 sps=783.63 grp_calls=188.67 history_pool_size=31
- 1v3 vs SFT: first_place_rate=0.2831 top2_rate=0.5181 mean_rank=2.440 point_diff_mean=+1400.4 ci95=[408.2229166666666, 2344.851041666665]

## 2026-08-18 update=1080

- reward_mean=0.01052 value_loss=0.30835 q_loss=0.36183 entropy=0.10414 actor_grad_norm=1.1906 critic_grad_norm=0.59424 shared_grad_norm=1.7094
- rollout_wall_s=16.956 update_wall_s=21.558 sps=816.74 grp_calls=189 history_pool_size=33
- 1v3 vs SFT: first_place_rate=0.2875 top2_rate=0.5463 mean_rank=2.399 point_diff_mean=+1765.1 ci95=[783.1562500000001, 2729.672916666667]

## 2026-08-18 update=1140

- reward_mean=-0.013111 value_loss=0.33055 q_loss=0.35564 entropy=0.093661 actor_grad_norm=1.3199 critic_grad_norm=0.49688 shared_grad_norm=1.8837
- rollout_wall_s=16.346 update_wall_s=21.569 sps=826.75 grp_calls=189 history_pool_size=35
- 1v3 vs SFT: first_place_rate=0.2913 top2_rate=0.5450 mean_rank=2.390 point_diff_mean=+1958.0 ci95=[1035.3520833333332, 2923.564583333333]

## 2026-08-18 update=1200

- reward_mean=0.013965 value_loss=0.34616 q_loss=0.37309 entropy=0.091836 actor_grad_norm=1.3309 critic_grad_norm=0.33039 shared_grad_norm=1.8641
- rollout_wall_s=16.48 update_wall_s=21.589 sps=816.75 grp_calls=187.67 history_pool_size=37
- 1v3 vs SFT: first_place_rate=0.2806 top2_rate=0.5306 mean_rank=2.429 point_diff_mean=+1437.0 ci95=[472.4916666666667, 2437.7729166666663]

## 2026-08-18 run4 归档(update=1200 全量完成)

- 本次双卡 PPO 训练(2026-08-17 21:53 → 2026-08-18 13:45,1200 iterations)产物
  已整体归档到 `archive_20260818_run4`:
  - checkpoint/metrics/performance/tensorboard:
    `checkpoints/train_riichi_v16/archive_20260818_run4/ppo/`(41 个
    checkpoint + latest.pt)
  - 训练日志:`logs/v16/archive_20260818_run4/ppo_run4.log`
  - 1v3 例行评测(40 次更新点, vs SFT)+ 2 个 v14 对比评测 + 中间 shards:
    `checkpoints/train_riichi_v16/archive_20260818_run4/eval/`
  - 3 个 vs_v14_base 1v3 评测日志:同目录。
- 1v3 vs SFT 全程轨迹(update=30→1200):first_place_rate 0.2556→0.2806,
  峰值出现在 update=150(0.3287, top2 0.5731);中后段(update≥480)基本在
  0.27–0.30 区间震荡,未见持续上升趋势。
- vs v14 基线(checkpoint_00510)同种子对比(2400 半庄):
  - u090:first_rate=0.249, point_diff=−1473.4(落后)
  - u120:first_rate=0.255, point_diff=−735.0
  - u1050:first_rate=0.236, point_diff=−1368.7
  - 6 候选集 experiment(u090 起):first_rate 0.21–0.25 区间,均落后 v14
    (see `vs_v14_base_same_seeds_2400_6candidates_plus_base3v1.json`)
- 结论:1200 updates 后 vs SFT 接近持平(mean_rank 2.4 左右),但 vs v14 基线
  全面落后;下一步将进行新一轮超参数方案修改(本轮配置自包含副本在
  `riichi_ppo_v1/configs/v16_ppo.yaml`,resume 用
  `riichi_ppo_v1/configs/v16_ppo_resume.yaml`)。
