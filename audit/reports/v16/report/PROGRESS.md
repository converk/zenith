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

## 关键指标

- 语义正确性:待记录(20 slot 独立 oracle 比对)
- SFT 验证集 Recall@3:**98.02%@完整 epoch(183,485 steps,GPU 0)**,≥98% 门槛
  达标;最终 top1 81.66%、policy_ce 0.4754,checkpoint 在
  `checkpoints/train_riichi_v16/sft/{best,latest}.pt`。
- 奖励契约更新:utility `[24,8,-12,-24]`(末位 -24 非零和)、σ_GRP=2.7112/
  σ_Score=4.2656 固化值不变、外层 clip ±10、内层分差 clip ±24 千点。
- PPO 性能基线(后两轮):待记录
- 宪法修订:待记录(预期 1.3.0→1.4.0 MINOR)
- V16 网络:总参数 7,811,587 / Actor 5,525,761(含 Top-3 Q scorer)

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
