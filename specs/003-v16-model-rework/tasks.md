# Tasks: V16 模型重构与训练

**Input**: Design documents from `/specs/003-v16-model-rework/`

**Prerequisites**: plan.md、spec.md、research.md、data-model.md、contracts/、
quickstart.md(均已就绪)

**Tests**: 本特性把语义与业务正确性测试作为硬性门槛(任务约束④),因此每个用户
故事都包含「先写测试、确认失败、再实现」的任务。

**Organization**: 任务按用户故事分阶段,便于独立实现与验证。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可与同阶段其他任务并行(不同文件、无未完成依赖)
- **[Story]**: 所属用户故事(US1…US5)
- 描述必须带具体文件路径;完成后把 `- [ ]` 改为 `- [X]`

## Phase 1: Setup(共享基础设施)

**Purpose**: 产物目录与基线

- [X] T001 创建 V16 产物目录骨架:`audit/reports/v16/{design,eval,report,scripts}`、
  `logs/v16/`、`checkpoints/train_riichi_v16/{sft,grp,ppo}`,并初始化
  `audit/reports/v16/report/PROGRESS.md`
- [X] T002 记录变更前基线:全仓 `python -m pytest riichi_ppo_v1/tests -q` 与
  `cargo test --manifest-path RiichiEnv/Cargo.toml` 全绿快照写入
  `audit/reports/v16/report/PROGRESS.md`
- [X] T003 [P] 确认当前分支为 `V16`、工作区无无关改动,把 HEAD commit 写入
  `audit/reports/v16/report/PROGRESS.md`

---

## Phase 2: Foundational(阻塞性前置)

**Purpose**: 所有用户故事共用的协议版本与治理基础

**⚠️ CRITICAL**: 此阶段完成前不得开始任何用户故事实现

- [X] T004 经 `$speckit-constitution` 修订 `.specify/memory/constitution.md`
  Principle II:现行契约声明改为「信息编码协议 v16;v16 是当前活跃实验代」,记录
  Sync Impact Report(预期 1.3.0→1.4.0 MINOR)
- [X] T005 新建 `riichi_ppo_v1/model/encoding_protocol.py`:`ENCODING_PROTOCOL_VERSION=16`
  单一来源,承载 20 个 query slot 的语义/基数/N/A 规则与终局动作约定常量(逐条对照
  `contracts/actor-input-v16.md`)
- [X] T006 更新 `riichi_ppo_v1/model/schema.py`:`TOKEN_SCHEMA_VERSION` 改为引用
  `encoding_protocol.py` 的单一常量;136/34/241 领域常量保持不变
- [X] T007 [P] 新建 `riichi_ppo_v1/tests/unit/test_encoding_protocol.py`,校验协议
  常量表与 contracts 逐项一致(cardinality 总表、N/A 规则、终局约定)

**Checkpoint**: 宪法契约与协议常量就位,用户故事可开始

---

## Phase 3: User Story 1 - 统一 Action Query 输入契约与语义落地 (Priority: P1) 🎯 MVP

**Goal**: Actor 输入按「Objective Facts + Compact Snapshot + 每合法动作一对
10-slot Offense/Defense Query」编码,删除全部 Derived Features,20 个 slot 与独立
oracle 100% 一致。

**Independent Test**: 抽样局面与全部边界局面清单上,每个合法动作的 20 slot 与
独立 oracle 重算逐项一致;Actor 无隐藏信息;categorical 零越界;回放/桥接一致。

### Tests for User Story 1(先写,必须先失败)

- [X] T008 [P] [US1] 编写 `riichi_ppo_v1/tests/integration/test_v16_query_semantics.py`:
  独立 oracle 直接用 RiichiEnv Observation 与 core `HandEvaluator`/公开河牌重算
  20 slot,与编码器输出逐项比对(禁止编码器自证)
- [X] T009 [P] [US1] 编写 `riichi_ppo_v1/tests/protocol/test_v16_cardinalities.py`:
  覆盖全部 slot 基数、N/A 规则、bucket 边界与终局/流局/立直/吃碰杠约定
- [X] T010 [P] [US1] 编写 `riichi_ppo_v1/tests/integration/test_v16_replay_bridge.py`:
  回放局面→编码 tensor 一致、Actor 不含隐藏信息、特权信息只出现在 critic_factors

### Implementation for User Story 1

- [X] T011 [US1] 新增 `RiichiEnv/riichienv-core/src/offense_analysis.rs`:实现 O0
  动作后向听、O1/O2 有效牌种类与剩余枚数、O3 等待、O4 有无役、O5 基础番数、O6
  振听、O8 可立直(复用 `shanten.rs`、`HandEvaluator`、`yaku`),PyO3 暴露并附 Rust
  单测
- [X] T012 [P] [US1] 扩展 `RiichiEnv/riichienv-state-machine/src/analysis.rs`:
  实现 D0–D5 现物/筋、D9 公开出现数、D6–D8 动作后现物库存、每对手 7 项摘要
  (立直/立直巡目/副露数/门清/舍牌数/手切/摸切),保持公开模块名 `riichi` 不依赖
  `riichienv`,附 Rust 单测
- [X] T013 [P] [US1] 新建 `riichi_ppo_v1/model/snapshot.py`:编码基础场况(场风/
  局数/庄家/本场/立直棒/剩余牌数/宝牌指示/四家点数/当前顺位)+ 3 个相对分差 +
  3×7 对手摘要;不含三家完整牌河/副露
- [X] T014 [P] [US1] 新建 `riichi_ppo_v1/model/action_query.py`:为每个合法动作
  计算一对 10-slot Offense/Defense Query 并按单 token 聚合(E_action+E_queryType
  +ΣE_{slot}(answer)→LayerNorm/Projection 到 d_model=256)
- [ ] T015 [US1] 修改 `riichi_ppo_v1/model/bridge.py`:装配 Objective Facts+
  Snapshot+queries;critic_factors 只保留三家对手手牌与后续 5 张牌山
- [X] T016 [US1] 运行 T008–T010 直至全绿;语义硬门槛结果写入
  `audit/reports/v16/report/PROGRESS.md`

**Checkpoint**: 输入契约语义正确,US1 可独立交付验证

---

## Phase 4: User Story 2 - V16 网络结构与对称策略头 + SFT 从头训练 (Priority: P1)

**Goal**: 网络扩容到 d_model=256/16Q/4KV/FFN=1088/4+1 Actor/4+2 Critic(总参数
7.5–7.8M),Offense/Defense 对称融合、无 zero-init,并用重编码数据集从头 SFT。

**Independent Test**: 参数统计在容差内;策略头对称融合;canary/全量编码 manifest
符合 v16 单版本契约;SFT 可训练且验证集 Recall@3 记录达标。

### Tests for User Story 2(先写,必须先失败)

- [X] T017 [P] [US2] 编写 `riichi_ppo_v1/tests/unit/test_v16_architecture.py`:
  参数量(总 7.5–7.8M ±0.3M、Actor 推理约 5.3M ±0.3M)、d_model=heads×head_dim
  校验、Offense/Defense 对称融合、不存在 zero-init 与 241 维 Q head

### Implementation for User Story 2

- [X] T018 [US2] 修改 `riichi_ppo_v1/model/architecture.py`:`ModelConfig` 增加
  v16 preset(d_model=256、Q=16、KV=4、head_dim=16、FFN=1088、shared=4、
  actor=1、critic=2);策略头改为对称融合(concat→Linear 512→256→SiLU→Policy
  MLP),删除 `offense_fusion` zero-init 分支
- [X] T019 [US2] 修改 `riichi_ppo_v1/model/architecture.py` Critic 路径:公共序列
  使用新 Snapshot,特权输入(三家手牌+后 5 牌山)与 Value Query 保持不变,不追加
  Action Query Token
- [ ] T020 [P] [US2] 修改 `riichi_ppo_v1/sft/precompute.py`:接入 V16 编码器,
  manifest 写入 `format=riichi-sft-encoded-v16`、单一
  `encoding_protocol_version=16` 与协议契约 sha256,删除多版本字段
- [ ] T021 [P] [US2] 修改 `riichi_ppo_v1/sft/contract.py`、
  `riichi_ppo_v1/sft/data.py` 与 `riichi_ppo_v1/sft/train.py`:新增
  `validate_v16_manifest`,并把 `data.py`(约 L238–241 对
  `riichi-sft-encoded-v*` 无条件调用 `validate_v13_manifest`)与训练入口的
  manifest 校验切换到 v16 契约
- [ ] T022 [P] [US2] 新建 `riichi_ppo_v1/configs/v16_sft.yaml`:自包含完整配置
  (v16 preset、`checkpoint_dir: checkpoints/train_riichi_v16/sft`、日志
  `logs/v16/`);节奏键不得复制(沿用 sft.yaml 单点定义)
- [ ] T023 [P] [US2] 修改 `riichi_ppo_v1/sft/train.py`:接入 V16 输入与 checkpoint
  路径,延续 3000 steps 节奏并输出 train/validation top3(Recall@3)
- [ ] T024 [US2] 跑 canary 编码(quickstart 场景 2,`datasets/tenhou_sft_canary_v16`):
  审计每个合法/专家动作组非零、数值越界为 0、manifest 契约正确,通过后删除 canary
- [ ] T025 [US2] 跑 40% 全量编码到
  `datasets/tenhou_sft_2024_2025_encoded_40pct_v16`(quickstart 场景 3):
  train/validation 决策计数与独立统计一致,manifest 哈希稳定
- [ ] T026 [US2] 跑 V16 SFT 冒烟训练(quickstart 场景 4),验证集 Recall@3 ≥ 98%
  作为进入 PPO 的前置检查,实际值写入 `audit/reports/v16/report/PROGRESS.md`

**Checkpoint**: US1+US2 可组合交付(新输入 + 新网络 + 新 SFT 权重)

---

## Phase 5: User Story 3 - GRP 模型、数据集与离线训练 (Priority: P1)

**Goal**: 从 Tenhou 数据构造 GRP 数据集(每半庄 4 视角、prefix→最终排名),训练
50–70K 参数的轻量 GRP 并冻结。

**Independent Test**: 旋转/prefix 标签正确;参数量 50–70K;排名预测显著优于均匀
随机;训练后冻结且 PPO 不更新。

### Tests for User Story 3(先写,必须先失败)

- [ ] T027 [P] [US3] 编写 `riichi_ppo_v1/tests/unit/test_grp.py`:4 视角旋转、
  首局 START、prefix→最终排名标签、CE 损失形状、参数量 50–70K、冻结权重不变

### Implementation for User Story 3

- [ ] T028 [US3] 新建 `riichi_ppo_v1/model/grp.py`:GRP 网络(Linear 64→2 层
  GRU(64)→Linear 64→32→SiLU→Linear 32→4→Rank Softmax)与输入契约常量
- [ ] T029 [US3] 新建 `riichi_ppo_v1/training/grp/prepare.py`:从
  `datasets/tenhou_sft_2024_2025` 构造 `datasets/tenhou_grp_2024_2025_v16`
  (40% 划分、4 视角、prefix 标签),离线计算并固化 σ_GRP/σ_Score 到数据集 JSON
- [ ] T030 [US3] 新建 `riichi_ppo_v1/training/grp/train.py`:GRP 离线训练入口,
  checkpoint 保存到 `checkpoints/train_riichi_v16/grp`(含配置快照)
- [ ] T031 [US3] 新建 `riichi_ppo_v1/configs/v16_grp.yaml`:自包含 GRP 训练配置
- [ ] T032 [US3] 跑 GRP prepare+train 冒烟(quickstart 场景 5),验证损失下降、
  验证集排名预测优于均匀随机、冻结校验通过,结果写入
  `audit/reports/v16/report/PROGRESS.md`

**Checkpoint**: GRP 可独立交付(与 US1 并行、不依赖 US2)

---

## Phase 6: User Story 4 - PPO 集成:Top-3 Q-boosting 与 GRP+Score 奖励 (Priority: P2)

**Goal**: PPO 使用 70% 归一化 GRP delta + 30% 归一化小局分差,Critic 特权输入 +
Top-3 Q-boosting(候选 Top-3∪行为动作 ≤4、h_a detach),训练稳定跑通。

**Independent Test**: 奖励公式/截断/固定 σ 正确;Q 候选集与 detach 正确;性能基线
3 轮跑通,后两轮单独报告。

### Tests for User Story 4(先写,必须先失败)

- [ ] T033 [P] [US4] 编写 `riichi_ppo_v1/tests/unit/test_v16_q_scorer.py`:训练
  候选 = Top-3 ∪ 行为动作(≤4)、boost = Top-3、动作表示 detach、Q loss 不直接
  更新 Actor 参数
- [ ] T034 [P] [US4] 编写 `riichi_ppo_v1/tests/unit/test_v16_reward.py`:奖励公式
  R=0.7·clip(R_GRP/σ_GRP,±5)+0.3·clip(clip(Δscore/1000,±12)/σ_Score,±5)、
  utility [12,4,-6,-10]、σ 训练期不变、终局用真实排名 utility

### Implementation for User Story 4

- [ ] T035 [US4] 修改 `riichi_ppo_v1/model/architecture.py`:新增 Top-3 Q scorer
  (输入 [z_critic; detach(h_a)] →512→256→SiLU→1),删除 241 维 `q_head`
- [ ] T036 [US4] 移除独立半庄排名奖励分量:删除
  `riichi_ppo_v1/training/rewards/terminal.py` 的
  `terminal_hanchan_rank_rewards` 及其在 `riichi_ppo_v1/training/worker.py`
  的调用(排名效用已由 GRP 终局 V_terminal=U(rank) 覆盖,避免双计);utility
  [12,4,-6,-10] 仅由 `riichi_ppo_v1/training/grp/reward.py` 提供
- [ ] T037 [US4] 新建 `riichi_ppo_v1/training/grp/reward.py`:GRP 期望/delta 与
  归一化组合(加载离线固化的 σ_GRP/σ_Score)
- [ ] T038 [US4] 修改 `riichi_ppo_v1/training/learner.py`:Top-3 Q loss/boost
  与候选集、GRP+分差奖励装配、GRP 权重冻结检查
- [ ] T039 [US4] 修改 `riichi_ppo_v1/training/worker.py`:小局边界 GRP 推理
  (每局边界一次,不每动作执行)与奖励计算,并补充断言测试(SC-011):GRP 调用次数
  等于小局边界数、不随动作数增长
- [ ] T040 [US4] 新建 `riichi_ppo_v1/configs/v16_ppo.yaml`:自包含完整配置
  (init_model=v16 SFT checkpoint、1v3 对手/种子/输出目录显式给出、
  `eval1v3_output_dir: audit/reports/v16/eval`);1v3 节奏键不得复制
- [ ] T041 [US4] 跑 PPO 性能基线 3 轮(quickstart 场景 6,target_kl=0.0、
  update_epochs=4、kyokus_per_worker=16、CUDA_DEVICE=0,1、learner_gpus=2),
  首轮预热、单独报告后两轮耗时与性能指标,冒烟产物删除

**Checkpoint**: US1–US4 全部功能闭环,PPO 训练可运行

---

## Phase 7: User Story 5 - 治理闭环:协议、版本、清理与文档 (Priority: P2)

**Goal**: 协议文档、宪法契约、旧代码零引用清理、常量收敛、自包含配置、产物路径
与评测机制全部合规。

**Independent Test**: 删除目标 rg 零引用 + 全仓测试通过;协议文档与实现差异为 0;
评测机制常量 diff 为空;README/docs 路径同步。

### Implementation for User Story 5

- [ ] T042 [US5] 新增 `riichi_ppo_v1/docs/v16_input_protocol.md`,并同步
  `riichi_ppo_v1/docs/KyokuEventTupleProtocol.md` 的状态后缀/查询段落(与
  `contracts/actor-input-v16.md` 逐条一致)
- [ ] T043 [P] [US5] 清理 v13 契约与派生特征:`riichi_ppo_v1/model/feature_schema.py`、
  `riichi_ppo_v1/model/actor_features.py`、`critic_features.py` 的
  `encode_public_summary`(每主题一个 commit;先全仓 rg 零引用、测试通过再删)
- [ ] T044 [P] [US5] 清理模型旧分支:zero-init `offense_fusion` 与
  `action_value` 241 维 Q head 残留(先 rg 零引用、测试通过)
- [ ] T045 [P] [US5] 清理奖励旧组合:`training/rewards/efficiency.py` 与
  `decision.py` 中 V16 不再引用的候选分析/奖励逻辑(先 rg 零引用、每主题 commit)
- [ ] T046 [US5] 领域常量收敛核查:136/34/241 与全部 slot 基数全仓唯一来源
  (rg 校验,无散落魔法数字)
- [ ] T047 [US5] 同步 README、`docs/directory-responsibilities.md`、AGENTS.md
  与代码路径(新增 `model/encoding_protocol.py`、`training/grp/`、
  `docs/v16_input_protocol.md` 等)
- [ ] T048 [US5] 评测机制不变核查:`git diff -- riichi_ppo_v1/evaluation/mechanism.py
  riichi_ppo_v1/configs/sft.yaml` 为空
- [ ] T049 [US5] 全仓 `python -m pytest` + `cargo test` 全绿;删除目标零引用复核
- [ ] T050 [US5] 更新 `audit/reports/v16/report/PROGRESS.md`:完成状态、Recall@3、
  性能基线后两轮统计、宪法修订版本

**Checkpoint**: 全部完成判定达成,可进入 analyze/converge

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: 全链复跑与一致性收口

- [ ] T051 [P] 按 `quickstart.md` 场景 1–7 逐项复跑并核对输出(含语义硬门槛与治理门)
- [ ] T052 运行 `$speckit-analyze` 做 spec/plan/tasks 一致性分析,修复差异后再
  `$speckit-converge` 核对实现覆盖
- [ ] T053 冒烟结束清理 `logs/v16/` 临时日志与结果文件,确认工作区按主题提交、
  无残留

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup(Phase 1)→ Foundational(Phase 2,阻塞全部)
- US1(Phase 3)与 US3(Phase 5)在 Foundational 后即可开始,互不依赖
- US2(Phase 4)依赖 US1(新编码器)
- US4(Phase 6)依赖 US2(v16 SFT 权重)与 US3(冻结 GRP)
- US5(Phase 7)依赖 US1–US4
- Polish(Phase 8)依赖全部故事

### Within Each User Story

- 测试任务(TDD)必须先写并确认失败,再实现,最后全绿
- 环境分析(core/state-machine)先于编码装配(bridge)
- 配置与入口在实现之后;运行类任务(编码/训练冒烟)在相应代码与测试完成后

### Parallel Opportunities

- Phase 2:T007 与 T005/T006 并行
- Phase 3:T008/T009/T010 并行;T011(core)与 T012(state-machine)并行;T013/T014
  在 T011+T012 完成后并行
- Phase 4:T020/T021/T022/T023 在 T018/T019 后并行
- Phase 5/6:各阶段测试任务并行
- Phase 7:T043/T044/T045 并行(不同文件,均先 rg 零引用)
- 跨故事:US1 与 US3 可并行推进

## Parallel Example: User Story 1

```text
# 先并行写三个测试(确认失败):
Task: "test_v16_query_semantics.py 独立 oracle 逐 slot 比对"
Task: "test_v16_cardinalities.py 基数/N-A/边界"
Task: "test_v16_replay_bridge.py 回放/桥接一致性"

# 再并行实现两个环境分析层:
Task: "core offense_analysis.rs(O0–O6/O8)"
Task: "state-machine analysis.rs(D0–D9、对手摘要)"
```

## Implementation Strategy

### MVP First(User Story 1 Only)

1. Phase 1 Setup + Phase 2 Foundational(宪法修订 + 协议常量)
2. Phase 3 US1:环境分析 + 编码装配 + 语义硬门槛全绿
3. **STOP and VALIDATE**:任一 query slot 与独立 oracle 不一致即失败

### Incremental Delivery

1. US1 语义契约(可独立验证)→
2. US2 网络 + SFT 重编码与训练(Recall@3 门)→
3. US3 GRP(可与 US1 并行)→
4. US4 PPO 集成(性能基线)→
5. US5 治理闭环(协议/清理/常量/评测不变)

### Governance Ordering

- T004(宪法 Principle II 修订)必须在任何引用「协议 v16」为现行契约的代码提交前
  完成
- 删除类任务(T043–T045)必须逐个主题、rg 零引用、测试通过后提交,可独立回滚

## Notes

- [P] 任务 = 不同文件、无未完成依赖
- 每个用户故事可独立完成与验证;完成一项就把任务勾选为 `[X]`
- 提交遵循「每主题一个 commit、测试通过、可独立回滚」
- 冒烟测试结束必须删除其日志与结果文件
- 禁止在实验配置复制评测机制节奏键;机制改动必须走宪法修订
