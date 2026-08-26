# Tasks: V18 当前局面输入与 Actor 决策架构重构

**Input**: Design documents from `specs/010-v18-current-state-input-sft/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Required by the feature. Story test tasks precede their implementation tasks.

**Organization**: Tasks are grouped by user story and executed in dependency order. Every completed
task is changed to `[X]` only after its stated validation passes. 注释一律中文。

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 固化重构起点与构建基线。

- [X] T001 记录初始仓库状态、已确认基线（扩展从源码重建成功、25 个 V18 测试通过）与 PPO 不参与范围，写入 `audit/reports/v18/report/PROGRESS.md`
- [X] T002 建立本 feature 的协议契约基准 `specs/010-v18-current-state-input-sft/contracts/v18-current-state-contract.md`（段/kind/字段/域/分隔符/注意力/上界）
- [X] T003 全仓只读盘点旧契约 reference（history_*/snapshot_*/`_isolated_action_layout`/54 行/局部 position），分类为活跃路径与 PPO 待迁移，输出到 `audit/reports/v18/report/PROGRESS.md`

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 先锁定 Python schema 单源与语义校验骨架，再实现 Rust 编码器；两者用真实 fixture 对齐。

- [X] T004 重写 `riichi_ppo_v1/model/encoding_protocol.py`：segment/kind/separator/类别字段/域/`TOKEN_ROW_WIDTH=32`/`TOKEN_NUMERIC_WIDTH=8`/`CONTEXT_TOKENS=256`/query 槽位（O0-O9/D0-D9 沿用），并更新 `riichi_ppo_v1/model/schema.py` 常量单点
- [X] T005 新增 `RiichiEnv/riichienv-python/src/current_state_encoding.rs`：`CurrentStateBatch` + `prepare_current_state_batch`，实现桌况/自身手牌/SELF_STATE_ANALYSIS/四家 PLAYER/三家牌河+两摘要/当前副露/34 TILE_STATE/三个 OPPONENT_ANALYSIS 的编码（复用 `riichi::shanten` 与 analysis 内核），并在 `riichienv-python/src/lib.rs` 注册
- [X] T006 将 `RiichiEnv/riichienv-state-machine/src/analysis.rs` 的 `suji_safe`/`wall_class` 改为 `pub`（或新增等价 `pub` 辅助）供新编码器复用，并加注释说明
- [X] T007 重写 `riichi_ppo_v1/model/semantic_validation.py`：按契约 §2/§3/§6 校验序列顺序、域、摘要长度/顺序、tile-state 守恒、无历史/tiles_left/独立供牌字段、action 排序与 pair、Critic 隔离，fail closed
- [X] T008 重建两个 Rust 扩展（`bash RiichiEnv/scripts/install_conda_extension.sh`）并跑 `cargo test` + Python 冒烟：`riichienv.prepare_current_state_batch` 在真实 replay fixture 上产生合法行
- [X] T009 [P] 新增/重写 Rust 编码器单测（桌况模式、自手锁定、牌河 stage/supplied/age、摘要槽位、meld kakan 不重复、tile-state 守恒、opponent analysis 计数）于 `RiichiEnv/riichienv-python/src/current_state_encoding.rs` 与 `riichi_ppo_v1/tests/unit/test_v18_encoding_protocol.py`
- [X] T010 [P] 重写模型测试 fixture `riichi_ppo_v1/tests/v18_fixtures.py`（新 actor 序列构造器 + critic 行构造器），并重写 `unit/test_v18_encoding_protocol.py`（新 schema 顺序/域/分隔符/无旧字段）

**Checkpoint**: 新 schema 单源 + Rust 编码器可产出合法行，语义校验可拒绝越界/错序。

## Phase 3: User Story 1 - 当前局面公共快照编码 (Priority: P1) 🎯 MVP

**Goal**: 公共快照（三家完整牌河/摘要/副露/34 tile-state/自身手牌/self 分析/player/桌况）编码正确、无旧输入。

**Independent Test**: 真实 replay + 合成边界（0/1/5/6/7/18 张牌河、0/4 副露、暗杠、kakan、红五、
四张全见、one-chance/no-chance、立直三态）逐决策断言与契约一致；无 tiles_left/独立供牌/历史 token。

### Tests for User Story 1

- [X] T011 [P] [US1] 新增公共快照结构/顺序/域/摘要/牌河/meld/tile-state 单元测试于 `riichi_ppo_v1/tests/unit/test_v18_encoding_protocol.py` 与 `riichi_ppo_v1/tests/unit/test_v18_snapshot.py`（重写为 current-state 语义）
- [X] T012 [P] [US1] 新增真实 replay → Rust 批编码一致性测试（覆盖早/中/晚巡、立直、副露、红五）于 `riichi_ppo_v1/tests/integration/test_v18_replay_bridge.py`
- [X] T013 [P] [US1] 新增合成极端 fixture：三家牌河 0/1/5/6/7/18、副露 0/1/2/4、暗杠、kakan、四张全见，断言严格上界与无截断，于 `riichi_ppo_v1/tests/integration/test_v18_encoding_bridge.py`

### Implementation for User Story 1

- [X] T014 [US1] 实现 `riichi_ppo_v1/model/current_state.py`：包装 `riichienv.prepare_current_state_batch`，输出按决策切分的 shared+analysis 行，并调用现有 `encode_action_queries_batch_native` 得到 query 行
- [X] T015 [US1] 更新 `riichi_ppo_v1/model/bridge.py` 的 `PreparedBatch` 与 `BatchedStateBridge.prepare`：新字段 `actor_factors/actor_numeric/actor_lengths/query_rows/query_action_ids/query_pair_counts/legal_mask/critic_factors/critic_lengths`；保留 action_jsons/解码/mask 逻辑
- [X] T016 [US1] 删除 `riichi_ppo_v1/model/snapshot.py`（先全仓 `rg` 零引用）并把 `model/critic_features.py` 切换到新行宽（segment 4/5、kind 13/14、1-based 相对座次/position）
- [X] T017 [US1] 运行 US1 Rust/单元/集成测试并记录证据到 `audit/reports/v18/report/PROGRESS.md`

**Checkpoint**: 公共快照独立可用且严格上界验证通过。

## Phase 4: User Story 2 - Actor-only Opponent Analysis 与 Action Query (Priority: P1)

**Goal**: 三个 Opponent Analysis（Actor-only、Query 前、参与 logits）与按 action ID 升序的 O/D 对正确。

**Independent Test**: 打乱环境合法动作输入顺序→规范排序→序列与按 action ID 对齐 logits 不变；
每个 Analysis 有效字段改变影响 embedding 并在 fixture 中影响 action logits；O0-O9/D0-D9 语义保留。

### Tests for User Story 2

- [X] T018 [P] [US2] 新增 Opponent Analysis 字段敏感性/只读公开信息/参与 logits 测试于 `riichi_ppo_v1/tests/unit/test_v18_dense_embedding.py`
- [X] T019 [P] [US2] 重写 action 排序/pair 隔离/supplier 域/非法 -inf 测试于 `riichi_ppo_v1/tests/integration/test_v18_query_semantics.py` 与 `riichi_ppo_v1/tests/protocol/test_protocol_matrix.py`
- [X] T020 [P] [US2] 新增「打乱合法动作输入→规范排序→张量/logits 不变」测试于 `riichi_ppo_v1/tests/unit/test_v18_architecture.py`

### Implementation for User Story 2

- [X] T021 [US2] 在 `riichi_ppo_v1/model/architecture.py` 实现新前向：actor 序列装配、查询对定位、`raw_policy_logits` scatter、非法动作 -inf、`action_fusion`+`policy_mlp`
- [X] T022 [US2] 在 `riichi_ppo_v1/model/dense_embedding.py` 实现 Opponent Analysis 与 Action Query 的 DENSE 融合（每槽独立表、concit、512 融合、gated MLP）
- [X] T023 [US2] 更新 `riichi_ppo_v1/model/action_query.py`/`native_encoding.py` 行常量引用并保持 query 行语义不变
- [X] T024 [US2] 运行 US2 测试并记录排序/隔离/字段敏感性证据到 `audit/reports/v18/report/PROGRESS.md`

**Checkpoint**: Actor 尾部信息正确且顺序无关。

## Phase 5: User Story 3 - RoPE、分隔符、注意力布局与信息边界 (Priority: P1)

**Goal**: 全 token RoPE 连续位置；公共双向 GQA；Actor 结构化 mask；Critic 隔离。

**Independent Test**: 固定 Actor 输入只改三家闭手/未来五张→逐 action ID raw logits 不变；改 Critic
私有→value 变化；Critic shape/length 不含 Analysis/Action；分隔符顺序 fail closed；打乱动作顺序
位置与 logits 不变。

### Tests for User Story 3

- [X] T025 [P] [US3] 新增 RoPE 连续唯一/分隔符占位/分支续接测试于 `riichi_ppo_v1/tests/unit/test_v18_architecture.py`
- [X] T026 [P] [US3] 新增共享双向 mask/Analysis 可见性/Action pair 隔离/padding 不可见测试于 `riichi_ppo_v1/tests/unit/test_v18_architecture.py`
- [X] T027 [P] [US3] 重写 Actor/Critic 信息边界测试（私有输入改变不影响 Actor；Critic 无 Analysis/Action）于 `riichi_ppo_v1/tests/integration/test_v18_information_boundaries.py`

### Implementation for User Story 3

- [X] T028 [US3] 在 `riichi_ppo_v1/model/architecture.py` 实现 `_bidirectional_layout`/`_actor_structured_layout`/`_critic_layout` 与新位置生成（删除 `_isolated_action_layout` 与局部 position）
- [X] T029 [US3] 实现分离的 shared backbone（双向）/actor backbone（结构化）/critic backbone（全双向+value 末尾），Critic 分支不接收 Analysis/Action 行
- [X] T030 [US3] separator 单点定义与严格顺序校验（`semantic_validation.py` + 架构层）
- [X] T031 [US3] 运行 US3 测试并记录 RoPE/mask/隔离证据到 `audit/reports/v18/report/PROGRESS.md`

**Checkpoint**: 位置/掩码/信息边界自动证明。

## Phase 6: User Story 4 - 密集 token 融合与 GQA 生产契约 (Priority: P1)

**Goal**: `d_model=256`、16Q/4KV、`dense_slot_dim=32`、`dense_fusion_dim=512`、≤6.0M 参数。

**Independent Test**: 槽位敏感性/内部顺序/padding 与零值/梯度/尺度/无碰撞/元数据消融；参数分项统计。

### Tests for User Story 4

- [X] T032 [P] [US4] 新增 `riichi_ppo_v1/tests/unit/test_v18_dense_embedding.py`（槽位独变改变 embedding、交换摘要槽位改变、padding 严格零、梯度到达所有槽位、幅度不爆炸、随机组合无完全相等 embedding）
- [X] T033 [P] [US4] 更新参数契约测试 `riichi_ppo_v1/tests/unit/test_v18_parameter_count.py` 与 `integration/test_v18_validation.py`：≤6.0M、分项报告、无 MHA/Q 分支
- [X] T034 [P] [US4] 新增 ModelConfig strict 校验测试（d_model/heads/head_dim/ffn/layers/context/rope/dense 维度）

### Implementation for User Story 4

- [X] T035 [US4] 实现 `riichi_ppo_v1/model/dense_embedding.py`：`DenseSlotFusion`（类别专属投影→共享 RMSNorm+gated MLP→256）与 `SimpleConcatEmbedding`，并接入 `architecture.py`
- [X] T036 [US4] 更新 `riichi_ppo_v1/model/parameter_count.py`：统一报告 embedding/shared/actor/critic/head 分项，上限 6.0M
- [X] T037 [US4] 更新 `riichi_ppo_v1/configs/v18_sft.yaml`：自包含（d_model/16Q/4KV/dense_slot_dim=32/dense_fusion_dim=512/层数/FFN/rope_base/context_tokens=256/新 policy_head_type）
- [X] T038 [US4] 运行 US4 测试并记录参数/上下文统计证据到 `audit/reports/v18/report/PROGRESS.md`

**Checkpoint**: 生产拓扑与参数统计锁定。

## Phase 7: User Story 5 - Actor-only SFT 全链路 (Priority: P1)

**Goal**: replay→precompute→shard→collator→Actor-only SFT 一致且可保存/加载。

**Independent Test**: 小规模端到端集成；Critic/value 冻结无梯度；manifest/hash 错误 fail closed；
同一局面多入口张量级一致。

### Tests for User Story 5

- [X] T039 [P] [US5] 新增 `riichi_ppo_v1/tests/integration/test_v18_sft_lifecycle.py`：临时目录内 replay→precompute→shard→`iter_precomputed_samples`→collate→训练 2 步→保存/加载 → 断言 loss 有限、critic 参数零梯度、strict load
- [X] T040 [P] [US5] 更新 `riichi_ppo_v1/tests/unit/test_v18_actor_sft.py`（新 forward 键名、critic 冻结）
- [X] T041 [P] [US5] 更新 `riichi_ppo_v1/tests/unit/test_v18_sft_contract.py`（新 contract hash、manifest 校验 fail closed）
- [X] T042 [P] [US5] 新增多入口一致性测试：同一 fixture 分别经 `current_state.encode_batch`、precompute shard 读取、collator 得到相同 actor 行（`tests/integration/test_v18_encoding_bridge.py`）

### Implementation for User Story 5

- [X] T043 [US5] 重写 `riichi_ppo_v1/sft/data.py`：`EncodedSample` 新字段 + `encode_kyoku` 使用 `current_state.py` 装配（含 separator/query 规范排序），去掉 history/snapshot
- [X] T044 [US5] 重写 `riichi_ppo_v1/sft/precompute.py`：写/读 `actor_offsets/actor_factors/actor_numeric/query_offsets/action_offsets/query_rows/action_ids/legal/actions/身份字段`，manifest 新键
- [X] T045 [US5] 更新 `riichi_ppo_v1/sft/trainer.py`：`collate_samples` 新张量、`_forward_actor` 新键、token 长度与溢出检查、指标
- [X] T046 [US5] 更新 `riichi_ppo_v1/sft/actor_bc.py`（新参数根）与 `riichi_ppo_v1/sft/contract.py`/`checkpoint.py`（契约 payload/版本）
- [X] T047 [US5] 运行 US5 测试并记录生命周期证据到 `audit/reports/v18/report/PROGRESS.md`

**Checkpoint**: SFT 全链路端到端通过。

## Phase 8: User Story 6 - 清理、文档与审计收敛 (Priority: P2)

**Goal**: 活跃路径旧引用清理；文档/配置/审计同步；PPO 待迁移盘点。

- [X] T048 [P] 全仓 `rg` 检查并删除 `riichi_ppo_v1/model/snapshot.py` 与任何零引用旧 history/snapshot 适配文件；若模块有独立职责（如 `action_jsons`/241 解码）保留在 `bridge.py`/`action_groups.py` 并更新注释
- [X] T049 [P] 更新 `riichi_ppo_v1/tools/v18_token_statistics.py`（actor_offsets + segment 贡献统计）与 `riichi_ppo_v1/tools/validate.py`（新参数契约）
- [X] T050 [P] 重写 `riichi_ppo_v1/docs/v18_input_protocol.md`，更新 `riichi_ppo_v1/docs/v18_sft.md`、`riichi_ppo_v1/docs/KyokuEventTupleProtocol.md`（事件仅同步、不作模型输入）
- [X] T051 [P] 更新根 `README.md`、`riichi_ppo_v1/README.md`、`docs/directory-responsibilities.md`、`AGENTS.md` 中 V18 描述；PPO 相关文档仅加「待迁移、当前不适用新 V18 输入」标记
- [X] T052 [P] 在 `audit/reports/v18/report/PROGRESS.md` 记录：改动范围、schema hash、参数量、token 统计（mean/p50/p95/p99/max + segment 贡献）、编码性能、SFT 性能、测试命令与结果、PPO 待迁移清单
- [X] T053 运行全仓一致性复核（活跃路径零旧引用、PPO 分类完整、无未清理临时产物）并输出结论

**Checkpoint**: 全仓单一活跃契约、文档一致、审计完整。

## Phase 9: Performance & Convergence

- [X] T054 [P] CPU/Rust/PyO3 编码吞吐与分段耗时测量（真实 replay 抽样，N≥100 决策，3 轮取后两轮）
- [X] T055 [P] GPU Actor-only SFT 前向/反向、显存、tokens/s（`CUDA_DEVICE=0,1`，`learner_gpus=2`；文档化命令与结果；结束后 `ray stop --force` 与临时日志清理）
- [X] T056 运行 `speckit-converge` 口径的代码-任务复核，追加缺口任务并完成，直到收敛
- [X] T057 冒烟测试临时目录/日志/结果全部清理并确认 `git status` 无新临时产物

## Dependency Graph

```
T001-T003 (Setup)
  → T004-T010 (Foundational: schema + Rust encoder + validation)
    → T011-T017 (US1 公共快照)
      → T018-T024 (US2 Analysis/Query)
        → T025-T031 (US3 RoPE/mask/边界)
          → T032-T038 (US4 融合/参数)
            → T039-T047 (US5 SFT 链路)
              → T048-T053 (US6 清理/文档/审计)
                → T054-T057 (性能/收敛)
```

## Parallel Execution Examples

- T005/T006/T009（Rust 编码器与单测）可与 T004/T007/T010（Python schema/校验/fixture）并行，
  文件边界不重叠；T011/T012/T013 可并行；T018/T019/T020 可并行；T025/T026/T027 可并行；
  T032/T033/T034 可并行；T039/T040/T041/T042 可并行；T048/T049/T050/T051/T052 可并行。
- 所有 [P] 任务由不同 sub-agent 执行时必须获得明确文件所有权；`encoding_protocol.py`、
  `architecture.py`、`dense_embedding.py`、`current_state_encoding.rs`、`sft/data.py`、
  `sft/precompute.py`、`sft/trainer.py`、`bridge.py`、`semantic_validation.py` 与
  `lib.rs` 属于主 agent 串行整合的“单文件单所有权”文件，不并行修改同一文件。

## Implementation Strategy

- 先协议/schema/Rust 编码器（Phase 2/3），再模型（Phase 4/5/6），再 SFT（Phase 7），
  最后清理/文档/性能（Phase 8/9）。
- 每个 Phase 完成即运行对应测试，失败修复根因并同步任务状态。
- 主 agent 负责契约、整合、破坏性删除与最终验证；路线明确的机械工作优先委派 sub-agent。
