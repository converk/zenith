---
description: "Task list for 代码与目录治理"
---

# Tasks: 代码与目录治理(Code & Directory Governance)

**Input**: `specs/001-code-dir-governance/` 下的 spec.md、plan.md、research.md、
data-model.md、contracts/、quickstart.md

**Tests**: 本 feature 要求"每主题一个 commit、每个 commit 测试通过",且宪法要求
新增/修改代码附带测试;每个主题 phase 内包含测试改写与验证任务。

**Organization**: 按 spec 用户故事分组;执行顺序遵循 research.md Decision 11
(先删 v11 再搬迁),每个主题以独立 commit 收尾。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行(不同文件、无依赖)
- **[Story]**: 用户故事标签(US1–US6)
- 描述包含精确文件路径

---

## Phase 1: Setup(基线与环境)

**Purpose**: 记录治理前基线,确认运行环境

- [X] T001 确认 Conda 环境 `Mahjong-AI` 可用、`riichi`/`riichienv` 扩展可导入:
  运行 `conda run -n Mahjong-AI python -c "import riichi, riichienv; print('ok')"`
- [X] T002 记录治理前测试基线:分别运行
  `conda run -n Mahjong-AI python -m pytest riichi_ppo_v1/tests/unit riichi_ppo_v1/tests/protocol -q`、
  `conda run -n Mahjong-AI python -m pytest riichi_lab_bot/tests -q`、
  `conda run -n Mahjong-AI python -m pytest RiichiEnv/tests -q`,把既有失败
  (如 `tests/unit/test_head_to_head.py` 引用已删模块)记入本 feature 的
  工作笔记,不得绕过
- [X] T003 确认 git 状态:分支 `sft`,`AGENTS.md` 的未提交修改属于本 feature
  (research.md Decision 9);`checkpoints/`、`datasets/` 由 .gitignore 排除,提交
  不得触碰

---

## Phase 2: Foundational(阻塞性前提)

**Purpose**: 建立删除/搬迁的安全前提(零引用检查与测试命令约定)

- [X] T004 [P] 全仓库 `rg -n "legacy/v11|legacy\.v11|V11PolicyAdapter|build_legacy_v11|legacy_fixed"`(排除 `.pyc`、`target/`、产物目录),产出引用清单作为 US2 删除依据
- [X] T005 [P] 全仓库 `rg -n "sft\.(head_to_head_1v3|head_to_head_1v3_shards|policy_adapter|action_groups)|riichi_ppo_v1\.tests\.validate|exp/"`,产出搬迁/幽灵引用清单作为 US1/US5 依据

**Checkpoint**: 引用清单与 research.md Decision 1/3/9 一致后进入用户故事

---

## Phase 3: User Story 2 - 移除 v11 checkpoint 兼容(Priority: P1)→ Commit 1

**Goal**: 彻底移除 v11 适配器、`legacy_fixed` 模型头、`build_legacy_v11` 与相关
测试;checkpoint/数据集不动

**Independent Test**: 删除后 `rg` 对 `legacy/v11|legacy_fixed|build_legacy_v11`
零命中;全量单元测试通过

### Implementation for US2

- [X] T006 [US2] 删除 `riichi_ppo_v1/legacy/v11/`(adapter.py、contract.py、
  encoder.py、model.py、__init__.py)与 `riichi_ppo_v1/legacy/__init__.py`,
  随后删除空目录 `riichi_ppo_v1/legacy/`
- [X] T007 [US2] 改写 `riichi_ppo_v1/model/architecture.py`:默认
  `policy_head_type="isolated_action_query"`,校验只允许
  `isolated_action_query`,删除第 290/362/412/500 行附近的 `legacy_fixed` 分支
  与 `extra` 逻辑,docstring 中 v11/legacy 描述改为中文并说明现行头
- [X] T008 [US2] 改写 `riichi_ppo_v1/training/rewards/decision.py`:删除
  `DecisionAnalysisBatch.build_legacy_v11` 与 `_legacy_v11` 参数及其分支
- [X] T009 [US2] 改写 `riichi_ppo_v1/sft/policy_adapter.py`:删除
  `load_policy_adapter` 的 V11 回退分支(约第 145–148 行),错误信息
  "no supported v11/v13 evaluation contract" 改为 v13 表述
- [X] T010 [US2] 删除 `riichi_ppo_v1/tests/integration/test_v11_policy_adapter.py`
- [X] T011 [US2] 改写 `riichi_ppo_v1/tests/unit/test_decision_analysis.py`:删除
  第 391 行附近 `build_legacy_v11` 用例及无关的 `_legacy_v11` 断言
- [X] T012 [US2] 改写 `riichi_ppo_v1/tests/unit/test_sft_contract.py`:
  `test_v13_training_modules_do_not_import_legacy_v11` 改为断言
  `riichi_ppo_v1/legacy` 目录不存在、全训练模块无 legacy import
- [X] T013 [US2] 改写 `riichi_ppo_v1/pyproject.toml`:packages 移除
  `riichi_ppo_v1.legacy` 与 `riichi_ppo_v1.legacy.v11`
- [X] T014 [US2] 改写 `riichi_ppo_v1/docs/v13_sft.md`:删除 legacy/v11 描述,
  说明仅支持 v13 现行契约
- [X] T015 [US2] 运行
  `conda run -n Mahjong-AI python -m pytest riichi_ppo_v1/tests/unit riichi_ppo_v1/tests/protocol -q`
  与 `rg -n "legacy/v11|legacy_fixed|build_legacy_v11"`,通过后提交 Commit 1
  (主题: 移除 v11 checkpoint 兼容)

---

## Phase 4: User Story 1 - 目录职责归位(Priority: P1)→ Commit 2

**Goal**: 1v3 评测与 `riichi-ppo-validate` 入口搬到职责目录,引用与打包同步

**Independent Test**: `riichi_ppo_v1/evaluation/` 含 1v3 与策略适配器;
`sft/` 无 1v3 评测代码;`tests/` 无生产入口;全量单元测试通过

### Implementation for US1

- [X] T016 [US1] 新建 `riichi_ppo_v1/evaluation/__init__.py`(中文模块说明),把
  `riichi_ppo_v1/sft/policy_adapter.py`、
  `riichi_ppo_v1/sft/head_to_head_1v3.py`、
  `riichi_ppo_v1/sft/head_to_head_1v3_shards.py` 移入并修正相对 import
  (原 `..model`/`..training` 层级不变,原 `.action_groups`/`.policy_adapter` 改
  `..model.action_groups`/`.` 同级引用)
- [X] T017 [US1] 把 `riichi_ppo_v1/sft/action_groups.py` 移为
  `riichi_ppo_v1/model/action_groups.py`;更新
  `riichi_ppo_v1/sft/train.py`、`riichi_ppo_v1/training/metrics.py`、
  `riichi_ppo_v1/evaluation/head_to_head_1v3.py` 的 import
- [X] T018 [US1] 把 `riichi_ppo_v1/tests/validate.py` 移为
  `riichi_ppo_v1/tools/validate.py`;`riichi_ppo_v1/pyproject.toml` 入口改为
  `riichi_ppo_v1.tools.validate:main`,packages 移除 `riichi_ppo_v1.tests`、
  新增 `riichi_ppo_v1.evaluation`
- [X] T019 [US1] 更新引用:`riichi_ppo_v1/training/train.py` 改为
  `..evaluation.head_to_head_1v3_shards`;`riichi_ppo_v1/sft/heuristic_evaluation.py`
  的 `.policy_adapter` 改 `..evaluation.policy_adapter`(启发式评测本身按
  research.md Decision 1 留在 sft,training 侧引用在 Commit 4 随旧机制删除);
  `riichi_lab_bot/src/riichi_lab_bot/policy.py` 改
  `riichi_ppo_v1.evaluation.policy_adapter`
- [X] T020 [US1] 更新测试引用:`riichi_ppo_v1/tests/unit/test_head_to_head_1v3_shards.py`
  改 `riichi_ppo_v1.evaluation.head_to_head_1v3_shards`;其余测试中
  `sft.policy_adapter`/`sft.head_to_head_1v3*` 引用同步
- [X] T021 [US1] 运行
  `conda run -n Mahjong-AI python -m pytest riichi_ppo_v1/tests/unit riichi_ppo_v1/tests/protocol riichi_lab_bot/tests -q`
  与 `conda run -n Mahjong-AI riichi-ppo-validate --help`,通过后提交 Commit 2
  (主题: 1v3 评测与验证入口搬迁)

---

## Phase 5: User Story 3 - 配置归一(Priority: P1)→ Commit 3

**Goal**: 版本配置自包含,`v14_ppo_resume.yaml` 展平,加载器移除 full-config
overlay

**Independent Test**: `v15_ppo.yaml`/`v14_ppo_resume.yaml` 单独加载即完整;
加载器无分组叠加参数;配置测试改写后通过

### Implementation for US3

- [X] T022 [US3] 改写 `riichi_ppo_v1/training/train.py` 的 `load_config`:
  `path` 非空时直接加载自包含文件(不叠加打包默认),`path` 为空时合并打包
  `training`+`monitoring`;删除 `model_path`/`environment_path`/`training_path`
  参数、`_parser` 中 `--model-config`/`--environment-config`/`--training-config`
  三个标志,更新 `main`/`smoke_main` 调用
- [X] T023 [US3] 补全 `riichi_ppo_v1/configs/v14_ppo.yaml` 为自包含(补
  model_size/context_tokens/policy_head_type/num_workers/envs_per_worker/
  env_step_threads/inference_*/kyokus_per_worker/PPO 完整参数,值取现行
  `training.yaml` 默认与 v14 既有覆盖的并集,删除已废弃键)
- [X] T024 [US3] 展平 `riichi_ppo_v1/configs/v14_ppo_resume.yaml`:内容为
  `v14_ppo.yaml` 全量 + `resume: checkpoints/train_riichi_ppo_v14/checkpoint_00600.pt`
  + `init_model: null`,不再只有两行
- [X] T025 [US3] 补全 `riichi_ppo_v1/configs/v15_ppo.yaml` 为自包含(同 T023
  规则,保留 v15 特有 sft_kl_coef_middle/fraction 与 eval1v3 块)
- [X] T026 [US3] 改写 `riichi_ppo_v1/configs/training.yaml`:`checkpoint_dir`
  改中性 `checkpoints/train_riichi_current`(contracts 见
  cli-config-contract.md)
- [X] T027 [US3] 改写 `riichi_ppo_v1/tests/unit/test_config_loading.py`:
  `test_group_overrides_precede_the_legacy_full_config_overlay` 删除,改为
  "自包含配置不叠加打包默认";`test_v15_overlay_resolves...` 改为
  "v15 配置自包含";新增 `v14_ppo_resume.yaml` 展平断言
- [X] T028 [US3] 改写 `riichi_ppo_v1/tests/unit/test_cleanup_contract.py` 中
  `checkpoint_dir` 断言为 `checkpoints/train_riichi_current`
- [X] T029 [US3] 运行 quickstart.md 第 3 节配置契约脚本与
  `conda run -n Mahjong-AI python -m pytest riichi_ppo_v1/tests/unit -q`,
  通过后提交 Commit 3(主题: 配置归一与移除 overlay)

---

## Phase 6: 移除旧 evaluation_* 评测机制(宪法 IV,跨 US3/US4)→ Commit 4

**Goal**: 删除 PPO 旧启发式评测双轨,1v3 为唯一评测机制

**Independent Test**: `training/evaluation.py` 不存在;`rg "evaluation_interval_updates"`
在代码与打包默认配置中零命中;单测通过

### Implementation

- [X] T030 删除 `riichi_ppo_v1/training/evaluation.py`
- [X] T031 改写 `riichi_ppo_v1/training/inference.py`:删除
  `evaluate_heuristics` 方法及其内部 `sft.heuristic_evaluation` /
  `training.evaluation` import
- [X] T032 改写 `riichi_ppo_v1/training/train.py`:删除
  `run_evaluation` 定义、调用与 `training.evaluation` import;`smoke_main` 删除
  `"evaluation_enabled": False`
- [X] T033 改写 `riichi_ppo_v1/configs/monitoring.yaml`:删除
  `evaluation_enabled`/`evaluation_interval_updates`/`evaluation_hanchan_count`/
  `evaluation_parallel_hanchan_count`/`evaluation_seed_base`/
  `evaluation_game_mode`/`evaluation_max_steps`/`evaluation_cache_capacity`/
  `evaluation_jsonl` 键(其余性能/监控键保留)
- [X] T034 删除 `riichi_ppo_v1/tests/unit/test_ppo_evaluation.py` 与
  `riichi_ppo_v1/tests/unit/test_head_to_head.py`(后者引用 HEAD 已删的
  `sft/head_to_head.py`)
- [X] T035 改写 `riichi_ppo_v1/tests/unit/test_config_loading.py` 与
  `test_cleanup_contract.py`:删除对 `evaluation_enabled` 等旧键的断言,
  新增"打包默认不含 evaluation_* 键"断言
- [X] T036 运行
  `conda run -n Mahjong-AI python -m pytest riichi_ppo_v1/tests/unit riichi_ppo_v1/tests/protocol -q`
  与 `rg -n "evaluation_interval_updates|run_evaluation|evaluate_heuristics"`,
  通过后提交 Commit 4(主题: 移除旧 evaluation_* 评测机制)

---

## Phase 7: User Story 4 - 消除硬依赖与收敛领域常量(Priority: P2)→ Commit 5

**Goal**: 三组件硬编码版本/路径/种子参数化;241/34/136 单一来源

**Independent Test**: 代码默认值无历史版本/日期/实验目录;`rg "= 241|= 34|= 136"`
仅剩单一定义;全量测试(含 Rust)通过

### Implementation for US4

- [X] T037 [US4] 改写 `riichi_ppo_v1/model/schema.py`:新增 `NUM_ACTIONS = 241`、
  `TILE_KINDS = 34`;`riichi_ppo_v1/model/bridge.py` 与
  `riichi_ppo_v1/model/architecture.py` 删除各自 `NUM_ACTIONS = 241` 定义并
  `from .schema import NUM_ACTIONS`(bridge 保留对外再导出名)
- [X] T038 [US4] 替换 `riichi_ppo_v1/sft/`(audit.py、train.py、precompute.py、
  contract.py)、`riichi_ppo_v1/model/`(actor_features.py、dora.py、validation.py)
  与相关测试中代表动作空间/牌类的 `241`/`34` 字面量为
  `NUM_ACTIONS`/`TILE_KINDS`
- [X] T039 [US4] 改写 `riichi_ppo_v1/training/train.py` 的 1v3 参数化:
  `_progress_md_path` 删除 `audit/reports/v14_ppo_20260812/eval` 历史默认;
  `run_1v3_evaluation` 开启时 `eval1v3_model_b`/`eval1v3_seed_base`/
  `eval1v3_output_dir` 缺失即抛明确错误,`eval1v3_devices` 缺省 `("0","1")`;
  机制常量(10/160/30)保留单源引用 `REQUIRED_1V3_PROCESSES`
- [X] T040 [US4] 改写 `riichi_ppo_v1/sft/train.py`:固定节奏常量
  (3000 steps、96 hanchan)定义为单一命名常量并替换
  `heuristic_evaluation_interval_steps: 7000` 与
  `heuristic_evaluation_final_hanchan_count` 默认 128 的旧值;删除
  DEFAULT_CONFIG 中 `heuristic_evaluation_seed_base=20260717`(改由配置/CLI 提供,
  缺省 0);`riichi_ppo_v1/configs/sft.yaml` 删除与代码重复的固定节奏键
- [X] T041 [US4] 改写 `riichi_lab_bot/src/riichi_lab_bot/cli.py`:
  `--checkpoint` 默认只取 `RIICHI_CHECKPOINT`,两者都缺时
  `parser.error`,删除 `train_riichi_ppo_v14/checkpoint_00510.pt` 硬编码;
  `local --seed` 默认 0
- [X] T042 [US4] 改写 `riichi_lab_bot/src/riichi_lab_bot/policy.py`:
  字面量 `13` 替换为 `TOKEN_SCHEMA_VERSION` 引用
- [X] T043 [US4] 改写 `riichi_ppo_v1/sft/precompute.py`: `--output` 改为必填
  (删除锁定历史编码名的默认值);改写 `riichi_ppo_v1/tools/validate.py`:
  `--seed` 默认 0
- [X] T044 [US4] 改写 RiichiEnv Rust:`riichienv-core/src/observation/sequence_features.rs`
  等处的 `136`/`34` 字面量引用 `TILES_4P`/命名常量;
  `riichienv-state-machine/src/MjaiKyokuStateMachine/protocol.rs` 与
  `analysis.rs` 的 `241` 引用 `NUM_ACTIONS`;
  `RiichiEnv/src/riichienv/convert.py` 新增 `TID_COUNT = 136` 并替换两处字面量
- [X] T045 [US4] 更新受影响测试(如 `riichi_ppo_v1/tests/unit/test_cleanup_contract.py`
  的 config 断言、`riichi_lab_bot/tests` 中默认 checkpoint/seed 假设)
- [X] T046 [US4] 运行
  `conda run -n Mahjong-AI python -m pytest riichi_ppo_v1/tests riichi_lab_bot/tests RiichiEnv/tests -q`、
  `cd RiichiEnv/riichienv-core && cargo test`、
  `cd RiichiEnv/riichienv-state-machine && cargo test`,
  通过后提交 Commit 5(主题: 硬依赖参数化与领域常量收敛)

---

## Phase 8: User Story 5 - 清理幽灵引用(Priority: P2)→ Commit 6

**Goal**: 删除 `exp/` 等幽灵引用,修正 AGENTS.md/README/docs 路径

**Independent Test**: `rg` 对 `exp/`、已删模块、历史错误路径零命中;
文档路径与仓库实际一致

### Implementation for US5

- [X] T047 [US5] 改写 `riichi_ppo_v1/training/learner.py`(第 247/391 行注释)与
  `riichi_ppo_v1/model/architecture.py`(第 499 行注释):删除 `exp/training`
  幽灵引用,改为中文说明现行实现
- [X] T048 [US5] 改写 `riichi_ppo_v1/README.md`:`CUDA_DEVICE=0,3` 改 0,1;
  性能命令 `--kyokus-per-worker 1` 改 16;`encoded_10pct_v2` 示例改为现行
  `datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16`;删除旧启发式评测
  (每 15 updates/96 hanchan)段落,改为 1v3 机制(10×160,每 30 updates)
- [X] T049 [US5] 改写 `riichi_ppo_v1/docs/v13_sft.md`:`encoded_40pct_v13` 改为
  实际目录 `encoded_40pct_v13_v16`;`kyokus_per_worker=1` 改 16;legacy/v11 描述
  已随 T014 处理,复核无残留
- [X] T050 [US5] 改写 `riichi_lab_bot/README.md`:`CUDA_DEVICE=2,3` 示例改为
  0,1(映射注释见 AGENTS.md);默认 checkpoint 描述改为"必须显式
  `--checkpoint` 或 `RIICHI_CHECKPOINT`";删除
  `riichi_lab_bot/tools/verify_candidate_token_drift.py`(引用已删
  `sft/head_to_head.py`,零运行引用)并清理
  `riichi_lab_bot/src/riichi_lab_bot/bridge.py` 第 18 行对它的注释引用
- [X] T051 [US5] 提交 `AGENTS.md`(既有未提交修改:删 `exp/` 引用、
  `CUDA_DEVICE=0,1`、测试基线 `kyokus_per_worker=16`)纳入本主题
- [X] T052 [US5] 运行 `rg -n "exp/|encoded_10pct|CUDA_DEVICE=0,3|CUDA_DEVICE=2,3|train_riichi_ppo\b|sft\.head_to_head\b"`
  (排除产物目录)与三组件测试,通过后提交 Commit 6(主题: 清理幽灵引用并修正文档)

---

## Phase 9: User Story 6 - 命名、注释与交付卫生(Priority: P3)→ Commit 7

**Goal**: 冒烟清理、目录职责一句话清单、中文注释核查

**Independent Test**: 冒烟后无残留产物;`docs/directory-responsibilities.md`
覆盖三组件目录;新增/修改注释为中文

### Implementation for US6

- [X] T053 [US6] 改写 `riichi_ppo_v1/training/train.py` 的 `smoke_main`:
  运行结束(含 finally 路径)删除 `checkpoints/riichi_ppo_v1_smoke` 与本次冒烟
  产生的日志文件,清理失败只告警不抛错;`riichi_ppo_v1/tests/unit/test_cleanup_contract.py`
  新增"冒烟清理函数可删除自身产物"测试
- [X] T054 [US6] 新建 `docs/directory-responsibilities.md`:为
  `riichi_ppo_v1/{evaluation,model,sft,training,tools,configs,tests,docs}`、
  `riichi_lab_bot/{src,tests,tools}`、`RiichiEnv/{src,riichienv-core,
  riichienv-state-machine,riichienv-python,tests,docs,scripts}` 各写一句话职责
- [X] T055 [US6] 复查本 feature 全部新增/修改文件的注释为中文(含 docstring 中
  的英文说明改为中文),文件名自描述
- [X] T056 [US6] 运行 quickstart.md 全部快速验证与三组件测试,通过后提交
  Commit 7(主题: 目录职责清单与冒烟清理)

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: 三件套一致性与最终验收

- [X] T057 交叉核对 spec.md/plan.md/tasks.md:每个用户故事在 plan 有设计、在
  tasks 有任务;research Decision 1–11 均有对应任务;修正任何不一致
- [X] T058 运行最终验收:全仓库测试(Python 三组件 + Rust 两 crate);
  `rg` 对 `legacy/v11|legacy_fixed|build_legacy_v11|exp/` 零命中;quickstart.md
  第 3/4/5/7 节全过;确认 `specs/001-code-dir-governance` 三件套自洽
- [X] T059 生成完成报告:7 个主题 commit 列表、测试结果、目录职责清单路径、
  遗留说明(如数据集删除按用户约束未执行)

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup(Phase 1)→ Foundational(Phase 2)→ US2(Phase 3)→ US1(Phase 4)→
  US3(Phase 5)→ 旧机制移除(Phase 6)→ US4(Phase 7)→ US5(Phase 8)→
  US6(Phase 9)→ Polish(Phase 10)
- 顺序依据 research.md Decision 11:先删 v11 再搬 policy_adapter,避免搬迁
  commit 期间 `legacy/v11/adapter.py` 的 `..sft.policy_adapter` 相对 import 失效

### User Story Dependencies

- US2 与 US1 必须按序(US2 先);US3 依赖 US1 的 `evaluation/` 布局(T022 不直接
  依赖,但 Commit 3 在 Commit 2 之后);旧机制移除依赖 US3 的配置测试改写;
  US4 依赖旧机制移除后 `training/train.py` 的稳定结构;US5/US6 相对独立但按
  commit 顺序收尾

### Parallel Opportunities

- Phase 2 的 T004/T005 可并行(只读扫描)
- 同一主题内修改不同文件的任务可并行(T006–T014 中 T007/T008/T009 分属不同
  文件),但删除类任务必须先完成 T004/T005 引用核对

---

## Implementation Strategy

### 增量交付(每主题一个 commit)

1. Commit 1(US2):v11 兼容移除,测试通过
2. Commit 2(US1):1v3/验证入口搬迁,测试通过
3. Commit 3(US3):配置归一,配置测试通过
4. Commit 4:旧 evaluation_* 机制移除,测试通过
5. Commit 5(US4):硬依赖参数化 + 领域常量,Python/Rust 测试通过
6. Commit 6(US5):幽灵引用与文档,`rg` 零命中
7. Commit 7(US6):目录职责清单与冒烟清理

### 破坏性操作守则

- 删除文件前先 `rg` 全仓库零引用;checkpoint 与数据集一律不删;
- 任何超出授权范围的破坏性操作(如删除 checkpoint/数据集、改写 git 历史)先停下
  向用户确认。

---

## Notes

- 本 feature 只删代码/测试/文档/配置;`checkpoints/`、`datasets/` 内容不修改。
- 宪法 III 提及的 `tenhou-to-mjai` 数据集删除决策不在本 feature 执行,遵循用户
  "数据集一律不删"约束,最终报告中说明。
- 性能/训练基线(3 轮、首轮预热)在 T058 验收时按宪法 V 执行;日常主题 commit
  使用快速单元/协议测试,训练冒烟按需在 GPU 可用时补充。
