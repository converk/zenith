# Tasks: 产物存储与评测机制固化

**Input**: Design documents from `/specs/002-artifact-storage-eval/`

**Prerequisites**: [plan.md](./plan.md)(必需)、[spec.md](./spec.md)(必需,用户故事)、
[research.md](./research.md)、[data-model.md](./data-model.md)、
[contracts/](./contracts/)、[quickstart.md](./quickstart.md)

**Tests**: 本 feature 按宪法"新增或修改代码必须附带对应测试"生成测试任务;测试先写、
先失败,再实现。

**Organization**: 任务按用户故事(US1–US7)组织;本仓库单实现者顺序执行,
`[P]` 表示文件/目录互不冲突、可并行。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 不同文件/目录、无未完成依赖,可并行
- **[Story]**: 所属用户故事(如 US1)
- 描述必须含精确文件路径

## 关键执行顺序约束(不可违反)

1. **T008(删除 `logs/` 现有内容)必须先于 US2 归位(T011–T013)**:US2 会把历史
   audit 目录里的运行日志移入 `logs/<版本号>/`;若先归位再删 `logs/` 存量,会误删
   刚移入的历史日志。
2. T008 与数据集删除(如触发)是破坏性操作,执行前必须停下向用户确认。
3. checkpoint 目录只 `mv` 归档,禁止删除;T018 改名后若 T020 测试无法通过,按
   spec 回退为存量例外保留 `train_riichi_v13_sft` 原名。

## Phase 1: Setup(共享基础)

**Purpose**: 环境与基线确认

- [X] T001 [P] 确认 conda 环境 `Mahjong-AI` 可用并跑基线子集:
  `python -m pytest riichi_ppo_v1/tests/unit/test_cleanup_contract.py
  riichi_lab_bot/tests -q`(记录通过状态)
- [X] T002 [P] 生成移动前清单快照到 `/tmp/artifact-layout-before.txt`:
  `find logs audit/reports checkpoints datasets -mindepth 1 | sort`,供 T014 对账
  (临时文件不提交)

## Phase 2: Foundational(阻塞前置)

**Purpose**: 所有用户故事共享的契约测试骨架

- [X] T003 新建 `riichi_ppo_v1/tests/unit/test_artifact_conventions.py` 骨架:
  仓库根定位、YAML 读取、`git check-ignore` 探测辅助(中文注释);本文件后续各
  故事逐步扩展断言

**Checkpoint**: 骨架就绪,各用户故事可开始

---

## Phase 3: User Story 1 - 日志规范(Priority: P1)🎯 MVP

**Goal**: `logs/` 存量清除,三个组件日志写入点统一到 `logs/<版本号>/`

**Independent Test**: `logs/` 根目录无文件;代码默认值与运行脚本/文档不再写
`logs/` 根目录或 audit 目录;最小冒烟产物只落在 `logs/<版本号>/`

### Tests for User Story 1

- [X] T004 [US1] 在 `riichi_ppo_v1/tests/unit/test_artifact_conventions.py` 增加:
  解析 `riichi_ppo_v1/sft/audit.py` 的 `--report` 默认值为 `None`;断言
  `riichi_ppo_v1/training/train.py` 源码含 `RAY_LOG_TO_STDERR` 注入(先写,预期失败)

### Implementation for User Story 1

- [X] T005 [P] [US1] `riichi_ppo_v1/sft/audit.py`:`--report` 默认改为 `None`,
  `main()` 仅在显式提供时调用 `write_audit_report`;help 注明合法路径为
  `logs/<版本号>/`(中文注释)
- [X] T006 [P] [US1] `riichi_ppo_v1/training/train.py`:在 `ray.init` 前
  `os.environ.setdefault("RAY_LOG_TO_STDERR", "1")`,注释说明与运行脚本重定向配合
  (中文注释)
- [X] T007 [US1] 运行 `python -m pytest riichi_ppo_v1/tests/unit/
  test_artifact_conventions.py -q` 至通过,并提交主题 1「日志代码写入点统一」
- [X] T008 [US1] **停止并确认**:向用户报告 `logs/` 现有内容规模(约 127MB)与清单,
  经用户确认后删除 `logs/` 现有全部内容(不归档);删除后确认 `logs/` 根目录为空

**Checkpoint**: 日志写入点规范落地,`logs/` 存量清除

---

## Phase 4: User Story 2 - audit 报告规范(Priority: P1)

**Goal**: `audit/reports/<版本号>/` 固定类型目录;存量按版本归位;design/report/
scripts 进版本控制

**Independent Test**: `audit/reports/` 顶层为 `v13 v14 v15`,各版本目录只有
design/report/eval/scripts;文件总数与移动前一致;design/report/scripts 被 git
跟踪、eval 被忽略

### Tests for User Story 2

- [X] T009 [US2] 在 `riichi_ppo_v1/tests/unit/test_artifact_conventions.py` 增加:
  `git check-ignore` 探测 `audit/reports/v15/design|report|scripts` 下文件
  NOT ignored、`audit/reports/v15/eval` 下文件 IGNORED;断言
  `audit/reports/{v13,v14,v15}` 各仅含四个固定类型子目录(先写,预期失败)

### Implementation for User Story 2

- [X] T010 [US2] 根 `.gitignore`:把 `audit/` 一行替换为方案 A 放行规则
  (`audit/*`、`!audit/reports/`、`audit/reports/*`、`!audit/reports/*/`、
  `audit/reports/*/*`、`!audit/reports/*/design|report|scripts/`),中文注释;
  用 `git check-ignore` 手工验证三类放行、eval 忽略
- [X] T011 [P] [US2] 归位 `audit/reports/v13_sft_20260802` → `audit/reports/v13`:
  `REPORT.md`→`report/`,`scripts/*.py`→`scripts/`,审计输出 `*.json`→`eval/`,
  `environment.json`/`git_status.txt`/`commands.txt`→`report/`(只移动)
- [X] T012 [P] [US2] 归位 `audit/reports/v14_ppo_20260812` → `audit/reports/v14`:
  `PROGRESS.md`→`report/`,`*.py|*.sh`→`scripts/`,`eval/`、`eval_holdout2k/`、
  `eval_510_vs_120_2400/`→`eval/`,`*.log` 与 `ranked_*.jsonl`→`logs/v14/`,
  `smoke/`→`logs/v14/smoke/`;脚本内 `LOG_FILE`/输出路径改新规范,`run_sft_baseline.sh`
  的 `riichi_ppo_v1.sft.head_to_head_1v3_shards` 改
  `riichi_ppo_v1.evaluation.head_to_head_1v3_shards`(只移动+路径修正)
- [X] T013 [P] [US2] 归位 `audit/reports/v15_ppo_20260814` → `audit/reports/v15`:
  `V15 Q-Boosting 简要设计.md`/`V15 版本修正提纲.md`/`V15 执行计划.md`→`design/`,
  `V15 实现与验证记录.md`→`report/`,`run_*.py|*.sh`→`scripts/`,`eval/`、
  `v15_sft_1v3_2k/`、`v14_u510_reval_2k/`→`eval/`,`*.log`→`logs/v15/`;脚本内
  `LOG_FILE` 改为 `logs/v15/...`(只移动+路径修正)
- [X] T014 [US2] 对账:以 `/tmp/artifact-layout-before.txt` 为基准确认三个历史
  目录文件总数不变(零删除);`git add audit/reports/*/design audit/reports/*/report
  audit/reports/*/scripts`;跑 T009 测试至通过后提交主题 2「audit 固定类型规范」

**Checkpoint**: audit 布局与版本控制规范落地

---

## Phase 5: User Story 3 - checkpoint 规范化(Priority: P1)

**Goal**: `checkpoints/train_riichi_<版本号>` 布局;v14 与 v13_sft 归档改名,零删除,
引用全同步

**Independent Test**: `checkpoints/` 顶层为 `train_riichi_v13 v14 v15`;旧路径
全仓库零引用;全量测试通过;文件数与移动前一致

### Tests for User Story 3

- [X] T015 [US3] 更新既有测试断言为规范路径,并在
  `riichi_ppo_v1/tests/unit/test_artifact_conventions.py` 增加:各版本配置不含
  `train_riichi_ppo_v14`/`train_riichi_v13_sft` 旧路径,`sft/train.py` 的
  `DEFAULT_CONFIG["checkpoint_dir"] == "checkpoints/train_riichi_current"`
  (先写,预期失败)

### Implementation for User Story 3

- [X] T016 [US3] 归档移动 `checkpoints/train_riichi_ppo_v14` →
  `checkpoints/train_riichi_v14`(`mv`,禁止删除)
- [X] T017 [US3] 更新 v14 引用:`riichi_ppo_v1/configs/v14_ppo.yaml`、
  `riichi_ppo_v1/configs/v14_ppo_resume.yaml`、
  `riichi_ppo_v1/tests/unit/test_config_loading.py`、
  `riichi_lab_bot/tests/conftest.py`
- [X] T018 [US3] 归档移动 `checkpoints/train_riichi_v13_sft` →
  `checkpoints/train_riichi_v13/sft`(`mv`,禁止删除)
- [X] T019 [US3] 更新 v13_sft 引用:`riichi_ppo_v1/configs/sft.yaml`、
  `v14_ppo.yaml`、`v14_ppo_resume.yaml`、`v15_ppo.yaml`、
  `v15_sft_offense_warmup.yaml`、`v15_sft_actor_finetune.yaml`、
  `riichi_ppo_v1/sft/train.py`(DEFAULT 改 `checkpoints/train_riichi_current`)、
  `tests/unit/test_cleanup_contract.py`、`tests/unit/test_learner.py`、
  `tests/unit/test_sft_tensorboard.py`、
  `tests/integration/test_v13_sft_golden.py`
- [X] T020 [US3] 运行 `python -m pytest riichi_ppo_v1/tests -q` 与
  `riichi_lab_bot/tests`;若无法通过,回退 T018 为存量例外(保留
  `train_riichi_v13_sft` 原名,仅保留中性默认)
- [X] T021 [US3] 测试通过后提交主题 3「checkpoint 目录规范化」

**Checkpoint**: checkpoint 布局规范落地

---

## Phase 6: User Story 4 - 数据集清理(Priority: P1)

**Goal**: 废弃数据集不复活;默认路径参数化

**Independent Test**: 两个废弃目录不存在,现行两个数据集完好;`--archive-dir` 必填

### Tests for User Story 4

- [X] T022 [US4] 在 `riichi_ppo_v1/tests/unit/test_artifact_conventions.py` 增加:
  `riichi_ppo_v1/sft/prepare.py` 的 `--archive-dir` 无默认值(必填);断言
  `datasets/tenhou-to-mjai` 与 `datasets/..._80pct_v11` 不存在、现行两个数据集
  存在(先写,预期失败)

### Implementation for User Story 4

- [X] T023 [US4] `riichi_ppo_v1/sft/prepare.py`:`--archive-dir` 改为必填
  (删除默认值),help 注明废弃目录 `datasets/tenhou-to-mjai` 已被决策删除(中文注释)
- [X] T024 [US4] 核实两个废弃目录经 `find` 全盘确认已不存在并记录结论;若仍存在,
  **停止并二次确认**后删除(共约 20GB);跑 T022 测试至通过后提交主题 4
  「数据集默认路径参数化」

**Checkpoint**: 数据集规范落地

---

## Phase 7: User Story 5 - 评测机制统一(Priority: P1)

**Goal**: 1v3 常量单一来源;CLI 默认对齐;输出固定 `audit/reports/<版本号>/eval`

**Independent Test**: 机制常量(10/160/1600/30)仅一处定义;CLI 默认 1600/160/0;
版本配置 `eval1v3_output_dir` 为 `audit/reports/<版本号>/eval`;摘要与 PROGRESS
落点正确

### Tests for User Story 5

- [X] T025 [US5] 在 `riichi_ppo_v1/tests/unit/test_artifact_conventions.py` 增加:
  `evaluation/mechanism.py` 四常量值断言;`head_to_head_1v3.py` CLI 默认
  (1600/160/0)断言;`_progress_md_path` 推导 `audit/reports/v15/report/PROGRESS.md`
  断言;configs 的 `eval1v3_output_dir` 匹配 `audit/reports/v[0-9]+/eval`
  (先写,预期失败)

### Implementation for User Story 5

- [X] T026 [US5] 新建 `riichi_ppo_v1/evaluation/mechanism.py`:
  `REQUIRED_1V3_PROCESSES=10`、`DEFAULT_1V3_HANCHANS_PER_PROCESS=160`、
  `TOTAL_1V3_HANCHANS=1600`、`DEFAULT_1V3_INTERVAL_UPDATES=30`,中文注释注明
  修改须走宪法修订
- [X] T027 [P] [US5] `riichi_ppo_v1/evaluation/head_to_head_1v3_shards.py` 与
  `riichi_ppo_v1/training/train.py` 的常量导入改为 `evaluation/mechanism.py`
  (删除本地定义,保留再导出可选)
- [X] T028 [P] [US5] `riichi_ppo_v1/evaluation/head_to_head_1v3.py`:
  `--hanchans` 默认 1600、`--parallel-hanchans` 默认 160、`--seed-base` 默认 0;
  `evaluate_1v3` 函数默认同步,常量从 `mechanism.py` 导入
- [X] T029 [P] [US5] `riichi_ppo_v1/configs/v14_ppo.yaml`、
  `v14_ppo_resume.yaml`、`v15_ppo.yaml`:`eval1v3_output_dir` 改为
  `audit/reports/v14/eval`、`audit/reports/v15/eval`
- [X] T030 [US5] `riichi_ppo_v1/training/train.py`:`eval1v3.jsonl` 改追加到
  `output_dir/eval1v3.jsonl`;`_progress_md_path` 改为
  `output_dir.parent/report/PROGRESS.md`,`eval1v3_output_dir` 缺省时跳过进度
  写入;检查并同步既有测试对 `eval1v3.jsonl`/`PROGRESS.md` 的引用
- [X] T031 [US5] 跑 `test_artifact_conventions.py` 与训练/评测相关测试至通过,
  提交主题 5「1v3 输出固化与机制常量单一来源」

**Checkpoint**: 评测机制固化

---

## Phase 8: User Story 6 - SFT 节奏固定(Priority: P2)

**Goal**: 3000/96 只在 `sft.yaml`(及契约常量)定义,实验配置零复制

**Independent Test**: `sft.yaml` 数值==契约常量;实验配置不含任何节奏键

### Tests for User Story 6

- [X] T032 [US6] 在 `riichi_ppo_v1/tests/unit/test_artifact_conventions.py` 增加:
  加载 `configs/sft.yaml` 断言五个节奏键等于 `sft/contract.py` 常量;加载
  `v15_sft_offense_warmup.yaml`、`v15_sft_actor_finetune.yaml` 断言不含
  `interval_steps`/`hanchan_count` 键(先写,预期失败)

### Implementation for User Story 6

- [X] T033 [US6] `riichi_ppo_v1/configs/sft.yaml` 显式增加五个节奏键:
  `validation_interval_steps: 3000`、`checkpoint_interval_steps: 3000`、
  `heuristic_evaluation_interval_steps: 3000`、
  `heuristic_evaluation_hanchan_count: 96`、
  `heuristic_evaluation_final_hanchan_count: 96`(中文注释)
- [X] T034 [US6] `riichi_ppo_v1/configs/v15_sft_offense_warmup.yaml` 删除
  `validation_interval_steps: 3000` 与 `heuristic_evaluation_enabled: true`;
  `riichi_ppo_v1/configs/v15_sft_actor_finetune.yaml` 删除
  `heuristic_evaluation_enabled: true`
- [X] T035 [US6] 跑 `test_artifact_conventions.py` 与 SFT 相关测试至通过,
  提交主题 6「SFT 节奏 sft.yaml 单点定义」

**Checkpoint**: SFT 节奏固化

---

## Phase 9: User Story 7 - 通用性与文档路径一致(Priority: P2)

**Goal**: 默认值不锁定历史版本;README/docs 路径与实际产物一致

**Independent Test**: 扫描无历史版本锁;文档示例路径真实存在且符合规范

### Tests for User Story 7

- [X] T036 [US7] 在 `riichi_ppo_v1/tests/unit/test_artifact_conventions.py` 增加:
  扫描 `configs/`、两个 README 与 `docs/` 文本,断言不含
  `train_riichi_ppo_v14`、`train_riichi_v13_sft`、`80pct_v11`、
  `tenhou-to-mjai` 默认路径、`audit/reports/v14_ppo_20260812` 等历史锁
  (先写,预期失败)

### Implementation for User Story 7

- [X] T037 [P] [US7] `riichi_ppo_v1/README.md`:`--init-model` 示例改为
  `checkpoints/train_riichi_v13/sft/best_heuristic.pt`;precompute 重定向改为
  `logs/v13/sft-precompute-40pct.log`;补充 logs/audit 目录规范段
- [X] T038 [P] [US7] `riichi_lab_bot/README.md`:checkpoint 示例改为
  `checkpoints/train_riichi_v15/ppo/checkpoint_00480.pt`;`--jsonl-log` 示例改为
  `logs/v15/bot-online.jsonl`
- [X] T039 [P] [US7] 复核 `riichi_ppo_v1/docs/v13_sft.md` 与根 `AGENTS.md`
  的路径一致性;存在与实际不符处按宪法修正(中文注释)
- [X] T040 [US7] 跑 `test_artifact_conventions.py` 至通过,提交主题 7
  「文档路径与产物一致」

**Checkpoint**: 通用性与文档收口

---

## Phase 10: Polish & Cross-Cutting Concerns

- [X] T041 全量 Python 测试:`python -m pytest -q`(riichi_ppo_v1、
  riichi_lab_bot、RiichiEnv Python 测试)
- [X] T042 Rust 测试:`cargo test --manifest-path
  RiichiEnv/riichienv-state-machine/Cargo.toml` 与
  `cargo test --manifest-path RiichiEnv/riichienv-core/Cargo.toml`
- [X] T043 运行 [quickstart.md](./quickstart.md) 的六个验证场景并逐项通过
- [X] T044 三件套一致性复查(spec/plan/tasks 相互映射、无矛盾),检查每个主题
  commit 状态测试通过,输出交付报告

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup(Phase 1)→ Foundational(Phase 2)→ US1→US7 顺序(本仓库单实现者;
  `[P]` 标记的任务可并行,但不改变故事顺序)
- **T008 必须先于 T011–T013**(详见"关键执行顺序约束")
- Polish(Phase 10)依赖全部故事完成

### User Story Dependencies

- US1(P1):只依赖 Foundational;其 `logs/` 删除独立于其他故事
- US2(P1):归位移动需在 T008 之后;`.gitignore` 变更无外部依赖
- US3(P1):只依赖 US2 的目录归位(避免旧 audit 目录干扰引用扫描)
- US4(P1):独立
- US5(P1):依赖 US3(configs 已先行改 checkpoint 路径)
- US6(P2):依赖 US3(sft.yaml 路径已改)+ US5(机制常量模块)
- US7(P2):依赖 US3/US5(configs 与 README 路径已定)

### Within Each User Story

- 测试先写并确认失败,再实现;实现后测试通过方可提交该主题
- 文件系统移动与引用更新必须同一主题完成,避免中间态

### Parallel Opportunities

- T001/T002 并行;US2 的 T011/T012/T013 互不重叠目录可并行;US7 的
  T037/T038/T039 不同文件可并行;其余同文件任务顺序执行

---

## Parallel Example: User Story 2

```bash
# 三个版本目录互不重叠,可并行归位:
Task: "归位 audit/reports/v13_sft_20260802 → audit/reports/v13"
Task: "归位 audit/reports/v14_ppo_20260812 → audit/reports/v14"
Task: "归位 audit/reports/v15_ppo_20260814 → audit/reports/v15"
```

---

## Implementation Strategy

### MVP First(US1 优先)

1. Setup + Foundational
2. US1:日志写入点统一 → T008 停下确认后删除 `logs/` 存量 → 提交主题 1
3. **STOP and VALIDATE**:`logs/` 根为空、测试通过
4. 继续 US2 → US7,每主题一个 commit

### Incremental Delivery

- US1 完成后即满足日志规范;US2 补齐 audit 规范;US3/US4 完成产物目录规范;
  US5/US6 固化评测节奏;US7 收口文档与通用性;最后全量验证

---

## Notes

- `[P]` = 不同文件/目录、无依赖
- `[Story]` 标签映射任务到用户故事,便于追溯
- 每个故事可独立验收;提交前该故事测试通过
- 破坏性操作(T008、T024 若触发)必须停下向用户确认,绝不静默执行
- 禁止删除任何 checkpoint;审计归位只移动/重命名
