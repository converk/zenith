# Feature Specification: 代码与目录治理(Code & Directory Governance)

**Feature Branch**: `sft`

**Created**: 2026-08-15

**Status**: Draft

**Input**: 对 riichi_ppo_v1、riichi_lab_bot、RiichiEnv(含 Python 与 Rust)三组件及
其测试、文档、配置执行代码与目录治理:跨阶段代码搬迁、移除 v11 checkpoint 兼容、
配置归一、消除固定版本/配置硬依赖、清理幽灵引用,并统一命名与注释规范。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 目录职责归位:跨阶段代码搬迁 (Priority: P1)

作为维护者,我希望训练期使用的 1v3 评测代码不再留在 SFT 目录、生产 CLI 入口不再
放在 tests 包内,让每个目录的职责可以从名称和位置直接判断,且搬迁后所有引用与测试
仍然通过。

**Why this priority**: 跨阶段放置是后续所有治理工作的前置条件,1v3 评测与验证入口
是当前最明确的错误归置,且宪法原则 I 明确禁止。

**Independent Test**: 搬迁完成后,1v3 评测模块与验证 CLI 位于各自职责目录;
`python -m pytest` 全量通过;生产入口 `riichi-ppo-validate` 可正常调用且不依赖
tests 包。

**Acceptance Scenarios**:

1. **Given** `riichi_ppo_v1/sft/` 中含 1v3 评测与验证入口代码, **When** 将
1v3 评测代码移到共享评测目录、`riichi-ppo-validate` 入口移出 `tests/`,
**Then** 全部 import、入口脚本、打包清单与测试同步更新,全量测试通过。
2. **Given** 搬迁完成, **When** 查看各目录, **Then** `sft/` 只含 SFT 阶段职责,
评测代码位于评测职责目录,`tests/` 只含测试代码。
3. **Given** 搬迁完成, **When** 运行 `riichi-ppo-validate --help`,
**Then** 命令可用且行为与原入口一致。

---

### User Story 2 - 移除 v11 checkpoint 兼容 (Priority: P1)

作为维护者,我希望彻底移除 v11 checkpoint 兼容代码(适配器、legacy_fixed 模型头、
build_legacy_v11 构造器及相关测试),使仓库只有 v13 单一现行契约,而 v11 权重仍以
冷存储保留,不删除任何 checkpoint。

**Why this priority**: 双版本兼容是宪法原则 II 明令禁止的技术债,移除后才能收敛
模型头与决策分析的实现。

**Independent Test**: 删除后 `rg` 全仓库对 `legacy/v11`、`legacy_fixed`、
`build_legacy_v11` 零引用;全量测试通过;checkpoint 目录与数据集未被改动。

**Acceptance Scenarios**:

1. **Given** `riichi_ppo_v1/legacy/v11/` 及相关测试存在, **When** 确认无
训练/推理路径引用后删除适配器、contract、encoder、model 与测试, **Then** 模型头
只支持 `isolated_action_query`,全量测试通过。
2. **Given** `model/architecture.py` 含 `legacy_fixed` 头分支, **When** 移除该分支
与默认值, **Then** 模型构造默认使用现行头,旧头相关测试同步删除或改写。
3. **Given** `training/rewards/decision.py` 含 `build_legacy_v11`,
**When** 移除该方法与 `_legacy_v11` 参数分支, **Then** 决策分析仅支持现行契约。
4. **Given** 历史 v11 checkpoint 文件存在于 `checkpoints/`, **When** 执行清理,
**Then** 所有 checkpoint 与数据集保持不变(仅代码与测试删除)。

---

### User Story 3 - 配置归一:自包含配置与移除 overlay (Priority: P1)

作为维护者,我希望每个版本配置自包含、不再通过 overlay/继承叠加历史配置,
`v14_ppo_resume.yaml` 展平为独立完整配置,配置加载不再隐式叠加历史 full-config
overlay,默认配置与现行 v13 schema / v15 活跃实验对齐。

**Why this priority**: 配置继承是引入隐式历史依赖的主要来源,直接违反宪法原则 II,
也是原则 VI 参数化的前提。

**Independent Test**: 展平后的 `v14_ppo_resume.yaml` 单独加载即可得到完整有效配置;
加载器不再存在 full-config overlay 合并逻辑;相关配置测试改写后通过。

**Acceptance Scenarios**:

1. **Given** `v14_ppo_resume.yaml` 仅含 `resume`/`init_model` 两行, **When** 展平为
与 `v14_ppo.yaml` 等价的完整自包含配置, **Then** 该文件独立可加载、不依赖任何
父配置。
2. **Given** 配置加载器先合并打包默认组再叠加 full-config overlay, **When** 移除
历史 full-config overlay 机制, **Then** 版本配置文件必须自包含,组级 CLI 覆盖保持
显式、可审计。
3. **Given** 默认配置 `training.yaml`, **When** 对照宪法与 v15 实验配置,
**Then** 默认值与现行 v13 schema / v15 活跃实验一致,不含已废弃的旧评测机制项。

---

### User Story 4 - 通用性优先:消除固定版本与配置硬依赖 (Priority: P2)

作为维护者,我希望全仓库(三个组件)不再硬编码版本号、contract/schema ID、对手
模型、数据集名、种子基数、各类计数与间隔、默认路径;这些依赖全部通过 CLI 参数或
配置项传入,默认值不锁定任何历史版本;领域不变常量(241 维动作空间、34 类牌、
136 TID)收敛为单一命名常量、单一来源。

**Why this priority**: 宪法原则 VI 的全仓库通用性要求;不消除硬依赖,新版本仍需
复制粘贴代码,违背"以参数与配置切换版本"。

**Independent Test**: 全仓库扫描后,固定版本/路径/种子等仅出现在实验配置或 CLI
参数处;领域常量在三个组件内各只有单一命名定义;全量测试通过。

**Acceptance Scenarios**:

1. **Given** 训练入口含 `eval1v3_output_dir`、`evaluation_seed_base`、
`eval1v3_seed_base` 等锁定历史版本的默认值, **When** 改为中性默认值或必填
配置/CLI 项, **Then** 默认值不再指向任何历史实验目录或日期。
2. **Given** 对手模型、数据集名、checkpoint 名散落在代码默认值中,
**When** 收敛为 CLI 参数或配置项, **Then** 代码默认值不含具体历史版本。
3. **Given** 241/34/136 等常量散落多处, **When** 收敛为单一命名常量,
**Then** 其余位置引用该常量,不再出现重复魔法数字。
4. **Given** riichi_lab_bot 的默认 checkpoint 路径与种子, **When** 改为环境变量/
CLI 可覆盖且默认值不锁定历史版本, **Then** 在线 bot 与新版本切换无需改代码。
5. **Given** RiichiEnv(Python 与 Rust)中存在版本/计数硬编码,
**When** 扫描并参数化或收敛为领域常量, **Then** Rust 与 Python 测试全部通过。

---

### User Story 5 - 清理幽灵引用 (Priority: P2)

作为维护者,我希望删除对已不存在目录(如 `exp/`)的注释与文档引用,并修正
AGENTS.md、README、docs 中与实际不符的路径,三个组件一视同仁。

**Why this priority**: 误导性路径引用会引导新维护者读写错误位置,是治理完成度的
直接验收项。

**Independent Test**: `rg` 全仓库(排除产物目录)对 `exp/` 及历史错误路径零命中;
文档中引用的每个路径真实存在;全量测试通过。

**Acceptance Scenarios**:

1. **Given** AGENTS.md/README/docs 引用 `exp/` 或已删除模块, **When** 删除或改写
这些引用, **Then** 无任何对不存在目录的引用。
2. **Given** README/docs 中的命令与路径(如 checkpoint、dataset、日志目录),
**When** 与仓库实际目录逐一核对, **Then** 全部一致。
3. **Given** 三个组件各自的 README/docs, **When** 检查, **Then** 每个组件的文档
只引用其真实存在的路径与入口。

---

### User Story 6 - 命名、注释与交付卫生 (Priority: P3)

作为维护者,我希望文件名自描述,新增与修改的代码注释一律中文;冒烟测试结束后自动
清理其产生的日志与结果文件;每个目录能用一句话说明职责;每次交付按主题一个
commit,且每个 commit 测试通过。

**Why this priority**: 这些是可观测的工程质量约束,使治理结果可审计、可回滚。

**Independent Test**: 每个目录职责可一句话说明;新增/修改注释为中文;冒烟运行后
无残留日志/结果文件;git 历史按主题切分且逐 commit 可通过对应测试。

**Acceptance Scenarios**:

1. **Given** 新增/移动的文件, **When** 检查文件名, **Then** 名称自描述其职责。
2. **Given** 新增/修改的代码注释, **When** 检查, **Then** 全部为中文。
3. **Given** 运行冒烟测试, **When** 结束, **Then** 其产生的日志与结果文件被删除。
4. **Given** 完成一项治理主题, **When** 提交, **Then** 一个主题一个 commit,
且该 commit 状态下的测试通过。

---

### Edge Cases

- 删除 v11 代码后,外部脚本或 bot 仍按旧 import 路径导入时,应当立即失败并给出
明确错误,而不是静默降级。
- 移动模块后,`pyproject.toml` 打包清单、console script 入口与 `pytest` 导入路径
必须同步,否则安装后的入口与源码树行为不一致。
- 配置展平后,若同时通过 CLI 传入覆盖值,CLI 优先级必须明确且与文档一致。
- 领域常量(241/34/136)若在 Rust 与 Python 两侧分别定义,需保证数值一致并有
交叉验证测试,防止漂移。
- `riichienv-state-machine` 的公开模块名必须保持为 `riichi`,且不得依赖
`riichienv`;任何治理改动不得破坏该协议边界。
- 冒烟/测试产生的日志、结果文件清理范围只限测试自身产物,不得误删 checkpoint、
数据集或历史实验日志。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统 MUST 将 1v3 评测代码(头对头评测、分片评测、评测策略适配器、
评测用例与共享评测工具)移出 `riichi_ppo_v1/sft/`,放入职责一致的评测目录;
`sft/` 仅保留 SFT 阶段职责。
- **FR-002**: 系统 MUST 将生产入口 `riichi-ppo-validate` 移出
`riichi_ppo_v1/tests/` 包,并同步更新 console script、打包清单与全部引用。
- **FR-003**: 每次文件/代码块搬迁 MUST 同步更新全部 import 与引用,并跑通全量
测试;按"一个主题一个 commit"交付,每个 commit 状态测试通过。
- **FR-004**: 系统 MUST 移除 v11 checkpoint 兼容:`legacy/v11` 适配器包、
`model/architecture.py` 的 `legacy_fixed` 模型头、`training/rewards/decision.py`
的 `build_legacy_v11` 及相关测试全部删除;模型头只支持现行契约。
- **FR-005**: v11 checkpoint 权重 MUST 保留为冷存储;任何 checkpoint 与数据集
文件 MUST NOT 被删除。
- **FR-006**: 每个版本配置 MUST 自包含:`v14_ppo_resume.yaml` 展平为完整配置,
不再依赖父配置 overlay。
- **FR-007**: 配置加载 MUST 移除历史 full-config overlay 合并机制,不隐式叠加
历史配置;默认配置 MUST 与现行 v13 schema 与 v15 活跃实验对齐。
- **FR-008**: 全仓库(riichi_ppo_v1、riichi_lab_bot、RiichiEnv 及三者测试/文档/
配置)MUST 扫描并把版本号、contract/schema ID、对手模型、数据集名、种子基数、
各类计数与间隔、默认路径等硬编码依赖改为 CLI 参数或配置项;默认值 MUST NOT
锁定任何历史版本。
- **FR-009**: 领域不变常量(241 维动作空间、34 类牌、136 TID)MUST 收敛为单一
命名常量、单一来源;三组件内不得重复定义魔法数字。
- **FR-010**: 系统 MUST 删除对已不存在目录(如 `exp/`)的注释与文档引用,并修正
AGENTS.md、README、docs 中与实际不符的路径;三个组件一视同仁。
- **FR-011**: 文件名 MUST 自描述;新增与修改的代码注释 MUST 一律中文。
- **FR-012**: 冒烟测试结束后 MUST 自动清理其产生的日志与结果文件。
- **FR-013**: 不再被需要的代码块与代码文件 MAY 删除,删除前 MUST 用 `rg` 全仓库
确认零引用并跑通测试;checkpoint 与数据集一律不删。
- **FR-014**: `RiichiEnv` 的协议边界 MUST NOT 被破坏:`riichienv-state-machine`
公开模块名保持 `riichi`,且不得依赖 `riichienv`。
- **FR-015**: 每个目录 MUST 能用一句话说明其职责;治理完成后提供目录职责清单。

### Key Entities

- **治理主题(Governance Topic)**: 一次可独立提交、可回滚的变更单元,对应一个
commit,包含文件搬迁/删除/配置改造及其测试,范围以目录职责或技术债为边界。
- **配置单元(Configuration Unit)**: 自包含的版本配置,含该版本运行所需的全部
参数,不依赖其它配置叠加。
- **契约与领域常量(Contract & Domain Constants)**: 单一来源的 schema/契约 ID
与领域不变常量(动作空间维度、牌类数、TID 数),由一处命名常量定义。
- **目录职责清单(Directory Responsibility Map)**: 每个目录职责的一句话描述,
作为验收与后续治理的参考。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 全仓库测试(riichi_ppo_v1、riichi_lab_bot、RiichiEnv Python 与 Rust)
100% 通过。
- **SC-002**: `rg` 对已删除模块(如 `legacy/v11`、`legacy_fixed`、
`build_legacy_v11`)与幽灵路径(如 `exp/`)零命中。
- **SC-003**: 代码默认值中不再出现锁定历史版本/日期/实验目录的硬编码;全部改为
中性默认值或 CLI/配置项,覆盖率为 100%(以扫描清单为准)。
- **SC-004**: 241、34、136 三个领域常量在三个组件内各只存在单一命名定义。
- **SC-005**: 三个组件的每个目录都有一句话职责说明,职责清单与实际结构一致。
- **SC-006**: spec/plan/tasks 三件套相互一致:spec 的用户故事可映射到 plan 的设计
与 tasks 的任务,无缺失、无矛盾。
- **SC-007**: 治理改动按主题提交,每个 commit 可独立回滚且其状态测试通过。

## Assumptions

- 审计报告 `audit/reports/project_cleanup_audit_20260815/report.md` 不存在时,以
仓库当前状态与宪法 v1.3.0 为唯一事实来源;本 spec 已内嵌全部关键结论。
- 工作区当前对 `AGENTS.md` 的未提交修改(移除 `exp/` 引用、修正测试基线)属于本
feature 的清理范围,将在对应主题 commit 中一并提交。
- 评测的固定机制(1v3 对抗 1600 hanchan、每 30 updates 一次、SFT 每 3000 steps
评测、最终 96 hanchan)作为机制常量保留,但对手模型、数据集、输出目录等可变项
必须来自配置/CLI。
- `evaluations/` 组件无需兼容,将整体重写,不在本 feature 治理范围内。
- checkpoint 与数据集目录(`checkpoints/`、`datasets/`)内容一律不动,仅代码、测试、
文档与配置变化。
- 默认性能/训练测试基线遵循宪法:`target_kl=0.0`、`update_epochs=4`、
`kyokus_per_worker=16`,默认 `CUDA_DEVICE=0,1`、`learner_gpus=2`,3 轮、首轮预热。
