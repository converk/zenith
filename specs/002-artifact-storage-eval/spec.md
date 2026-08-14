# Feature Specification: 产物存储与评测机制固化(Artifact Storage & Evaluation Consolidation)

**Feature Branch**: `002-artifact-storage-eval`(无 before_specify 钩子,不自动创建
git 分支;实现在当前工作分支 `sft` 上进行)

**Created**: 2026-08-15

**Status**: Draft

**Input**: 在 zenith 仓库全部组件(riichi_ppo_v1、riichi_lab_bot、RiichiEnv)固化
产物存储与评测机制:日志全部收敛到 `logs/<版本号>/` 并清空现有 `logs/`;审计产物
收敛到 `audit/reports/<版本号>/` 并固定类型命名;checkpoint 目录按
`checkpoints/train_riichi_<版本号>` 规范化(只归档移动不删除);清理两个废弃数据集;
PPO 只保留固定 1v3 评测(1600 hanchan,10 进程 × 160,每 30 updates 一次,对手参数化,
输出固定目录);SFT 验证/启发式评测/保存统一每 3000 steps 一次、最终评估 96 hanchan,
参数单点定义;全仓库消除历史版本硬依赖,CLI 默认值与文档路径与实际产物一致。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 日志规范:清空存量并收敛写入点 (Priority: P1)

作为维护者,我希望 `logs/` 根目录与仓库他处的既有运行日志全部删除(用户已决策,
不再保存),此后三个组件产生的所有运行日志(json/txt/log 等)一律写入
`logs/<版本号>/`,任何代码默认值、脚本重定向或文档示例都不再在 `logs/` 根目录或
其他位置生成日志文件。

**Why this priority**: 这是宪法原则 III 的直接要求;先清存量、后固化写入点,才能
保证新日志不再散落,并为后续版本建立可复现的日志布局。

**Independent Test**: 清空并扫描后,`logs/` 根目录无任何文件;全仓库代码默认值、
运行脚本与文档中不再出现写入 `logs/` 根目录或 `audit/` 等他处日志文件的路径;
运行一次最小冒烟,产物只出现在 `logs/<版本号>/`。

**Acceptance Scenarios**:

1. **Given** `logs/` 根目录含 127MB 历史日志(ranky/ppo_vs_sft/sft-benchmark 等),
   **When** 用户确认删除, **Then** 现有内容全部删除,不留备份。
2. **Given** `riichi_ppo_v1/sft/audit.py` 的 `--report` 默认写
   `logs/sft-audit-10kyokus.json`, **When** 统一写入点, **Then** 默认值不再写入
   `logs/` 根目录(改为不落盘或由调用方显式指定 `logs/<版本号>/` 路径)。
3. **Given** 训练入口、运行脚本与文档示例的重定向(如 `> logs/sft-precompute-40pct.log`、
   `run_v15_ppo.sh`/`run_v15_sft.sh` 的 `LOG_FILE`), **When** 统一写入点,
   **Then** 全部指向 `logs/<版本号>/`,不在脚本自身目录或 `audit/` 下生成日志。
4. **Given** Ray/训练进程产生运行时日志, **When** 启动训练, **Then** 这些日志
   流向 `logs/<版本号>/`(通过重定向或日志目录配置),不散落在仓库他处。
5. **Given** `riichi_lab_bot` 的 `--jsonl-log` 与 `--log-level`, **When** 在线运行,
   **Then** jsonl 日志路径由 CLI 显式指定且文档示例使用 `logs/<版本号>/`,不产生
   隐藏默认落盘路径。

---

### User Story 2 - audit 报告规范:版本化目录与固定类型命名 (Priority: P1)

作为维护者,我希望 `audit/reports/<版本号>/` 成为每个版本的初始设计文档、实验报告、
测试与验证脚本的唯一存放位置,目录内命名遵循固定类型规范;现有 `audit/reports/`
 下按日期命名的三个目录按版本号归位重命名,内容只归档移动不删除。

**Why this priority**: 审计产物是版本可追溯性的核心;日期命名与散落的日志/评测
输出会破坏"设计文档、实验报告、验证脚本"的可发现性,必须先行固化。

**Independent Test**: 归位后 `audit/reports/` 顶层只有 `v13`、`v14`、`v15` 三个
版本目录;每个版本目录内只有固定类型子目录(design/report/eval/scripts);三个历史
目录的总文件数不减少(纯移动/重命名);`design/`、`report/`、`scripts/` 内容被 git
跟踪,`eval/` 输出仍被忽略。

**Acceptance Scenarios**:

1. **Given** `audit/reports/v13_sft_20260802`、`v14_ppo_20260812`、
   `v15_ppo_20260814` 三个日期目录, **When** 按版本号归位重命名,
   **Then** 顶层变为 `v13`、`v14`、`v15`,内部内容按固定类型分类,文件总数不变。
2. **Given** 版本目录内散落设计文档、实验报告、验证脚本、评测输出与运行日志,
   **When** 按固定类型分类, **Then** 设计文档进 `design/`、报告与进度进 `report/`、
   评测与验证输出进 `eval/`、脚本进 `scripts/`;运行日志移入 `logs/<版本号>/`。
3. **Given** 新的实验版本, **When** 生成审计产物, **Then** 初始设计文档、实验报告、
   测试与验证脚本只能写入 `audit/reports/<新版本号>/` 的固定类型子目录,禁止随意
   命名或散落其他目录。
4. **Given** PPO 的 1v3 评测, **When** 运行评测, **Then** 输出固定写入
   `audit/reports/<版本号>/eval`。

---

### User Story 3 - checkpoint 规范化:统一目录布局与存量归档 (Priority: P1)

作为维护者,我希望新 checkpoint 固定保存到 `checkpoints/train_riichi_<版本号>` 下,
并按阶段分子目录(v15 布局为样板);现有目录按该规范归档移动、一律不删除,代码与
文档中的所有引用同步更新且测试通过。

**Why this priority**: checkpoint 布局是产物可追溯性的基础;`train_riichi_ppo_v14`
与 `train_riichi_v13_sft` 不符合 `train_riichi_<版本号>` 型,会持续误导新版本。

**Independent Test**: 归档后 `checkpoints/` 顶层为 `train_riichi_v13`、
`train_riichi_v14`、`train_riichi_v15`;全仓库对旧路径零引用;全量测试通过;
checkpoint 文件数量与移动前一致(零删除)。

**Acceptance Scenarios**:

1. **Given** `checkpoints/train_riichi_ppo_v14`, **When** 归档移动为
   `checkpoints/train_riichi_v14`, **Then** 所有引用(v14 配置、bot 文档/测试)
   同步更新,不删除任何文件。
2. **Given** `checkpoints/train_riichi_v13_sft` 被多处引用, **When** 按新规范归档
   为 `checkpoints/train_riichi_v13/sft` 并同步更新全部引用, **Then** 全量测试通过;
   若同步后测试无法通过,则作为存量例外保留原名、只对新增产物强制新规范。
3. **Given** 新版本训练, **When** 配置 checkpoint 目录, **Then** 路径必须是
   `checkpoints/train_riichi_<版本号>/<阶段>`,且每个 checkpoint 内部保存配置快照。
4. **Given** 代码中的默认 checkpoint 目录, **When** 检查, **Then** 默认值为中性
   路径(如 `checkpoints/train_riichi_current`),不锁定 `v13_sft` 等历史版本。

---

### User Story 4 - 数据集清理:删除废弃中间产物 (Priority: P1)

作为维护者,我希望删除 `datasets/tenhou_sft_2024_2025_encoded_remaining_80pct_v11`
与 `datasets/tenhou-to-mjai` 两个废弃中间产物(共约 20GB),只保留
`encoded_40pct_v13_v16` 与原始数据;删除前必须停下向用户二次确认。

**Why this priority**: 宪法原则 III 已决策删除这两个中间产物;它们占用大量磁盘,
且代码默认值若继续指向它们会不断复活已废弃的目录。

**Independent Test**: 两个废弃目录不再存在;`tenhou_sft_2024_2025` 与
`encoded_40pct_v13_v16` 完好;代码默认值与文档不再把这两个已删目录作为默认路径。

**Acceptance Scenarios**:

1. **Given** 两个废弃目录存在, **When** 用户二次确认, **Then** 删除完成,现行
   原始数据与 `encoded_40pct_v13_v16` 不受影响。
2. **Given** 两个废弃目录经核实已不存在, **When** 检查磁盘与代码引用,
   **Then** 以核实结果为准、跳过删除,并确保代码默认值不会静默重建它们。
3. **Given** `riichi_ppo_v1/sft/prepare.py --archive-dir` 默认指向
   `datasets/tenhou-to-mjai`, **When** 参数化, **Then** 默认值不再指向已废弃目录
   (改为必填或中性路径)。

---

### User Story 5 - 评测机制统一:唯一 1v3 机制与固定输出 (Priority: P1)

作为维护者,我希望 PPO 只保留 1v3 对抗评测:固定 1600 hanchan(10 进程 × 160)、
每 30 updates 一次;对手模型通过 CLI/配置指定,不硬编码任何具体版本;输出固定到
`audit/reports/<版本号>/eval`;旧的 `evaluation_*` 双轨不复存在,`train.py` 中不再
有硬编码的历史版本输出目录。

**Why this priority**: 评测机制是版本间可比的唯一依据;机制参数固化、可变项参数化
是宪法原则 IV 与 VI 的核心,且双轨已在前一 feature 移除,本 feature 收尾固化。

**Independent Test**: 机制常量(10/160/1600/30)在全仓库只有单一命名定义;代码默认
值不含任何对手模型版本或 `v14` 历史输出目录;版本配置的 `eval1v3_output_dir`
统一为 `audit/reports/<版本号>/eval`;评测摘要与 PROGRESS 写入该版本审计目录。

**Acceptance Scenarios**:

1. **Given** 1v3 机制常量, **When** 扫描代码, **Then** 10 进程、160 半庄/进程、
   共 1600 半庄、每 30 updates 一次,仅在一处命名常量定义,其余位置引用该常量。
2. **Given** 对手模型, **When** 配置评测, **Then** `eval1v3_model_b` 由版本配置或
   CLI 指定,代码默认值不含任何具体版本路径。
3. **Given** v14/v15 版本配置, **When** 检查 `eval1v3_output_dir`,
   **Then** 统一为 `audit/reports/v14/eval`、`audit/reports/v15/eval`,不再含日期
   后缀目录。
4. **Given** 评测摘要 `eval1v3.jsonl` 与 `PROGRESS.md`, **When** 运行训练评测,
   **Then** 写入 `audit/reports/<版本号>/eval` 与 `audit/reports/<版本号>/report`,
   不再写入 checkpoint 目录或版本目录根。
5. **Given** `train.py` 与评测入口, **When** 扫描, **Then** 不存在硬编码的 v14
   默认输出目录;缺失必要配置时明确报错。

---

### User Story 6 - SFT 节奏固定:3000 steps 单点定义 (Priority: P2)

作为维护者,我希望 SFT 的验证、启发式评测与 checkpoint 保存统一为每 3000 steps
一次,最终评估保持 96 hanchan;这些节奏参数在 `sft.yaml` 单点定义,实验配置不再
复制这些参数,改动只需改一处。

**Why this priority**: 节奏参数散落在实验配置中会漂移,违反宪法原则 IV"评测机制
任何改动必须走宪法修订";单点定义后新实验无法悄悄改节奏。

**Independent Test**: 3000/96 只在单一命名常量(机制源)与 `sft.yaml` 中出现;
任何实验配置不包含验证/启发式评测/checkpoint 间隔键;一致性测试保证 `sft.yaml`
 与机制常量数值相同。

**Acceptance Scenarios**:

1. **Given** `sft.yaml`, **When** 检查, **Then** 验证、启发式评测与 checkpoint
   保存间隔均为 3000 steps,最终评估为 96 hanchan,参数只在此文件与机制常量中定义。
2. **Given** `v15_sft_offense_warmup.yaml`、`v15_sft_actor_finetune.yaml` 等实验
   配置, **When** 检查, **Then** 不再复制 `validation_interval_steps`、
   `checkpoint_interval_steps`、`heuristic_evaluation_interval_steps` 等节奏键。
3. **Given** 未来调整节奏, **When** 修改, **Then** 只需改一处且测试立即发现
   不一致,不会出现实验配置与基础配置漂移。

---

### User Story 7 - 通用性与文档路径一致 (Priority: P2)

作为维护者,我希望本 feature 涉及的对手模型、数据集默认路径、种子基数、计数与
间隔、输出路径一律走 CLI 参数或配置项,默认值不锁定任何历史版本;CLI 默认值与
README/docs 中的数据集、checkpoint、日志路径与实际产物一致。

**Why this priority**: 宪法原则 VI 要求机制固化、版本不固化;文档与默认值若锁定
历史版本,新版本迭代仍会复制粘贴而非参数切换。

**Independent Test**: 全仓库扫描后,代码默认值不含 `v13_sft`/`v14`/`80pct_v11` 等
历史版本依赖;README/docs 中引用的每个数据集、checkpoint、日志与 audit 路径真实
存在;全量测试通过。

**Acceptance Scenarios**:

1. **Given** SFT/PPO 训练入口与 bot 入口的默认参数, **When** 扫描, **Then**
   默认值均为中性值或必填,不指向 `train_riichi_v13_sft`、
   `train_riichi_ppo_v14/checkpoint_00510.pt` 等历史产物。
2. **Given** 版本配置文件, **When** 引用自身版本的产物, **Then** 路径使用该版本
   的规范化目录(`train_riichi_v14`、`audit/reports/v14/eval` 等)。
3. **Given** README/docs 的示例命令, **When** 逐一核对, **Then** 数据集、
   checkpoint、日志与 audit 路径与实际产物一致,日志示例使用 `logs/<版本号>/`。

---

### Edge Cases

- `logs/` 存量删除与数据集删除属破坏性操作:执行前必须停下向用户确认;删除范围
  只限 `logs/` 现有内容与两个指定数据集目录,绝不误删 checkpoint 或其他数据。
- `audit/` 的 `.gitignore` 已确定为"整体忽略 + 仅放行 design/report/scripts 固定
  类型子目录";实现时需用 `git check-ignore` 验证放行规则生效且 `eval/` 输出仍被
  忽略。
- `train_riichi_v13_sft` 改名涉及多处引用:若同步更新后全量测试无法通过,回退为
  存量例外保留原名,只对新增产物强制新规范;绝不删除任何 checkpoint。
- 归档移动目录时,引用更新必须与移动在同一主题内完成,避免中间态指向不存在路径。
- 评测输出目录可能已存在:写入时不得破坏同版本既有结果,摘要文件追加而非覆盖。
- Ray/训练子进程日志可能写入系统临时目录:必须显式收敛到 `logs/<版本号>/`,否则
  视为"他处单独生成日志文件"。
- 冒烟测试结束必须清理其产生的日志与结果文件,不得污染 `logs/<版本号>/`。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统 MUST 在用户确认后删除 `logs/` 现有全部内容(不归档、不保存);
  此后任何运行不得在 `logs/` 根目录生成日志文件。
- **FR-002**: 三个组件(riichi_ppo_v1、riichi_lab_bot、RiichiEnv)的所有运行日志
  写入点 MUST 统一写入 `logs/<版本号>/`(json/txt/log 等运行时信息),禁止写入
  `logs/` 根目录或仓库他处。
- **FR-003**: 系统 MUST 盘点并改写全部日志写入点,至少包括 `sft/audit.py` 的
  `--report` 默认值、训练入口与运行脚本的重定向/Ray 日志、`riichi_lab_bot` 的
  `--jsonl-log` 文档示例、README/docs 中的日志重定向示例。
- **FR-004**: `audit/reports/<版本号>/` MUST 是每个版本初始设计文档、实验报告、
  测试与验证脚本的唯一存放位置;其下命名 MUST 遵循固定类型规范(`design/` 设计、
  `report/` 报告与进度、`eval/` 评测与验证输出、`scripts/` 测试与验证脚本),
  禁止随意命名或散落其他目录。
- **FR-005**: 现有 `audit/reports/v13_sft_20260802`、`v14_ppo_20260812`、
  `v15_ppo_20260814` MUST 按版本号归位重命名为 `v13`、`v14`、`v15`,并按固定类型
  重新归类;运行日志移入 `logs/<版本号>/`;整个归位过程只移动/重命名,不删除内容。
- **FR-006**: `audit/` MUST 保持整体忽略,但 `.gitignore` MUST 放行固定类型子目录
  (`audit/reports/**/design/`、`audit/reports/**/report/`、
  `audit/reports/**/scripts/`),使初始设计文档、实验报告与测试/验证脚本进入版本
  控制;`eval/` 及其它大体积评测输出 MUST 继续被忽略。
- **FR-007**: 新 checkpoint MUST 固定保存到 `checkpoints/train_riichi_<版本号>`
  下,按阶段(ppo/sft/stage 等)分阶段子目录;每个 checkpoint 内部 MUST 保存配置
  快照以便追溯。
- **FR-008**: 现有 checkpoint 目录 MUST 只归档移动、不删除:
  `train_riichi_ppo_v14` 归为 `train_riichi_v14`;`train_riichi_v13_sft` 归为
  `train_riichi_v13/sft` 并同步更新全部引用,若测试无法通过则作为存量例外保留
  原名。
- **FR-009**: checkpoint 目录的代码默认值 MUST 为中性路径
  (如 `checkpoints/train_riichi_current`),不得锁定 `v13_sft` 等历史版本。
- **FR-010**: 系统 MUST 删除 `datasets/tenhou_sft_2024_2025_encoded_remaining_80pct_v11`
  与 `datasets/tenhou-to-mjai`(执行前停下二次确认,共约 20GB);MUST 保留
  `tenhou_sft_2024_2025` 与 `encoded_40pct_v13_v16`;代码默认值 MUST NOT 静默重建
  已删除目录。
- **FR-011**: PPO MUST 只保留 1v3 评测机制:固定 1600 hanchan(10 进程 × 160)、
  每 30 updates 一次;这些机制常量 MUST 只在单一命名常量处定义,其余位置引用。
- **FR-012**: 1v3 对手模型 MUST 由 CLI/配置指定(如 `eval1v3_model_b`),代码默认
  值 MUST NOT 硬编码任何具体版本;评测输出 MUST 固定写入
  `audit/reports/<版本号>/eval`。
- **FR-013**: 评测摘要 `eval1v3.jsonl` 与 `PROGRESS.md` MUST 写入
  `audit/reports/<版本号>/eval` 与 `audit/reports/<版本号>/report`,不得写入
  checkpoint 目录或版本目录根;`train.py` MUST NOT 保留硬编码的 v14 默认输出目录。
- **FR-014**: SFT 的验证、启发式评测与 checkpoint 保存 MUST 统一每 3000 steps
  一次,最终评估 MUST 保持 96 hanchan;参数 MUST 在 `sft.yaml` 单点定义并与机制
  常量一致,实验配置 MUST NOT 复制这些参数。
- **FR-015**: 对手模型、数据集默认路径、种子基数、计数与间隔、输出路径等 MUST
  通过 CLI 参数或配置项传入;默认值 MUST NOT 锁定任何历史版本。
- **FR-016**: CLI 默认值与 README/docs 中的数据集、checkpoint、日志、audit 路径
  MUST 与实际产物一致,不得引用已删除或未规范化的路径。
- **FR-017**: 每个主题 MUST 一个 commit 交付,且该 commit 状态下测试通过;
  破坏性操作(日志、数据集删除)MUST 在用户确认后执行。
- **FR-018**: checkpoint 一律 MUST NOT 删除;新增或修改代码的注释 MUST 使用中文。

### Key Entities *(include if feature involves data)*

- **版本号(Version Tag)**: `v13`/`v14`/`v15` 等代际标识,同时用于
  `logs/<版本号>/`、`audit/reports/<版本号>/` 与
  `checkpoints/train_riichi_<版本号>`。
- **日志制品(Log Artifact)**: 运行时产生的 json/txt/log 文件,唯一合法位置为
  `logs/<版本号>/`。
- **审计制品类型(Audit Artifact Type)**: 固定四类:`design/`(初始设计文档)、
  `report/`(实验报告与进度)、`eval/`(评测与验证输出)、`scripts/`(测试与验证脚本)。
- **Checkpoint 布局(Checkpoint Layout)**: `checkpoints/train_riichi_<版本号>` 及
  其下阶段子目录,每个 checkpoint 内含配置快照。
- **评测运行(Evaluation Run)**: 固定 1v3 对抗(10 进程 × 160 半庄,每 30 updates
  一次),输出到 `audit/reports/<版本号>/eval`。
- **SFT 节奏参数(SFT Cadence)**: 3000 steps 的验证/启发式评测/保存间隔与 96
  hanchan 的最终评估量,单点定义。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 三组件全量测试(Python 与 Rust)100% 通过,按宪法测试基线运行。
- **SC-002**: 清理后 `logs/` 根目录无任何文件;全仓库扫描后,代码默认值、运行脚本
  与文档中不存在写入 `logs/` 根目录或仓库他处的日志路径。
- **SC-003**: `audit/reports/` 顶层恰为 `v13`、`v14`、`v15`,各版本目录只含固定
  类型子目录;PPO 1v3 输出路径为 `audit/reports/<版本号>/eval`。
- **SC-004**: `checkpoints/` 顶层符合 `train_riichi_<版本号>` 规范;全仓库对
  `train_riichi_ppo_v14` 与 `train_riichi_v13_sft` 旧路径零引用;checkpoint 文件
  数量与移动前一致(零删除)。
- **SC-005**: `80pct_v11` 与 `tenhou-to-mjai` 两个废弃数据集不存在,现行原始数据
  与 `encoded_40pct_v13_v16` 完好;代码默认值不再指向这两个目录。
- **SC-006**: 1v3 机制常量(10/160/1600/30)全仓库单一命名定义;代码默认值不含
  任何对手模型版本或 v14 历史输出目录。
- **SC-007**: SFT 节奏键只出现在 `sft.yaml`(及机制常量),任何实验配置不含复制;
  数值为 3000 steps / 96 hanchan,且有一致性测试防漂移。
- **SC-008**: README/docs 与 CLI 默认值引用的每个数据集、checkpoint、日志与 audit
  路径真实存在且符合规范。
- **SC-009**: spec/plan/tasks 三件套一致;实现按主题分 commit 且逐 commit 测试
  通过。

## Assumptions

- 版本号统一使用裸代际标签(`v13`/`v14`/`v15`),取代现有的
  `v13_sft_20260802`/`v14_ppo_20260812`/`v15_ppo_20260814` 混合命名;checkpoint
  目录据此为 `train_riichi_v13`、`train_riichi_v14`、`train_riichi_v15`。
- audit 固定类型规范采用 `design/`、`report/`、`eval/`、`scripts/` 四类子目录;
  历史目录内的运行日志(如 `train.log`、`ranked_*.jsonl`)归位到 `logs/<版本号>/`,
  评测输出归入 `eval/`,环境/命令快照并入 `report/`。
- `train_riichi_v13_sft` 优先按新规范改名为 `train_riichi_v13/sft` 并同步全部引用;
  若全量测试无法通过则回退为存量例外,只对新增产物强制新规范。
- 代码推导 `logs/<版本号>/` 时,版本号从 `checkpoint_dir` 的
  `train_riichi_<版本号>` 基名推导;运行脚本属版本专用产物,可直接写
  `logs/v15` 等字面路径。
- Ray/子进程日志通过标准输出重定向与日志目录配置收敛到 `logs/<版本号>/`;
  `metrics.jsonl` 与 TensorBoard 事件保持在 checkpoint 目录(属 v15 既定布局的
  训练指标,不视为 `logs/` 运行日志);SFT 启发式评测输出保持在对应阶段输出目录。
- 数据集清理前先核实两个废弃目录是否仍存在;不存在则跳过删除,但保留代码默认值
  的参数化改造。
- 破坏性操作(日志删除、数据集删除)在执行前停下向用户确认;`.gitignore` 已确认
  采用方案 A(整体忽略 audit/ + 放行 design/report/scripts 固定类型子目录)。
