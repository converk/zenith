<!--
Sync Impact Report
- Version change: unset (placeholder) -> 1.0.0 (initial ratification)
- Added sections: Core Principles, Additional Constraints, Development
  Workflow & Quality Gates, Governance
- Removed sections: none
- Renamed principles: none
- Deferred TODOs: none
- Amendment 2026-08-15: 1.0.0 -> 1.1.0 (MINOR). Modified Principle I
  (file naming + Chinese comments), Principle II (standalone configs, no
  overlay), Principle III (checkpoint directory convention), Principle IV
  (SFT 3000-step cadence), Quality Gates (smoke-test cleanup).
- Amendment 2026-08-15: 1.1.0 -> 1.2.0 (MINOR). Added Principle VI
  (generality-first: no hardcoded dependencies on fixed configs or
  declarations across the whole codebase); redefined Principle IV (fixed
  1v3 mechanism, opponent parameterized); added governance scope covering
  all three components.
- Amendment 2026-08-15: 1.2.0 -> 1.3.0 (MINOR). Expanded Principle III
  (logs/<version>/ and audit/reports/<version>/ conventions); added
  cleanup deletion authorization to Principle II (code blocks and files
  only, artifacts excluded).
- Amendment 2026-08-16: 1.3.0 -> 1.4.0 (MINOR). Updated Principle II
  active-contract declaration: single information-encoding protocol v16
  (replacing the token-schema/feature-schema multi-version split), active
  experiment generation v16.
- Amendment 2026-08-19: 1.4.0 -> 1.5.0 (MINOR). Updated Principle II
  (active experiment generation v16 -> v17; encoding protocol remains
  v16); redefined Principle IV (PPO 1v3 mechanism: 1600 hanchan / 30
  updates -> 4000 hanchan / 5 updates, keep 10 processes); Principle V
  (rollout baseline: complete-hanchan count replaces kyokus_per_worker
  for V17 training tests).
- Amendment 2026-08-19: 1.5.0 -> 1.6.0 (MINOR). Redefined Principle IV
  (PPO 1v3 mechanism: 4000 hanchan / 10 processes -> 6000 hanchan / 12
  processes, 6 per GPU card on two cards, keep 5-update interval).
- Amendment 2026-08-20: 1.6.0 -> 1.7.0 (MINOR). Redefined Principle IV
  (PPO 1v3 mechanism: 6000 hanchan / 12 processes -> 4000 hanchan / 10
  processes, 5 per GPU card on two cards, keep 5-update interval;
  per-shard internal batch size is an implementation detail supplied by
  the version config, not a mechanism constant).
-->
# Zenith Constitution

## Core Principles

### I. 目录按职责组织(Directory by Responsibility)

- 代码、配置、文档、测试按现有包布局就近放置;禁止跨阶段放置(例如训练期使用的
  1v3 评测代码不得留在 SFT 目录,生产 CLI 入口不得放在 tests 包内)。
- 组件有独立职责时,新建文件并放入对应目录,不得塞进无关模块。
- 新增文件前先核对包布局与命名约定。
- 文件名必须自描述,能从名称看出职责与版本。
- 代码注释一律使用中文(新增与修改的代码必须遵守)。
- 清理工作以"每主题一个 commit、测试通过、可独立回滚"为单位。

### II. 单一现行版本契约(Single Active Contract)

- 现行信息编码协议为 v16(单一版本,取代 token schema / feature schema 等多版本
  拆分),token schema 仍为 13;v16 是当前活跃实验代,v17 为下一实验代(经
  2026-08-19 宪法修订登记,PPO 采用 Mortal 式 GRP 纯奖励方案)。
- 彻底放弃 v11 checkpoint 兼容(2026-08-15 决策):移除 `legacy/v11` 适配器、
  `legacy_fixed` 模型头、`build_legacy_v11` 及相关测试;v11 权重仅作冷存储保留。
- v14 实验资产保留为冷存储,不再被当前代码引用,也不恢复其训练。
- 禁止为旧 checkpoint 长期保留双版本兼容代码;新增兼容层必须在本宪法中显式批准。
- 已批准的全仓清理中,任何不再被需要的代码块与代码文件均可删除;checkpoint 与
  数据集不适用本条,仍按原则 III 的保留与删除规定执行。
- 每个版本的配置必须自包含地写在自己的文件中,禁止 overlay/继承式覆盖(如
  resume overlay);配置加载不得依赖历史配置的隐式叠加。

### III. 产物存储规范(Artifact Storage Conventions)

- 现有 checkpoint 一律保留,只允许归档移动,禁止删除(除非另行批准)。
- checkpoint 保存目录固定为 `checkpoints/train_riichi_<版本号>`,阶段产物放在其下
  的子目录(如 `train_riichi_v15/ppo`、`train_riichi_v15/sft/stage_a`);每个
  checkpoint 内部保存配置快照以便追溯。
- 现行数据集为 `tenhou_sft_2024_2025_encoded_40pct_v13_v16` 与原始
  `tenhou_sft_2024_2025`;`80pct_v11` 与 `tenhou-to-mjai` 中间产物删除
  (2026-08-15 决策)。GRP 数据集按代际命名:`tenhou_grp_2024_2025_v17`。
- 所有运行日志(json、txt、log 等运行时信息)必须写入 `logs/<版本号>/`,禁止在
  `logs/` 根目录或他处单独生成日志文件。
- `audit/reports/<版本号>/` 是每个版本的初始设计文档、实验报告、测试与验证脚本
  的存放位置;目录名固定为 `audit/reports/<版本号>/`,其下命名遵循固定类型规范,
  禁止随意命名或散落其他目录。
- CLI 默认路径与 README/docs 必须与实际产物路径一致。

### IV. 固定训练评测机制(Fixed Evaluation Mechanisms)

- PPO 唯一评测机制为 1v3 对抗:固定 10 进程 × 400 = 4000 hanchan(每进程
  400,双卡各 5 进程),每 5 updates 一次;对手模型(如 V16 SFT)由配置或命令行
  参数指定,不得硬编码任何具体版本;输出到 `audit/reports/<run>/eval`。进程数
  (10)与单进程半庄数(400)是机制常量,间隔(5 updates)按宪法登记;分片内部
  并行度(每批半庄数)属实施细节,由版本配置提供,不属于机制常量。修改须走本
  原则修订。
- 旧 `evaluation_*` 机制与相关配置、代码移除,消除双轨。
- SFT 的验证、启发式评测与 checkpoint 保存统一固定为每 3000 steps 一次;最终评估
  保持 96 hanchan;参数只在一处定义,禁止在实验配置中复制。
- 评测机制的任何改动必须走宪法修订,不得在单个实验配置里悄悄修改。

### V. 测试基线与可观测性(Test Baseline & Observability)

- 性能与训练测试固定 `target_kl=0.0`、`update_epochs=4`;rollout 基线在 V16 及
  以前为 `kyokus_per_worker=16`,自 V17 起改为「每 update 收集 512 个完整半庄」
  (rollout 停止条件为完整半庄数,不再使用 `kyokus_per_worker`);该基线独立于
  长期训练默认。
- 默认使用 `CUDA_DEVICE=0,1` 与 `learner_gpus=2`;仅显式要求单卡时才用
  `CUDA_DEVICE=0`。
- 测试默认跑 3 轮,首轮视为预热,性能统计只对后两轮单独报告。
- 默认打印耗时监控与全部相关性能指标。

### VI. 通用性优先(Generality First)

- 全仓库代码不得依赖固定的实验版本、配置值或声明:版本号、checkpoint 名、数据集
  名、对手模型、schema/契约 ID、种子基数、各类计数与间隔、默认路径等,一律通过
  CLI 参数或配置项传入;默认值不得锁定任何历史版本。
- 允许的例外:与领域本身绑定、不随版本变化的常量(如 136 TID、34 类牌、241 维
  动作空间);这类常量必须收敛为单一命名常量、单一来源,禁止散落各处。
- 固化的是机制与流程(如 1v3 对抗、评测节奏、存储规范),而不是对手模型或数据的
  具体版本。
- 新版本迭代通过参数与配置切换实现,不得以复制粘贴代码文件并加版本后缀的方式
  派生实现。

## Additional Constraints

- 本宪法与清理工作覆盖仓库全部组件:`riichi_ppo_v1`、`riichi_lab_bot`、
  `RiichiEnv`(含各自测试、文档与配置),不限于训练框架。
- CUDA_DEVICE 映射:0→物理 GPU0,1→GPU1,2→GPU3,3→GPU4;训练入口必须在启动
  PyTorch/Ray 前完成映射。
- Python 命令与训练一律使用 Conda 环境 `Mahjong-AI`。
- `RiichiEnv` 是本项目训练环境;`riichienv-state-machine` 的公开模块名为
  `riichi`,且不得依赖 `riichienv`。
- `evaluations/` 无需兼容,将整体重写。
- 代码与文档不得引用已删除的目录(例如历史遗留的 `exp/`)。

## Development Workflow & Quality Gates

- 删除任何模块或文件前必须执行全仓库引用检查;零引用且测试通过才允许删除。
- 重构以"每主题一个 commit + 测试通过"为单位交付。
- 文档(README、docs/、AGENTS.md)必须与代码路径同步;AGENTS.md 与本宪法冲突时,
  以本宪法为准。
- 新增或修改代码必须附带对应测试;协议契约变更必须同步更新
  `KyokuEventTupleProtocol.md` 等协议文档。
- 冒烟测试结束时必须删除其产生的日志与结果文件。

## Governance

- 本宪法优先于其他工程惯例;修订通过 `$speckit-constitution` 完成,并记录
  Sync Impact Report。
- 版本语义:MAJOR=原则删除或重定义;MINOR=新增或实质扩展;PATCH=措辞澄清。
- 每次清理提交必须对照本宪法做合规检查。
- AGENTS.md 保留为运行时指导文件,不得与本宪法矛盾。

**Version**: 1.7.0 | **Ratified**: 2026-08-15 | **Last Amended**: 2026-08-20
