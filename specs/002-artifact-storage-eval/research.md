# Research: 产物存储与评测机制固化

本 feature 的"研究"以仓库现状盘点与决策收敛为主。盘点基于 2026-08-15 的 `sft`
分支现状(HEAD `397be8c`),上一 feature(001 代码与目录治理)59 项任务已全部完成;
`audit/reports/project_cleanup_audit_20260815/report.md` 不存在,以宪法 v1.3.0 与
仓库现状为唯一事实来源。按当前多代理协作模式约束,研究由主代理直接完成,不派发
独立研究子代理。

## 现状盘点(与 001 的边界)

- 001 已完成:旧 `evaluation_*` 双轨移除(`training/evaluation.py` 已删,
  `monitoring.yaml` 已纯化为性能监控)、1v3 机制常量(10/160/30)落地于
  `evaluation/head_to_head_1v3_shards.py`、SFT 节奏常量(3000/96)落地于
  `sft/contract.py`、配置自包含与硬依赖参数化、`_progress_md_path` 的 v14 历史默认
  已删除。
- 001 明确不做(本 feature 承接):`logs/` 存量删除与写入点统一、`audit/reports/`
  版本化归位、两个废弃数据集删除、checkpoint 目录归档改名、评测输出目录固化、
  SFT 节奏在 `sft.yaml` 的单点定义与实验配置去重。
- 现状核实:
  - `logs/` 根目录约 127MB 历史日志,无 `logs/<版本号>/` 布局;
  - `audit/reports/` 顶层为 `v13_sft_20260802`、`v14_ppo_20260812`、
    `v15_ppo_20260814` 三个日期目录,内容为设计/报告/脚本/日志/评测混合;
  - `checkpoints/` 顶层为 `train_riichi_ppo_v14`、`train_riichi_v13_sft`、
    `train_riichi_v15`(v15 已符合 `train_riichi_<版本号>/<阶段>` 样板);
  - `datasets/` 仅存 `tenhou_sft_2024_2025`(5.6G)与
    `tenhou_sft_2024_2025_encoded_40pct_v13_v16`(14G);`80pct_v11` 与
    `tenhou-to-mjai` 经 `find` 全盘核实已不存在(历史已删),代码/文档中亦无
    `80pct` 引用,仅 `prepare.py --archive-dir` 默认值仍指向 `tenhou-to-mjai`;
  - `audit/` 在根 `.gitignore` 中整目录忽略;
  - 1v3 分片驱动已按 10×160 执行,`v14_ppo.yaml`/`v15_ppo.yaml` 的
    `eval1v3_output_dir` 仍为 `audit/reports/<版本>_<日期>/eval` 混合命名;
    `eval1v3.jsonl` 摘要仍追加到 `checkpoint_dir/eval1v3.jsonl`,`PROGRESS.md`
    仍落在版本目录根。

## Decision 1: 日志版本目录与写入点统一

- **Decision**: `logs/` 存量经用户确认后全部删除;此后运行日志统一写
  `logs/<版本号>/`。代码级写入点改造为:
  - `sft/audit.py --report` 默认值由 `logs/sft-audit-10kyokus.json` 改为
    `None`(不落盘),调用方需显式指定合法路径;
  - `training/train.py` 在 `ray.init` 前 `setdefault("RAY_LOG_TO_STDERR","1")`,
    使 Ray/子进程日志流向进程 stderr,由运行脚本统一重定向到 `logs/<版本号>/`;
  - 运行脚本(`run_v15_ppo.sh`、`run_v15_sft.sh`、`run_sft_baseline.sh` 等)的
    `LOG_FILE` 改为 `logs/<版本号>/...`;
  - `riichi_lab_bot` 的 `--jsonl-log` 无默认值(已合规),README 示例改用
    `logs/<版本号>/`。
- **Rationale**: 宪法原则 III 明确"所有运行日志必须写入 `logs/<版本号>/`,禁止在
  `logs/` 根目录或他处单独生成";`RAY_LOG_TO_STDERR` + 脚本重定向是改动最小且不
  引入新配置键的方案;`metrics.jsonl`/TensorBoard 是训练指标制品,保持在
  checkpoint 目录(v15 既定布局),不属于 `logs/` 运行日志。
- **Alternatives considered**: 给每个训练配置新增 `log_dir` 键(引入与
  `checkpoint_dir` 重复的版本来源);把 Ray 日志目录重定向到仓库内(依赖 Ray 内部
  常量,脆弱)。均被否。

## Decision 2: audit/reports/<版本号> 固定类型规范与存量归位

- **Decision**: 版本号统一为裸代际标签(`v13`/`v14`/`v15`)。每个版本目录只允许
  四种固定类型子目录:
  - `design/`:初始设计文档(*.md);
  - `report/`:实验报告、进度记录与运行快照(REPORT.md、PROGRESS.md、
    environment/git_status/commands 等);
  - `eval/`:评测与验证输出(1v3 输出固定写此;历史评测输出归入);
  - `scripts/`:测试与验证脚本(*.py、*.sh)。
  运行日志(*.log、事件流 *.jsonl)不属于 audit,归位到 `logs/<版本号>/`。
  存量归位映射:
  - `v13_sft_20260802` → `v13`(REPORT.md→report/,scripts/→scripts/,
    审计输出 json→eval/,环境与命令快照→report/);
  - `v14_ppo_20260812` → `v14`(PROGRESS.md→report/,*.py|*.sh→scripts/,
    `eval/`、`eval_holdout2k/`、`eval_510_vs_120_2400/`→eval/,
    *.log 与 ranked_*.jsonl→`logs/v14/`,smoke/ 日志→`logs/v14/smoke/`);
  - `v15_ppo_20260814` → `v15`(设计文档→design/,实现与验证记录→report/,
    run_*.py|*.sh→scripts/,`eval/` 与 `v15_sft_1v3_2k/`、`v14_u510_reval_2k/`
    评测目录→eval/,*.log→`logs/v15/`)。
  整个过程只移动/重命名,不删除任何文件。
- **Rationale**: 用户要求"按版本号归位重命名、命名遵循固定类型规范";四类覆盖了
  用户列举的初始设计文档、实验报告、测试与验证脚本,并把宪法要求的 1v3 `eval`
  输出归入 `eval/` 类型;运行日志按 Decision 1 移出 audit。
- **Alternatives considered**: 类型用文件名前缀(design-*/report-*)而非子目录
  (与既有 `scripts/`、`eval/` 子目录结构不一致);把运行日志留在 audit 版本目录
  (违反日志收敛规范)。均被否。

## Decision 3: .gitignore 放行方案(用户已确认方案 A)

- **Decision**: 保持 `audit/` 整体忽略,但把固定类型子目录放行进 git:

  ```gitignore
  audit/*
  !audit/reports/
  audit/reports/*
  !audit/reports/*/
  audit/reports/*/*
  !audit/reports/*/design/
  !audit/reports/*/report/
  !audit/reports/*/scripts/
  ```

  即 `design/`、`report/`、`scripts/` 入库,`eval/` 及版本目录根散落文件继续忽略。
- **Rationale**: 用户选择方案 A;git 要求"被排除目录的父目录也必须放行"才能
  重新纳入,故逐层用 `audit/*`/`!audit/reports/` 结构;`eval/` 可能含大量 json/
  shards,不入库。
- **Alternatives considered**: 整体取消忽略(大文件入库);完全忽略(设计文档失去
  追溯)。均被否。实现时用 `git check-ignore` 对 design/eval/scripts 三类路径做
  验证。

## Decision 4: checkpoint 归档移动与中性默认

- **Decision**:
  - `checkpoints/train_riichi_ppo_v14` → `checkpoints/train_riichi_v14`(归档移动,
    更新 `v14_ppo.yaml`、`v14_ppo_resume.yaml`、bot 测试 fixture、bot README、
    `test_config_loading.py`);
  - `checkpoints/train_riichi_v13_sft` → `checkpoints/train_riichi_v13/sft`
    (归档移动,同步更新 6 个版本配置、`sft/train.py`、README、4 个测试);
    若更新后全量测试无法通过,回退为存量例外保留原名,只对新增产物强制新规范;
  - `sft/train.py` 的 `DEFAULT_CONFIG["checkpoint_dir"]` 由
    `checkpoints/train_riichi_v13_sft` 改为中性 `checkpoints/train_riichi_current`
    (与 `training.yaml` 一致);`sft.yaml` 显式指向 `train_riichi_v13/sft`。
- **Rationale**: 宪法原则 III 固定 `checkpoints/train_riichi_<版本号>`,v15 布局为
  样板;v14 冷存储仍保留;原则 VI 禁止默认值锁定历史版本。
- **Alternatives considered**: 只移动 v14、v13_sft 保留原名(不彻底,违背
  "现有目录按规范归档");删除 v14 引用(v14 冷存储的复现配置仍需指向真实路径)。
  均被否。fallback 例外已在 spec Edge Cases 声明。

## Decision 5: 数据集清理核实与 prepare.py 参数化

- **Decision**: `80pct_v11` 与 `tenhou-to-mjai` 经全盘 `find` 核实已不存在,
  无需执行删除(执行阶段向用户报告核实结果);`prepare.py --archive-dir` 由默认
  `datasets/tenhou-to-mjai` 改为必填,防止静默重建已废弃目录。
- **Rationale**: 用户要求删除前二次确认;先核实再动作,避免空操作与误删;
  原则 VI 要求默认值不锁定已废弃路径。
- **Alternatives considered**: 保留默认值(会重建已决策删除的目录);改为中性
  `datasets/raw`(与上游来源名脱节且制造新约定)。均被否。

## Decision 6: 1v3 机制常量单一来源与 CLI 默认对齐

- **Decision**: 新建 `riichi_ppo_v1/evaluation/mechanism.py` 承载四个机制常量
  (`REQUIRED_1V3_PROCESSES=10`、`DEFAULT_1V3_HANCHANS_PER_PROCESS=160`、
  `DEFAULT_1V3_INTERVAL_UPDATES=30`、`TOTAL_1V3_HANCHANS=1600`),`shards` 模块、
  `training/train.py` 与 `head_to_head_1v3.py` 统一从此导入;`head_to_head_1v3.py`
  的 `--hanchans` 默认 500→1600、`--parallel-hanchans` 默认 24→160、
  `--seed-base` 默认 20290000→0(中性),函数签名默认同步。
- **Rationale**: 机制固化、单一来源;独立 `mechanism.py` 避免 worker 反向依赖
  分片驱动模块;CLI 默认与固定机制一致,消除 500/24 的旧机制残留。
- **Alternatives considered**: 常量留在 `shards` 模块并由 runner 导入(依赖倒置);
  机制参数走配置(允许实验悄悄改机制,违背宪法 IV)。均被否。

## Decision 7: 1v3 输出落点固化

- **Decision**: 版本配置 `eval1v3_output_dir` 统一为
  `audit/reports/<版本号>/eval`(`v14_ppo*.yaml`→`audit/reports/v14/eval`,
  `v15_ppo.yaml`→`audit/reports/v15/eval`);`eval1v3.jsonl` 摘要由
  `checkpoint_dir/eval1v3.jsonl` 改为 `output_dir/eval1v3.jsonl`;
  `_progress_md_path` 改为 `output_dir.parent/report/PROGRESS.md`
  (即 `audit/reports/<版本号>/report/PROGRESS.md`),`eval1v3_output_dir` 缺省时
  跳过进度写入。
- **Rationale**: 宪法 IV"输出到 `audit/reports/<run>/eval`"+用户要求固定
  `audit/reports/<版本号>/eval`;PROGRESS.md 属实验报告类型,归 `report/`。
- **Alternatives considered**: 摘要留在 checkpoint 目录(v14/v15 历史布局,与
  "输出固定 eval 目录"不符);PROGRESS.md 留在版本根(违反固定类型子目录规范)。
  均被否。

## Decision 8: SFT 节奏单点定义

- **Decision**: 机制常量(`SFT_CADENCE_STEPS=3000`、
  `SFT_FINAL_EVAL_HANCHAN_COUNT=96`)保留在 `sft/contract.py` 作为机制源;
  `sft.yaml` 显式列出五个节奏键(`validation_interval_steps`、
  `checkpoint_interval_steps`、`heuristic_evaluation_interval_steps`、
  `heuristic_evaluation_hanchan_count`、`heuristic_evaluation_final_hanchan_count`),
  数值与常量一致;`DEFAULT_CONFIG` 继续引用常量作为无配置运行的中性兜底;
  `v15_sft_offense_warmup.yaml` 删除复制的 `validation_interval_steps: 3000` 与
  `heuristic_evaluation_enabled: true`,`v15_sft_actor_finetune.yaml` 删除复制的
  `heuristic_evaluation_enabled: true`;新增一致性测试断言 `sft.yaml` 数值等于
  常量、实验配置不含任何节奏键。
- **Rationale**: 用户要求"参数在 sft.yaml 单点定义、禁止实验配置复制";同时宪法
  II 要求实验配置自包含、不 overlay 历史配置,故"单点"= 契约常量(机制源)+
  `sft.yaml`(唯一 YAML 载体),实验配置依靠 `DEFAULT_CONFIG` 的中性兜底获得节奏,
  无需复制;一致性测试防漂移。
- **Alternatives considered**: 实验配置改为继承 `sft.yaml`(违反自包含);
  完全删除 `DEFAULT_CONFIG` 节奏兜底(破坏无配置运行)。均被否。

## Decision 9: Ray 与子进程日志收敛

- **Decision**: `training/train.py` 在 `ray.init` 前
  `os.environ.setdefault("RAY_LOG_TO_STDERR", "1")`;运行脚本以 `exec > >(tee
  logs/<版本号>/...)` 捕获全部 stdout/stderr。不改 Ray 的系统临时目录行为。
- **Rationale**: 最小侵入且与 Decision 1 的写入点清单闭合;Ray 的系统临时日志
  属进程外系统目录,不作为仓库产物治理对象。
- **Alternatives considered**: 使用 `ray.init(_logs_dir=...)`(内部接口易碎);
  接受 Ray 日志散落(违背"他处单独生成")。均被否。

## Decision 10: 文档与示例路径

- **Decision**:
  - `riichi_ppo_v1/README.md`:`--init-model` 示例改为
    `checkpoints/train_riichi_v13/sft/best_heuristic.pt`,precompute 重定向改为
    `logs/v13/sft-precompute-40pct.log`,并补充 logs/audit 目录规范段;
  - `riichi_lab_bot/README.md`:checkpoint 示例改为当前活跃产物
    `checkpoints/train_riichi_v15/ppo/checkpoint_00480.pt`,`--jsonl-log` 示例改为
    `logs/v15/bot-online.jsonl`;
  - `docs/v13_sft.md` 无日志/checkpoint 错误路径,不需改动(实测复核)。
- **Rationale**: 验收标准"文档路径与实际产物一致";示例使用真实存在的当前产物。
- **Alternatives considered**: 示例保留 v14(违反原则 VI);全部占位符(示例失去
  可执行性)。均被否。

## Decision 11: 交付切分(每主题一个 commit)

- **Decision**: 按主题顺序提交,每个 commit 测试通过:
  1) 日志写入点统一(logs/<版本号>/);
  2) audit 固定类型规范 + .gitignore 放行 + 存量归位;
  3) checkpoint 归档移动与引用同步;
  4) 数据集默认路径参数化;
  5) 1v3 输出固化与机制常量单一来源;
  6) SFT 节奏单点定义;
  7) 文档路径一致性收尾。
  破坏性操作(删 logs 存量、删数据集)与 checkpoint/audit 目录移动在对应主题前
  停下向用户确认;目录移动本身落在 git 忽略的产物目录,不进 commit,但配套引用
  更新与移动同一主题完成。
- **Rationale**: 硬性约束"每主题一个 commit、测试通过后交付";目录移动与引用
  更新同主题避免中间态。
- **Alternatives considered**: 单一大 commit(无法按主题回滚);先移动后改引用
  (产生指向不存在路径的中间态)。均被否。
