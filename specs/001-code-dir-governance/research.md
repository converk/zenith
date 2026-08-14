# Research: 代码与目录治理

本 feature 的"研究"以仓库现状盘点为主,目标是消除 spec 中的所有模糊点并给出可执行
决策。盘点基于 2026-08-15 的 `sft` 分支现状(`d50cfd8`),审计报告
`audit/reports/project_cleanup_audit_20260815/report.md` 不存在,以宪法 v1.3.0 与
仓库现状为准。

## Decision 1: 1v3 评测代码的搬迁目标

- **Decision**: 新建 `riichi_ppo_v1/evaluation/` 包,承载跨阶段确定性评测:
  `head_to_head_1v3.py`、`head_to_head_1v3_shards.py`(1v3 对抗)与
  `policy_adapter.py`(评测策略边界,含 `load_policy_adapter`)。`action_groups.py`
  是动作域分组(训练与评测共用),移入 `riichi_ppo_v1/model/`。
  `heuristic_evaluation.py` 与 `evaluation_cases.py` 是 SFT 阶段的启发式评测机制
  (宪法原则 IV 属 SFT 职责),保留在 `sft/`。
- **Rationale**: 用户明确要求 1v3 评测代码移出 `sft/`;`policy_adapter.py` 是 1v3
  与在线 bot 共用的评测边界,同属跨阶段放置;`action_groups.py` 是领域动作分组,
  与评测机制解耦后放 `model/` 最中性。
- **Alternatives considered**: 全部评测文件搬入 `evaluation/`(启发式评测是 SFT
  职责,搬迁会引入 sft→evaluation 反向依赖);1v3 文件并入 `training/evaluation/`
  (bot 会依赖 training 包,耦合更重)。均被否。

## Decision 2: `riichi-ppo-validate` 入口的新位置

- **Decision**: `riichi_ppo_v1/tests/validate.py` 移入
  `riichi_ppo_v1/tools/validate.py`;`pyproject.toml` 入口改为
  `riichi_ppo_v1.tools.validate:main`,并从打包清单移除 `riichi_ppo_v1.tests`。
- **Rationale**: `tools/` 已存在并存放独立 CLI 工具(`event_statistics.py`),
  职责一致;tests 包不再被打包安装。
- **Alternatives considered**: 新建 `riichi_ppo_v1/validation/`(单一文件不值得
  新包);留在 tests 并只改入口(违反宪法原则 I)。均被否。

## Decision 3: v11 兼容移除清单

- **Decision**: 删除 `riichi_ppo_v1/legacy/` 整包(`legacy/v11/` 5 个文件 +
  `legacy/__init__.py`);`model/architecture.py` 删除 `legacy_fixed` 头,默认头
  改为 `isolated_action_query`;`training/rewards/decision.py` 删除
  `build_legacy_v11` 与 `_legacy_v11` 分支;`evaluation/policy_adapter.py`
  (搬迁后)删除 V11 回退分支;删除 `tests/integration/test_v11_policy_adapter.py`;
  改写 `tests/unit/test_decision_analysis.py` 与 `test_sft_contract.py`。
  `checkpoints/` 与 `datasets/` 不动。
- **Rationale**: 宪法原则 II 明令放弃 v11 兼容、v11 权重冷存储保留;删除前 `rg`
  确认训练/推理路径零引用(`legacy/v11` 仅被自身、`sft/policy_adapter.py` 回退分支
  与对应测试引用)。
- **Alternatives considered**: 保留 adapter 但隔离入口(违反"彻底放弃"且留下
  双版本路径)。被否。

## Decision 4: 配置归一与加载器语义

- **Decision**: `load_config(path)` 改为:传入 `path` 时该文件必须是自包含完整
  配置,直接加载、不与打包默认叠加;不传时合并打包的 `training.yaml` +
  `monitoring.yaml` 作为当前默认。删除 `model_path`/`environment_path`/
  `training_path` 三个分组叠加参数与对应 CLI 标志(无对应配置文件,属历史残留)。
  `v14_ppo.yaml`、`v15_ppo.yaml` 补全为自包含;`v14_ppo_resume.yaml` 展平为
  `v14_ppo.yaml` 全量内容 + `resume` 覆盖 + `init_model: null`。
  `monitoring.yaml` 中的旧 `evaluation_*` 键随 Decision 5 删除。
- **Rationale**: 用户要求移除历史 full-config overlay、每个版本配置自包含;
  打包默认仅作为无版本配置时的当前默认(与 v13/v15 对齐),不再被历史配置隐式叠加。
- **Alternatives considered**: 保留组级叠加(继续允许部分配置,违背自包含要求);
  删除打包默认、强制 `--config`(破坏冒烟与最小 CPU 检查入口)。均被否。

## Decision 5: 移除旧 `evaluation_*` 评测机制

- **Decision**: 删除 `training/evaluation.py` 及其中 `should_run_evaluation`、
  `evaluation_shards`、`heuristic_evaluation_config`、`ppo_evaluation_metrics`、
  `merge_ppo_evaluation_summaries`;删除 `training/inference.py` 的
  `evaluate_heuristics` 方法与 `training/train.py` 的 `run_evaluation` 调用;
  删除 `monitoring.yaml` 的 `evaluation_*` 键;删除 `tests/unit/test_ppo_evaluation.py`
  与 `tests/unit/test_head_to_head.py`(后者引用已删除的 `sft/head_to_head.py`,
  属 2v2 旧机制)。1v3 为唯一 PPO 评测机制。
- **Rationale**: 宪法原则 IV 明确"旧 evaluation_* 机制与相关配置、代码移除,消除
  双轨",PPO 唯一评测为固定 1600 hanchan 1v3 对抗。
- **Alternatives considered**: 保留旧机制作为 fallback(双轨违背宪法)。被否。

## Decision 6: 1v3 机制常量与可变参数的边界

- **Decision**: 固定机制(10 进程、160 半庄/进程、共 1600 半庄、每 30 updates
  一次)在 `evaluation/head_to_head_1v3_shards.py` 单一命名常量定义;可变参数
  (`eval1v3_model_b`、`eval1v3_seed_base`、`eval1v3_devices`、
  `eval1v3_output_dir`)必须来自版本配置,代码不提供锁定历史版本的默认值。
  `_progress_md_path` 的 `audit/reports/v14_ppo_20260812/eval` 历史默认删除,
  `eval1v3_output_dir` 在开启 1v3 时必填,缺失即报错。
- **Rationale**: 宪法原则 IV/VI:固化机制、不固化版本;防止代码默认值锁定
  v14_ppo_20260812 等历史实验目录。
- **Alternatives considered**: 机制参数也走配置(机制可被实验悄悄修改,违背
  原则 IV"任何改动必须走宪法修订")。被否。

## Decision 7: 硬编码依赖参数化清单(全仓库)

- **Decision**: 以下默认值改为中性值或必填,不再锁定历史版本:
  - `training/train.py`: 删除 `eval1v3_seed_base=20260812`、`eval1v3_devices=("0","2")`
    的历史默认(1v3 开启时必填,devices 缺省用中性 `("0","1")`);
  - `configs/training.yaml`: `checkpoint_dir` 改为中性路径
    `checkpoints/train_riichi_current`(符合原则 III 命名型);
  - `sft/train.py` DEFAULT_CONFIG: 删除 `heuristic_evaluation_seed_base=20260717`
    (改由 `sft.yaml`/CLI 提供),固定节奏常量(3000 steps / 96 hanchan)收敛到
    单一命名常量,并修正现存的 7000 / 128 错误默认;
  - `riichi_lab_bot/cli.py`: `--checkpoint` 不再硬编码
    `train_riichi_ppo_v14/checkpoint_00510.pt`(环境变量 `RIICHI_CHECKPOINT` 或
    CLI 必填),`--seed` 默认改中性 0;
  - `riichi_lab_bot/policy.py`: 字面量 `13` 改为 `TOKEN_SCHEMA_VERSION`;
  - `sft/precompute.py`: `--output` 不再默认锁定历史编码名(必填),
    `--source` 保留现行数据集 `datasets/tenhou_sft_2024_2025`(宪法原则 III 现行
    数据集,非历史锁);
  - `tools/validate.py`(搬迁后): `--seed` 默认 20260713 改为中性 0。
- **Rationale**: 宪法原则 VI:版本号、数据集名、种子基数、默认路径等一律 CLI/配置
  传入,默认值不锁历史版本;现行数据集属于原则 III 认可的现行资产,不视为历史锁。
- **Alternatives considered**: 全部必填无默认(冒烟/最小入口大量参数必传,可用性差)。
  被否。

## Decision 8: 领域常量单一来源(241 / 34 / 136)

- **Decision**: Python 侧在 `riichi_ppo_v1/model/schema.py` 定义 `NUM_ACTIONS=241`
  与 `TILE_KINDS=34`,`model/bridge.py`、`model/architecture.py` 及所有 `sft/`、
  `training/` 模块与测试改为引用常量,删除重复定义(`bridge.py` 与
  `architecture.py` 目前各有一份 `NUM_ACTIONS=241`)。Rust 侧
  `riichienv-state-machine` 已有 `NUM_ACTIONS`,补齐对 241/34 字面量的引用替换;
  `riichienv-core` 已有 `TILES_4P=136`,`sequence_features.rs` 等改为引用。
  `RiichiEnv/src/riichienv/convert.py` 的 `136` 字面量收敛为命名常量
  `TID_COUNT`。跨语言一致性由既有协议/形状测试覆盖,不新增跨语言常量同步机制。
- **Rationale**: 宪法原则 VI 的允许例外必须"单一命名常量、单一来源,禁止散落"。
- **Alternatives considered**: 通过环境生成跨语言常量(过度工程,现协议测试已能
  捕获漂移)。被否。

## Decision 9: 幽灵引用清理清单

- **Decision**: 删除/改写以下幽灵引用: `training/learner.py` 与
  `model/architecture.py` 注释中的 `exp/training`;`riichi_ppo_v1/README.md` 的
  `encoded_10pct_v2`、`CUDA_DEVICE=0,3`、`kyokus-per-worker 1`、旧启发式评测段落;
  `docs/v13_sft.md` 的 `encoded_40pct_v13`(实际为 `_40pct_v13_v16`)与
  `legacy/v11` 描述、`kyokus_per_worker=1`;`riichi_lab_bot/README.md` 的
  `CUDA_DEVICE=2,3` 与默认 checkpoint 描述;删除引用已删模块的
  `riichi_lab_bot/tools/verify_candidate_token_drift.py`(引用
  `sft/head_to_head.py`,该模块在 `d50cfd8` 已删,工具已失效且全仓库仅注释引用)。
  `AGENTS.md` 未提交修改(删 `exp/` 引用、`CUDA_DEVICE=0,1`、`kyokus=16`)随本
  主题提交。
- **Rationale**: 宪法原则 I/III 与验收标准"rg 无幽灵路径引用";HEAD 提交删除了
  `sft/head_to_head.py` 但遗留其测试与工具引用,属本次治理应修复的既有破损。
- **Alternatives considered**: 修复 drift 工具以适配新模块(重写成本高、无运行
  验收基线,且在线 bot 已有等价测试)。被否。

## Decision 10: 冒烟清理与目录职责文档

- **Decision**: `smoke_main` 运行结束(含异常路径)后删除其产生的
  `checkpoints/riichi_ppo_v1_smoke` 与日志产物,并在 `test_cleanup_contract.py`
  增加断言;新建仓库根 `docs/directory-responsibilities.md`,为三组件每个目录
  写一句话职责。
- **Rationale**: 宪法"冒烟测试结束时必须删除其产生的日志与结果文件"与验收标准
  "每个目录职责可一句话说明"。
- **Alternatives considered**: 在各 README 分散记录(跨组件对照困难)。被否。

## Decision 11: 提交切分(每主题一个 commit)

- **Decision**: 依次提交 7 个主题,每个提交后跑对应测试:
  1) 移除 v11 兼容;2) 1v3/验证入口搬迁;3) 配置归一与移除 overlay;
  4) 移除旧 evaluation_* 机制;5) 硬依赖参数化与领域常量收敛;6) 幽灵引用与
  文档修正;7) 目录职责文档与冒烟清理收尾。删除类主题严格先 `rg` 零引用再删,
  再跑全量测试。
- **Rationale**: 宪法工作流"每主题一个 commit + 测试通过";顺序上先删 v11 再搬
  policy_adapter,避免搬迁 commit 期间 v11 adapter 相对 import 失效。
- **Alternatives considered**: 单一大 commit(不可独立回滚,违背宪法)。被否。
