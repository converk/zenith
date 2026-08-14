# Implementation Plan: 产物存储与评测机制固化

**Branch**: `002-artifact-storage-eval`(工作分支 `sft`)| **Date**: 2026-08-15
| **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-artifact-storage-eval/spec.md`

## Summary

在三个组件(riichi_ppo_v1、riichi_lab_bot、RiichiEnv)上固化产物存储与评测机制:
日志收敛到 `logs/<版本号>/`(清空存量,用户已决策);审计产物收敛到
`audit/reports/<版本号>/` 的 design/report/eval/scripts 固定类型(存量按版本归位、
只移动不删除);checkpoint 统一 `checkpoints/train_riichi_<版本号>` 布局(v14 与
v13_sft 归档改名、零删除);清理两个已核实不存在的废弃数据集并参数化
`prepare.py --archive-dir`;PPO 只保留固定 1v3 机制(10×160=1600、每 30 updates,
常量单一来源,对手参数化,输出固定 `audit/reports/<版本号>/eval`);SFT 节奏 3000
steps/96 hanchan 在 `sft.yaml` 单点定义;全仓库默认值与文档路径与实际产物一致。

## Technical Context

**Language/Version**: Python 3.12(conda `Mahjong-AI`);Rust(RiichiEnv crates)

**Primary Dependencies**: PyTorch、Ray、NumPy、PyYAML、pytest;Rust 侧 riichienv-core
与 riichienv-state-machine(公开模块名 `riichi`)

**Storage**: 本地文件系统目录——`checkpoints/`、`datasets/`、`logs/<版本号>/`、
`audit/reports/<版本号>/`;checkpoint 与数据集在 git 忽略之外由磁盘保留

**Testing**: `python -m pytest`(三组件 Python)+ `cargo test`(RiichiEnv Rust);
宪法测试基线 `target_kl=0.0`、`update_epochs=4`、`kyokus_per_worker=16`,
`CUDA_DEVICE=0,1`、`learner_gpus=2`,3 轮首轮预热

**Target Platform**: Linux + CUDA 服务器(conda 环境 `Mahjong-AI`)

**Project Type**: 训练/评测 CLI 框架(riichi_ppo_v1)、在线 bot CLI
(riichi_lab_bot)、Rust 游戏环境(RiichiEnv)

**Performance Goals**: 不改变训练/评测吞吐;1v3 固定 1600 hanchan(10 进程 × 160)
在现有分片实现上运行

**Constraints**: checkpoint 只归档移动禁止删除;日志/数据集删除须用户确认;
每主题一个 commit 且测试通过;版本/对手/种子/路径默认值不得锁定历史版本;
新增注释一律中文

**Scale/Scope**: 三个组件全量;`logs/` 存量约 127MB;现行数据集约 19.6GB;
`audit/reports` 三个版本目录归位

## Constitution Check

*GATE: 通过后进入 Phase 0;Phase 1 设计完成后复查。*

| 原则 | 检查项 | 状态 |
|------|--------|------|
| I 目录按职责组织 | 新增 `evaluation/mechanism.py`(机制常量单一来源)与 `tests/unit/test_artifact_conventions.py`(约定契约测试)放入既有对应包;中文注释;每主题一 commit | ✓ 计划通过 |
| II 单一现行版本契约 | 不新增 overlay;v14 冷存储仅归档改名;v13_sft 改名同步引用,测试失败回退存量例外;实验配置保持自包含 | ✓ 计划通过 |
| III 产物存储规范 | `logs/<版本号>/`、`audit/reports/<版本号>/` 四类型、`checkpoints/train_riichi_<版本号>`、废弃数据集清除且不复活、CLI 默认与文档一致 | ✓ 计划通过 |
| IV 固定训练评测机制 | 1v3 唯一机制常量单一来源、对手参数化、输出 `audit/reports/<版本号>/eval`;SFT 3000/96 单点定义 | ✓ 计划通过 |
| V 测试基线与可观测性 | 保持既有性能/训练测试基线不动;本 feature 只改路径与节奏定义 | ✓ 计划通过 |
| VI 通用性优先 | 机制固化、版本/对手/种子/路径参数化;默认值中性(`train_riichi_current`、seed 0、`--archive-dir` 必填、`--report` 无默认落盘) | ✓ 计划通过 |
| Quality Gates | 移动前全仓库引用检查;冒烟测试清理产物;README/docs/AGENTS 路径同步 | ✓ 计划通过 |

无需要 Complexity Tracking 备案的违规。

## Project Structure

### Documentation (this feature)

```text
specs/002-artifact-storage-eval/
├── plan.md              # 本文件
├── research.md          # Phase 0:现状盘点与 11 项决策
├── data-model.md        # Phase 1:六类实体与跨实体约束
├── quickstart.md        # Phase 1:六个验证场景
├── contracts/
│   ├── storage-layout.md    # logs/audit/checkpoints/datasets/.gitignore 契约
│   └── eval-mechanism.md    # 1v3 与 SFT 节奏契约
└── tasks.md             # Phase 2($speckit-tasks 生成)
```

### Source Code (repository root)

```text
riichi_ppo_v1/
├── evaluation/
│   ├── mechanism.py                 # 新增:1v3 机制常量单一来源(10/160/1600/30)
│   ├── head_to_head_1v3.py          # CLI/函数默认对齐机制常量(1600/160/seed 0)
│   └── head_to_head_1v3_shards.py   # 常量改为从 mechanism.py 导入
├── training/
│   └── train.py                     # RAY_LOG_TO_STDERR;eval1v3.jsonl→eval;
│                                    # PROGRESS.md→report;常量导入改 mechanism
├── sft/
│   ├── train.py                     # DEFAULT_CONFIG checkpoint_dir 中性化
│   ├── audit.py                     # --report 默认 None(不落盘)
│   └── prepare.py                   # --archive-dir 必填
├── configs/
│   ├── sft.yaml                     # 显式节奏键(3000/96)
│   ├── v15_sft_*.yaml               # 删除复制的节奏/开关键
│   ├── v14_ppo*.yaml, v15_ppo.yaml  # checkpoint/输出路径按新规范
│   └── training.yaml                # 已中性(train_riichi_current),不动
└── tests/
    ├── unit/test_artifact_conventions.py   # 新增:布局/机制/默认值契约测试
    ├── unit/test_cleanup_contract.py       # sft checkpoint 断言更新
    ├── unit/test_config_loading.py         # v14 路径断言更新
    ├── unit/test_learner.py                # v13_sft 引用更新
    ├── unit/test_sft_tensorboard.py        # v13_sft 引用更新
    └── integration/test_v13_sft_golden.py  # CHECKPOINT 路径更新

riichi_lab_bot/
├── tests/conftest.py               # v14 路径更新
└── README.md                       # checkpoint/jsonl-log 示例更新

audit/reports/
├── v13/   # ← v13_sft_20260802(design/report/eval/scripts 归类)
├── v14/   # ← v14_ppo_20260812(含 eval/ 归并;日志移出)
└── v15/   # ← v15_ppo_20260814(含 eval/ 归并;日志移出)

logs/
├── v13/ v14/ v15/   # 新写入点;根目录清空

checkpoints/
├── train_riichi_v13/sft   # ← train_riichi_v13_sft(归档移动)
├── train_riichi_v14       # ← train_riichi_ppo_v14(归档移动)
└── train_riichi_v15       # 样板布局,不动

.gitignore                   # audit 放行规则(方案 A)
```

**Structure Decision**: 不新建顶级包;机制常量进 `riichi_ppo_v1/evaluation/`,
布局契约测试进 `riichi_ppo_v1/tests/unit/`;产物目录移动遵循宪法 III 的命名型,
不引入额外抽象层。

## Complexity Tracking

无违规,无需备案。
