# Implementation Plan: 代码与目录治理(Code & Directory Governance)

**Branch**: `sft` | **Date**: 2026-08-15 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-code-dir-governance/spec.md`

## Summary

对 riichi_ppo_v1、riichi_lab_bot、RiichiEnv(Python + Rust)及其测试/文档/配置执行
全仓库治理:把 1v3 评测与 `riichi-ppo-validate` 入口搬到职责目录;彻底移除 v11
checkpoint 兼容;把版本配置改为自包含并移除配置加载的 full-config overlay;删除旧
`evaluation_*` 评测双轨;参数化全仓库固定版本/路径/种子等硬依赖并把 241/34/136
领域常量收敛为单一来源;清理 `exp/` 等幽灵引用;补齐目录职责文档与冒烟清理。交付
以"每主题一个 commit + 测试通过"为单位,checkpoint 与数据集一律不动。

## Technical Context

**Language/Version**: Python >=3.10(riichi_ppo_v1)、Python 3.12
(riichi_lab_bot)、Python(PyO3/maturin)+ Rust(riichienv-core /
riichienv-state-machine,公开模块名 `riichi`)

**Primary Dependencies**: PyTorch、Ray、NumPy、PyYAML、TensorBoard、pytest、
PyO3/maturin、websockets

**Storage**: 文件系统(checkpoints/、datasets/、logs/<版本>/、
audit/reports/<版本>/;checkpoint 与数据集在本 feature 中只读)

**Testing**: pytest(riichi_ppo_v1 / riichi_lab_bot / RiichiEnv Python)、
cargo test(riichienv-core / riichienv-state-machine);性能与训练基线固定
`target_kl=0.0`、`update_epochs=4`、`kyokus_per_worker=16`、`CUDA_DEVICE=0,1`、
`learner_gpus=2`,3 轮首轮预热

**Target Platform**: Linux + CUDA(Conda 环境 `Mahjong-AI`)

**Project Type**: 多组件仓库:训练框架(CLI + library)、在线 bot(CLI)、
环境库(Python/Rust)

**Performance Goals**: 不改变训练/推理算法语义;治理前后性能与训练测试基线一致

**Constraints**: 不破坏 `riichienv-state-machine` 协议边界(公开模块名 `riichi`,
不依赖 `riichienv`);checkpoint 与数据集不删;每主题一个 commit 且测试通过

**Scale/Scope**: 三组件全部源码、测试、文档、配置;约 30 个文件移动/删除 +
引用更新 + 配置改写 + 常量收敛

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原则 | 约束 | 本设计处置 | 状态 |
|------|------|-----------|------|
| I. 目录按职责组织 | 禁止跨阶段放置;文件名自描述;中文注释;按主题 commit | 1v3/验证入口搬迁(Decision 1/2);7 主题 commit(Decision 11);新增/修改注释一律中文 | PASS |
| II. 单一现行版本契约 | 彻底移除 v11 兼容;v14 冷存储;版本配置自包含、禁 overlay | Decision 3/4:v11 全删、resume 展平、加载器去除 overlay;v14 checkpoint 不动 | PASS |
| III. 产物存储规范 | checkpoint 目录 `checkpoints/train_riichi_<版本>`;日志/报告路径规范;不删产物 | checkpoint_dir 改中性路径(Decision 7);冒烟清理只删自身产物(Decision 10);dataset 不删(遵循用户"数据集一律不删") | PASS |
| IV. 固定训练评测机制 | PPO 唯一 1v3(1600 hanchan,每 30 updates);移除旧 evaluation_*;SFT 每 3000 steps、最终 96 hanchan、单一定义 | Decision 5/6/7:删旧机制、1v3 机制常量单源、SFT 节奏常量修正为 3000/96 | PASS |
| V. 测试基线与可观测性 | 固定测试基线;3 轮首轮预热;打印耗时与指标 | 治理不触碰基线代码;文档命令对齐基线 | PASS |
| VI. 通用性优先 | 版本/路径/种子/数据集等 CLI/配置传入,默认不锁历史;241/34/136 单一来源 | Decision 7/8 参数化清单与常量收敛 | PASS |
| 附加约束 | 三组件全覆盖;CUDA_DEVICE 映射;`riichi` 模块名不依赖 `riichienv`;不引用已删目录 | 治理范围覆盖三组件;移动不改 `RiichiEnv` 协议边界;幽灵引用清理(Decision 9) | PASS |

## Project Structure

### Documentation (this feature)

```text
specs/001-code-dir-governance/
├── plan.md              # 本文件
├── research.md          # Phase 0 决策(Decision 1-11)
├── data-model.md        # 配置单元、治理主题、契约与领域常量
├── quickstart.md        # 端到端验证指南
├── contracts/           # CLI/配置契约
└── tasks.md             # $speckit-tasks 产物
```

### Source Code (repository root)

治理后目标布局(仅列出本 feature 涉及的目录):

```text
riichi_ppo_v1/
├── configs/
│   ├── training.yaml          # 打包默认(与 v13/v15 对齐,无旧 evaluation_*)
│   ├── monitoring.yaml        # 纯性能/监控组(无旧 evaluation_*)
│   ├── sft.yaml               # SFT 配置(去除与代码重复的固定节奏参数)
│   ├── v14_ppo.yaml           # 自包含
│   ├── v14_ppo_resume.yaml    # 展平为 v14_ppo.yaml 全量 + resume
│   └── v15_ppo.yaml           # 自包含
├── evaluation/                # 新增:跨阶段确定性评测
│   ├── __init__.py
│   ├── policy_adapter.py      # ← sft/policy_adapter.py(去 V11 分支)
│   ├── head_to_head_1v3.py    # ← sft/head_to_head_1v3.py
│   └── head_to_head_1v3_shards.py  # ← sft/head_to_head_1v3_shards.py
├── model/
│   ├── action_groups.py       # ← sft/action_groups.py(领域动作分组)
│   ├── schema.py              # 领域常量单一来源:NUM_ACTIONS=241、TILE_KINDS=34
│   └── architecture.py        # 仅 isolated_action_query 头
├── sft/                       # 仅 SFT 数据/训练/契约/checkpoint 职责
├── tools/
│   ├── event_statistics.py
│   └── validate.py            # ← tests/validate.py(生产验证入口)
├── training/                  # PPO 训练(无 evaluation.py 旧机制)
└── tests/                     # 仅测试;无 validate.py、无 v11/旧评测测试

riichi_lab_bot/
├── src/riichi_lab_bot/        # --checkpoint 不再硬编码 v14 默认
└── tools/                     # 删除 verify_candidate_token_drift.py(引用已删模块)

RiichiEnv/                     # 协议边界不变;136/34/241 常量收敛
docs/
└── directory-responsibilities.md  # 新增:三组件目录职责一句话清单
```

**Structure Decision**: 新增 `evaluation/` 包承载跨阶段评测;`model/` 承载动作域
与领域常量;`tools/` 承载生产 CLI。`sft/` 收窄为 SFT 职责。三组件目录职责清单放
仓库根 `docs/`,便于跨组件对照。

## Complexity Tracking

无宪法违规需要豁免;上表所有门禁均以本设计正面满足。
