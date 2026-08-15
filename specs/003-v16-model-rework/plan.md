# Implementation Plan: V16 模型重构与训练

**Branch**: `V16`(实现分支;无 before_plan 钩子,不自动创建分支,spec 目录名与
分支相互独立)| **Date**: 2026-08-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-v16-model-rework/spec.md`

## Summary

按设计文档重构 V16:输入编码收敛为单一新协议 v16(Objective Facts + Compact
Snapshot + 每合法动作一对 10-slot Offense/Defense Query,删除全部 Derived
Features 与牌河/副露重复表示);网络扩容为 d_model=256/16Q/4KV/FFN=1088/4
shared+1 actor+2 critic(总参数 7.5–7.8M),策略头 Offense/Defense 对称融合、删除
zero-init;Critic 保留特权输入(三家手牌 + 后 5 张牌山);PPO 奖励改为 70% 归一化
GRP delta + 30% 归一化小局分差(utility [12,4,-6,-10],σ 离线固定),冻结轻量 GRP
(50–70K)并采用 Top-3 Q-boosting(训练候选 Top-3∪行为动作 ≤4、动作表示 detach);
V16 SFT 从头重编码训练(Recall@3 前置检查);同步完成协议文档、宪法 Principle II
修订、旧代码零引用清理与产物治理。

## Technical Context

**Language/Version**: Python 3.12(conda `Mahjong-AI`);Rust(`riichienv-core` +
`riichienv-state-machine`,PyO3 绑定,公开模块名 `riichi`)

**Primary Dependencies**: PyTorch、Ray、NumPy、PyYAML、pytest;maturin/PyO3(Rust
侧);tensorboard

**Storage**: 本地文件系统——`datasets/`(新编码 `…_encoded_40pct_v16`、GRP
`tenhou_grp_2024_2025_v16`)、`checkpoints/train_riichi_v16/{sft,grp,ppo}`(均含
配置快照)、`logs/v16/`、`audit/reports/v16/{design,eval,report,scripts}`;旧
checkpoint 与数据集只保留不删除

**Testing**: `python -m pytest`(unit/integration/protocol)+ `cargo test`
(RiichiEnv);语义正确性硬门槛(独立 oracle 逐 slot 比对);性能基线
target_kl=0.0、update_epochs=4、kyokus_per_worker=16、CUDA_DEVICE=0,1、
learner_gpus=2,3 轮首轮预热

**Target Platform**: Linux + CUDA 服务器(CUDA_DEVICE 映射 0→GPU0、1→GPU1;
启动 PyTorch/Ray 前完成映射)

**Project Type**: 训练/评测 CLI 框架(riichi_ppo_v1)+ Rust 游戏环境 + PyO3 绑定
(RiichiEnv)

**Performance Goals**: 不降低 SFT/PPO 吞吐;GRP 仅在小局边界执行(参数 50–70K,
可忽略);Top-3 Q 计算量 ≤4 候选/决策;总参数 7.5–7.8M、Actor 推理约 5.3M

**Constraints**: 宪法 v1.3.0 生效,Principle II 需经 `$speckit-constitution`
修订为「信息编码协议 v16」并记录 Sync Impact Report;新增代码注释一律中文;删除
前全仓 rg 零引用+测试通过、每主题一个 commit;checkpoint 与数据集不删除(v11
权重、v14 资产冷存储);评测机制常量不改动;版本/对手/种子/路径经 CLI/配置传入;
每个版本配置自包含、禁止 overlay

**Scale/Scope**: 全仓三组件;SFT 40% 重编码约 94M train / 96 万 validation 决策;
GRP 每半庄 4 视角;领域常量单一来源(136 TID/34 牌类/241 动作维/各 slot 基数)

## Constitution Check

*GATE: 通过后进入 Phase 0;Phase 1 设计完成后复查。*

| 原则 | 检查项 | 状态 |
|------|--------|------|
| I 目录按职责组织 | 模型/契约进 `model/`,GRP 数据与训练进 `training/grp/`,协议文档进
`docs/`,语义测试进 `tests/`(integration/protocol);新增文件先核对
`docs/directory-responsibilities.md`;中文注释;每主题一 commit | ✓ 计划通过 |
| II 单一现行版本契约 | 输入编码单一协议 v16;v13 契约与多版本字段随零引用清理移除;
Principle II 经 `$speckit-constitution` 修订(预期 1.3.0→1.4.0)后代码生效;配置
自包含无 overlay | ✓ 计划通过(含计划内宪法修订) |
| III 产物存储规范 | 数据集 `…_encoded_40pct_v16` 与 `tenhou_grp_2024_2025_v16`;
checkpoint `checkpoints/train_riichi_v16/{sft,grp,ppo}` 含配置快照;日志
`logs/v16/`;报告 `audit/reports/v16/` 四类型 + `report/PROGRESS.md`;只归档移动
不删除 | ✓ 计划通过 |
| IV 固定训练评测机制 | 1v3 沿用 `evaluation/mechanism.py` 常量(10×160=1600、
每 30 updates,对手/种子/输出目录由 v16 配置提供);SFT 节奏沿用 `sft.yaml`
3000/96 单点定义;不在实验配置复制 | ✓ 计划通过 |
| V 测试基线与可观测性 | 性能/训练测试固定 target_kl=0.0、update_epochs=4、
kyokus_per_worker=16,CUDA_DEVICE=0,1、learner_gpus=2,3 轮首轮预热、后两轮单独
报告,冒烟结束删除产物;语义 oracle 测试为硬门槛 | ✓ 计划通过 |
| VI 通用性优先 | 协议版本、checkpoint/数据集/对手/种子/间隔/路径一律 CLI/配置;
136/34/241 与 slot 基数收敛单一命名常量;固化机制不改 | ✓ 计划通过 |
| Quality Gates | 删除前 rg 零引用+测试通过;协议文档与实现同步;README/docs/AGENTS
路径同步;冒烟清理 | ✓ 计划通过 |

无需要 Complexity Tracking 备案的违规;Principle II 的修改是任务约束②明确要求的
宪法修订(经 `$speckit-constitution` 完成并记录 Sync Impact Report),不是对宪法的
静默偏离。

## Project Structure

### Documentation (this feature)

```text
specs/003-v16-model-rework/
├── plan.md                  # 本文件
├── research.md              # Phase 0:现状盘点与 R1–R12 决策
├── data-model.md            # Phase 1:七类实体与跨实体约束
├── quickstart.md            # Phase 1:七个验证场景
├── contracts/
│   ├── actor-input-v16.md   # 输入结构、20 slot 语义/基数/N-A/终局约定、不变量
│   ├── grp-v16.md           # GRP 模型、输入、训练与冻结契约
│   ├── reward-v16.md        # utility、归一化、70/30、Top-3 Q-boost 契约
│   └── sft-dataset-v16.md   # 数据集命名、manifest、40% 划分、GRP 统计量
├── checklists/
│   └── requirements.md
└── tasks.md                 # Phase 2($speckit-tasks 生成,不由本命令创建)
```

### Source Code (repository root)

```text
riichi_ppo_v1/
├── model/
│   ├── encoding_protocol.py      # 新增:协议 v16 单一版本 + slot 语义/基数单一来源
│   ├── snapshot.py               # 新增:基础场况/Score Pressure/3×7 对手摘要编码
│   ├── action_query.py           # 新增:每动作一对 10-slot query 的取值与单 token 聚合
│   ├── grp.py                    # 新增:GRP 网络与输入/损失常量(50–70K)
│   ├── architecture.py           # 修改:v16 preset、对称策略融合、Top-3 Q scorer;
│   │                              # 删除 offense_fusion zero-init 与 241 维 q_head
│   ├── schema.py                 # 修改:协议版本指向 encoding_protocol 单一常量
│   ├── bridge.py                 # 修改:装配新 snapshot+query;特权信息只进 critic
│   ├── critic_features.py        # 修改:仅保留对手手牌 + 后续 5 牌山;删除公开汇总
│   ├── feature_schema.py         # 删除:v13 多版本契约(零引用后)
│   └── actor_features.py         # 删除:v13 六行摘要(零引用后)
├── training/
│   ├── grp/
│   │   ├── prepare.py            # 新增:GRP 数据集构造(4 视角、prefix 标签、σ 固化)
│   │   ├── train.py              # 新增:GRP 离线训练入口(冻结后存 checkpoint)
│   │   └── reward.py             # 新增:utility、GRP delta、归一化组合
│   ├── rewards/
│   │   ├── terminal.py           # 修改:移除独立半庄排名奖励分量(排名效用并入
│   │   │                          # GRP 终局 V_terminal=U(rank))
│   │   ├── efficiency.py         # 候选删除:效率奖励被 GRP+分差取代(零引用后)
│   │   └── decision.py           # 候选删除/瘦身:v13 候选分析与奖励逻辑(零引用后)
│   ├── learner.py                # 修改:Top-3 Q loss/boost、GRP 奖励装配、冻结检查
│   ├── worker.py                 # 修改:小局边界 GRP 推理、奖励计算
│   ├── train.py                  # 修改:v16 配置键;1v3 常量仍从 mechanism 导入
│   └── inference.py              # 修改:新编码前向与 Top-3 输出
├── sft/
│   ├── precompute.py             # 修改:v16 编码器接入、manifest 单版本契约
│   ├── contract.py               # 修改:validate_v16_manifest(单协议版本 + 契约哈希)
│   ├── data.py                   # 修改:manifest 校验切换 validate_v16_manifest
│   ├── train.py                  # 修改:v16 输入、Recall@3 记录、checkpoint 路径
│   └── prepare.py                # 复用:原始 tar shard 流水线不变
├── configs/
│   ├── v16_sft.yaml              # 新增:自包含;节奏键不得复制(sft.yaml 单点)
│   ├── v16_grp.yaml              # 新增:自包含 GRP 训练配置
│   └── v16_ppo.yaml              # 新增:自包含;1v3/节奏键不复制
├── docs/
│   └── v16_input_protocol.md     # 新增:V16 输入协议(与实现同步)
└── tests/
    ├── integration/test_v16_query_semantics.py   # 新增:独立 oracle 逐 slot 比对
    ├── integration/test_v16_replay_bridge.py     # 新增:回放/桥接一致性
    ├── protocol/test_v16_cardinalities.py        # 新增:基数/N-A/bucket/终局约定
    ├── unit/test_v16_architecture.py             # 新增:参数量/对称融合/无 zero-init
    ├── unit/test_v16_q_scorer.py                 # 新增:候选集 ≤4/detach
    └── unit/test_grp.py                          # 新增:旋转/prefix/冻结/参数量

RiichiEnv/
├── riichienv-core/src/
│   └── offense_analysis.rs       # 新增:等待/有无役/基础番数/振听/可立直 + PyO3;
│                                 # 复用 shanten.rs、HandEvaluator、yaku
├── riichienv-state-machine/src/
│   └── analysis.rs               # 扩展:D0–D9(现物/筋/公开数/安全牌库存)、
│                                 # 对手 7 项摘要、visible count
└── tests/                        # 扩展:新分析函数的 Rust 语义测试
```

**Structure Decision**: 新组件一律放入既有职责包——模型与协议契约进
`riichi_ppo_v1/model/`,GRP 数据/训练/奖励进 `training/grp/`(PPO 训练期机制),
语义与协议测试进 `tests/integration|protocol`;手牌规则评价进
`riichienv-core`,公开状态事实进 `riichienv-state-machine`(公开模块名 `riichi`,
无反向依赖)。删除清单以全仓 rg 零引用为准,按主题分 commit。
