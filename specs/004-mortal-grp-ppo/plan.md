# Implementation Plan: Mortal 式 GRP 纯奖励 PPO

**Branch**: `004-mortal-grp-ppo` | **Date**: 2026-08-19 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/004-mortal-grp-ppo/spec.md`

## Summary

把 V16 PPO 训练方案升级为下一版(V17 代际):GRP 改为 Mortal 方案(7 维输入、
2 层 GRU hidden=64、24 类全排列预测、`calc_matrix` 行列概率、validation loss
最低 checkpoint);PPO 使用纯 GRP delta reward(utility `[1, 1/3, -1/3, -1]`,
删除点差分量);rollout 每 update 收集 512 个完整半庄;2 GPU DDP +
`minibatch_size=1536/GPU`(global effective 3072,显存不足时用 gradient
accumulation);PPO 超参收敛到 `total_updates=100`、`target_kl=0.01`、
`entropy 0.01→0.005`、`critic_bootstrap_updates=2`;Q-Boosting 保留但减弱;
SFT KL 为中等偏弱长期 anchor;对手纯 current self-play;每 5 updates 保存
checkpoint 并对 V16 SFT 1v3 评测 4000 半庄,最终按 1V3 表现选择最佳 checkpoint。

## Technical Context

**Language/Version**: Python 3.11+ / PyTorch / Ray / Rust(PyO3 环境 `Mahjong-AI`)

**Primary Dependencies**: torch、ray、numpy、yaml、tensorboard;环境
`RiichiEnv`(BatchedRiichiEnv + MjaiKyokuStateMachineManager)

**Storage**:
- GRP 数据集:`datasets/tenhou_grp_2024_2025_v17/`(7 维特征 + 24 类排列标签,
  train/validation,40% 子集)
- GRP checkpoint:`checkpoints/train_riichi_v17/grp/best.pt`(validation loss 最低)
- PPO checkpoint:`checkpoints/train_riichi_v17/ppo/checkpoint_00NNN.pt`
- 日志:`logs/v17/<run>.log`;评测:`audit/reports/v17/eval/`
- TensorBoard:`checkpoints/train_riichi_v17/ppo/tensorboard`

**Testing**: pytest(单测 + 集成冒烟);`riichi_ppo_v1/tests/unit/` 增加
`test_grp_mortal.py`、`test_v17_reward.py` 等;冒烟用
`python -m riichi_ppo_v1.training.train --smoke` 变体或直接短迭代运行。

**Target Platform**: Linux 服务器(CUDA 双卡,CUDA_DEVICE=0,1 → 物理 GPU 0,1)

**Project Type**: 机器学习训练框架(`riichi_ppo_v1` 包)

**Performance Goals**: 512 半庄/update × 100 updates;每 GPU minibatch 1536;
每 5 updates 一次 4000 半庄 1v3 评测；GPU 显存峰值需在当前双卡内受控。

**Constraints**:
- 不做无关重构,保持现有工程结构;`RiichiEnv` 与状态机不修改。
- 领域常量单一来源;配置自包含,不 overlay。
- 1v3 机制常量与 `mechanism.py` 单一来源;本方案要求 4000 半庄/次、每 5
  updates,必须经宪法原则 IV 修订。
- 冒烟测试结束必须删除其产生的日志与结果文件。
- 代码注释一律中文;删除文件前全仓 `rg` 零引用检查。

**Scale/Scope**: 1 个 GRP 离线训练入口 + 1 个 PPO 训练入口 + 1 个 1v3 评测
入口 + checkpoint 选择脚本;不新增算法(仅配置与机理调整)。

## Constitution Check

*GATE: 需先经 `$speckit-constitution` 修订(MAJOR/MINOR 按语义)后进入实施。*

需要修订的条款:
1. **原则 IV(固定训练评测机制)**:PPO 1v3 从「1600 半庄、每 30 updates」改为
   「4000 半庄、每 5 updates」;进程数保持 10,单进程半庄数 160→400;这是硬性
   要求,不能通过实验配置悄悄修改。
2. **原则 II(单一现行版本契约)**:登记 v17 为下一实验代(活跃实验代 v15→v17,
   协议编码仍 v16);现行数据集新增 GRP v17 数据集。
3. **产物目录**:checkpoint 固定目录 `train_riichi_v17/`,日志 `logs/v17/`,
   报告 `audit/reports/v17/`。

## Project Structure

### Documentation (this feature)

```text
specs/004-mortal-grp-ppo/
├── plan.md              # 本文件
├── research.md          # Mortal GRP 方案调研结论(视频/源码)
├── data-model.md        # GRP 输入/输出契约、24 类排列映射、calc_matrix 定义
├── quickstart.md        # GRP 训练、PPO 启动、评测命令
├── contracts/           # GRP 协议文档(输入/输出契约与 calc_matrix)
└── tasks.md             # /speckit-tasks 输出
```

### Source Code (repository root)

```text
riichi_ppo_v1/
├── model/
│   └── grp.py                    # 重写:GRPModel(Mortal 结构)+ calc_matrix
├── training/
│   ├── train.py                 # rollout 半庄数停止条件、5-update checkpoint/
│   │                            #   4000 半庄评测、best 选择、smoke 扩展
│   ├── learner.py               # gradient accumulation 支持(可选开关)
│   ├── learner_ddp.py           # 分片对齐适配 1536/GPU
│   ├── worker.py                # GrpRollout 改纯 GRP reward;半庄计数停止
│   └── grp/
│       ├── prepare.py           # 新 7 维特征 + 24 类标签 + v17 数据集
│       ├── train.py             # batch 512 / lr 1e-5 / val-loss best ckpt
│       └── reward.py            # 纯 GRP reward(utility [1,1/3,-1/3,-1])
├── configs/
│   ├── v17_grp.yaml             # GRP 自包含配置
│   └── v17_ppo.yaml             # PPO 自包含配置(512 半庄/3072 batch/4000 评测)
├── evaluation/
│   ├── mechanism.py             # 修订:DEFAULT_1V3_HANCHANS_PER_PROCESS=400
│   │                            #   DEFAULT_1V3_INTERVAL_UPDATES=5(宪法修订)
│   └── select_best_checkpoint.py # 按 1V3 vs SFT 表现选择最佳 checkpoint
└── tests/unit/
    ├── test_grp_mortal.py       # GRP 输入/24 类/calc_matrix/冻结
    ├── test_v17_reward.py       # 纯 GRP reward 公式
    └── test_v17_ppo_config.py   # 512 半庄/3072 batch/配置值断言
```

扩展说明(保持既有路径):
- `model/grp.py` 原地替换为 Mortal 结构(同一文件,避免新增文件破坏引用)。
- `training/grp/{prepare,train,reward}.py` 原地重写输入契约与 reward。
- `training/worker.py` 的 `GrpRollout` 直接改纯 GRP reward,删除 score 分量。
- `training/train.py` 保持主循环,新增:半庄数停止、5-update 评测节奏、
  best-checkpoint 选择;`run_1v3_evaluation` 复用 shards 机制。
- `evaluation/mechanism.py` 常量经宪法修订后更新(single source)。

## Implementation Phases

### Phase 0: 调研(Mortal 源码)

- `research.md`:记录 Mortal GRP 的 feature 构造(7 维)、模型结构(GRU(64,2) →
  fc)、训练(AdamW、batch 512、lr 1e-5?)、`calc_matrix` 语义、reward calculator
  公式(pts [3,1,-1,-3] 归一化到 [1,1/3,-1/3,-1] 等价)。
- 校验 tie-break 排序:按分数降序、同分按座位号稳定。

### Phase 1: 设计

- `data-model.md`:GRP 特征 schema(7 列)、24 类 permutation 映射(从 4! 生成并
  固化)、calc_matrix 推导、reward 公式与归一化约定、GRP 数据集格式(chunk npz
  沿用)。
- `contracts/grp-v17.md`:输入/输出契约、`calc_matrix` 矩阵语义、冻结契约。
- `quickstart.md`:GRP prepare/train 命令、PPO 启动命令、评测与 best 选择命令。

### Phase 2: 实施顺序(与 tasks.md 对齐)

1. 宪法修订(原则 IV 4000/5;原则 II v17 登记)。
2. GRP 重写(model/grp.py + prepare + train + 数据集 prepare 支持)。
3. reward 纯 GRP 重写(worker GrpRollout + reward.py + tests)。
4. rollout 512 半庄停止 + learner gradient accumulation + DDP 对齐。
5. PPO 配置 v17 + 5-update checkpoint + 4000 半庄 1v3 + best 选择。
6. TensorBoard 监控延续(无需新键,验证现有键)。
7. 全量测试 + 冒烟 + README/docs 同步。

## Known Risks

- **GRU 训练慢于 V16 Linear-embedding GRP**:GRU 前向不可并行(seq 维度),但
  7 维输入极小,CPU 推理在边界处仅 4 次/局,风险可控;训练在 GPU batch 512。
- **1536/GPU 显存**:V16-small 模型 3M 参数,batch 1536 的 padding 峰值
  ~长序列组合;已规划 gradient accumulation 兜底。
- **4000 半庄评测耗时**:10 进程 × 400,约等于原 1600 评测的 2.5 倍;每 5
  updates 一次,100 updates 共 20 次,总评测开销需纳入实验预算。
- **best-checkpoint 指标噪音**:4000 半庄的 point_diff CI 仍非 0;选择规则固定
  为 point_diff 均值,必要时参考 mean_rank 兜底。