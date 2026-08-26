# Implementation Plan: V18 GRP 输入扩展与重新训练

**Branch**: `009-v18-grp` | **Date**: 2026-08-26 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/009-v18-grp/spec.md`

## Summary

以 21 维边界状态扩展 GRP(全局排名预测)输入:在 V17 的 7 维
`[grand_kyoku, honba, kyotaku, s0..s3/1e4]` 之上新增局风类型、上一小局结果类型与
各玩家累计和了/放铳/听牌流局次数(4×3);模型结构从 7→64×2 层 GRU 提升为
21→96×2 层 GRU(fc 192→192→24,约 13.2 万参数);训练数据从 40%×280 shard
(4.34 万半庄)扩到**全量数据** × 全部 930 shard(约 36 万半庄),与 SFT 60%
子集完全重叠(GRP 为冻结奖励模型,仅评分用)。GRP 训练后完全冻结,纯 GRP delta reward 契约与 24 类排列标签不变。

## Technical Context

**Language/Version**: Python >=3.10(Mahjong-AI Conda 环境);PyTorch >=2.2、NumPy >=1.26

**Storage**: NPZ 数据集 `datasets/tenhou_grp_2024_2025_v18`(offsets/features/rank_by_player/
years/game_ids)、YAML 配置、PyTorch 冻结 checkpoint、Markdown 审计证据

**Testing**: pytest(契约/一致性/小数据集全链路)、临时目录 prepare 冒烟

**Target Platform**: Linux;CPU 正确性测试 + CUDA 训练接口;Python 命令经
`conda run -n Mahjong-AI`

## Design Decisions

- **DD-1 输入契约单一来源**:字段顺序常量与布局描述在 `model/grp.py`;离线
  (prepare)与在线(worker/GrpRollout)共用 `feature_row` 纯函数,累计计数由
  调用方维护(离线按边界链推进,在线按逐个 `boundary.previous` 推进),
  保证两条路径逐位一致。
- **DD-2 局风类型**:离线由整局 bakaze 集合的 max 风映射(0/1/2);在线由
  `game_mode` 字符串后缀映射;未知模式 fail-closed 抛 ValueError。
- **DD-3 模型可配置**:GRPModel 构造参数带默认常量;PPO 加载时优先读
  checkpoint `model_config`,避免形状耦合硬编码。
- **DD-4 版本化产物**:数据集/配置/checkpoint/脚本/文档全部使用 v18 显式路径,
  不 overlay、不继承;v17 产物只读保留。
- **DD-5 超参**:epochs=30、batch=2048(4× 批,LR 不线性放大)、lr=1e-5、
  val 每 200 步,沿用 v17 配方;全量数据下步数约 5.6 万量级。

## Phase 1 — 契约与模型

- [ ] P1.1 更新 `riichi_ppo_v1/model/grp.py`:21 维输入常量与布局、GRP_HIDDEN=96、
  GRPModel 可配置构造(input_size/hidden_size/num_layers/num_classes)。
- [ ] P1.2 更新 `riichi_ppo_v1/training/grp/prepare.py`:`game_type_from_content`
  /`game_type_from_mode`、`result_increment`、`feature_row`、
  `features_from_boundaries(boundaries, game_type)`,数据集 format
  `riichi-grp-v18` + 局风分布统计;`pyproject`/`__init__` 无需改动。
- [ ] P1.3 更新 `riichi_ppo_v1/training/grp/train.py`:快照使用 GRP_* 常量,
  去硬编码 64/2;输出 model_config 含 input_size/hidden/layers/feature_layout。

## Phase 2 — 运行时装配

- [ ] P2.1 更新 `riichi_ppo_v1/training/worker.py`:`GrpRollout(model, game_type)`
  每环境维护累计计数,用 `feature_row` 构造前缀行;GRP 加载按 checkpoint
  `model_config` 构造;注释更新参数规模。

## Phase 3 — 配置与脚本

- [ ] P3.1 新增 `riichi_ppo_v1/configs/v18_grp.yaml`(自包含)。
- [ ] P3.2 新增 `audit/reports/v18/scripts/run_v18_grp_prepare_and_train.sh`:
  prepare(denominator=6、remainders=0,1,2,3、全部 shard;`--workers` 默认 6
  进程并行解析)→ train,日志落 `logs/v18/`,支持 `--skip-prepare`。

## Phase 4 — 测试与验证

- [ ] P4.1 更新 `tests/unit/test_grp_mortal.py`(21 维、参数预算 110K–150K、
  累计计数、局风映射、离线/在线一致性)。
- [ ] P4.2 更新 `tests/unit/test_v17_reward.py` 的 GrpRollout 构造调用。
- [ ] P4.3 运行 `pytest riichi_ppo_v1/tests` 全绿;临时目录 prepare+train 冒烟。
- [ ] P4.4 文档同步:`riichi_ppo_v1/docs/v18_grp.md`、`audit/reports/v18/design/`
  设计文档、`audit/reports/v18/report/PROGRESS.md`;`git diff --check` 通过。
