# Implementation Plan: RiichiLab Bot V16 输入适配

**Branch**: `005-riichi-lab-bot-v16-input` | **Date**: 2026-08-20 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/005-riichi-lab-bot-v16-input/spec.md`

## Summary

迁移 `riichi_lab_bot` 到现行 V16 actor 输入协议。实现复用训练侧 `BatchedStateBridge.prepare_v16()`、`snapshot.py` 与 `action_query.py`;bot 只保留单席在线状态同步、缺字段补齐、安全响应与 checkpoint 推理边界。V16 SFT 与 V17 PPO checkpoint 均按 `symmetric_action_query` strict load,推理统一走 `forward_v16(policy_only=True)`。

## Technical Context

**Language/Version**: Python 3.12(conda `Mahjong-AI`)

**Primary Dependencies**: PyTorch 2.7.1、NumPy、pytest、websockets、`riichi` state-machine 扩展、`riichienv` PyO3 扩展

**Storage**: 本地 checkpoint、`logs/v16|v17/` JSONL 运行日志、spec 文档

**Testing**: pytest;本地 RiichiEnv 半庄;线上 validation 需用户提供 `RIICHI_BOT_TOKEN`

**Target Platform**: Linux + CUDA;`CUDA_DEVICE` 在入口导入 PyTorch 前映射

**Project Type**: 独立 CLI bot + 训练框架复用模块

**Performance Goals**: 单局本地推理不出现 fallback/withheld;GPU local 3 局按项目默认首轮 warmup、后两轮统计

**Constraints**: 宪法 v1.7.0;新增代码注释使用中文;不硬编码 checkpoint/版本路径;不改 PPO/SFT 训练评测机制;不接 ranked

**Scale/Scope**: 仅 `riichi_lab_bot` 运行时、bot 测试、必要的模型输入语义校验与文档更新;训练侧 V16 编码器作为权威复用

## Constitution Check

| 原则 | 检查项 | 状态 |
|------|--------|------|
| I 目录按职责组织 | bot 适配在 `riichi_lab_bot/src`,测试在 `riichi_lab_bot/tests`,通用 V16 输入校验可放 `riichi_ppo_v1/model` | 通过 |
| II 单一现行版本契约 | 使用现行 V16 输入协议;不新增旧版本兼容层 | 通过 |
| III 产物存储规范 | 日志指向 `logs/v16|v17/`;checkpoint 只读不移动不删除 | 通过 |
| IV 固定训练评测机制 | 不修改训练评测机制 | 通过 |
| V 测试基线与可观测性 | 本地测试按三轮、首轮 warmup;语义测试逐段比对 | 通过 |
| VI 通用性优先 | checkpoint 和日志路径通过 CLI/环境变量传入;测试默认路径仅为测试夹具 | 通过 |

## Project Structure

### Documentation (this feature)

```text
specs/005-riichi-lab-bot-v16-input/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── bot-v16-runtime.md
├── checklists/
│   └── requirements.md
└── tasks.md
```

### Source Code

```text
riichi_lab_bot/
├── src/riichi_lab_bot/
│   ├── bridge.py          # V16 prepare/decode 与在线字段重建
│   ├── observation.py     # 缺字段 normalization/tracker
│   ├── policy.py          # V16 checkpoint load/warmup/infer
│   ├── model.py           # 训练侧 V16 常量重导出
│   └── cli.py             # metadata/log 字段更新
├── tests/
│   ├── test_checkpoint.py
│   ├── test_bridge_semantics.py
│   └── test_bridge_integration.py
└── README.md

riichi_ppo_v1/model/
└── semantic_validation.py # 新增 V16 actor 输入断言
```

**Structure Decision**: bot 不复制训练侧 V16 编码器;所有 snapshot/query 事实计算通过训练侧模块完成,bot 仅补齐线上状态与打包单行输入。

## Design Notes

- `OnlineStateBridge.prepare()` 内部用 `BatchedStateBridge(...).prepare_v16([Decision(0, seat, observation)])`,保持训练等价。
- `PreparedDecision` 暴露 V16 三段数组与长度;旧 `token_factors/token_numeric/token_length` 字段移除,测试中假对象需同步。
- `PolicyEngine` 只接受 `symmetric_action_query`,但不检查 schema/hash/format 字段;V17 PPO format 3 与 V16 SFT 共享 strict weight path。
- `ThreatSnapshotTracker` 扩展为在线局面 tracker,按事件维护 `tiles_left`、riichi、tsumogiri、missed-agari 字段;当前完整 observation 有字段时,测试要求 tracker 值一致。
