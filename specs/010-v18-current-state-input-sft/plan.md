# Implementation Plan: V18 当前局面输入与 Actor 决策架构重构

**Branch**: `010-v18-current-state-input-sft` | **Date**: 2026-08-27 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/010-v18-current-state-input-sft/spec.md`

## Summary

把现行 V18「事件历史 + 54 行 Atomic Snapshot + 局部 position Query」替换为**决策时刻状态快照**
协议：Shared 公共前缀（桌况/自身手牌/self 分析/四家 player/三家完整牌河+两摘要/当前副露/
34 tile-state）+ Actor-only 尾部（三个 Opponent Analysis + 按 action ID 升序的 O/D Query），
全 token RoPE、公共双向 GQA、结构化 Actor mask、Critic 独立尾部；`d_model=256`、16Q/4KV GQA、
`dense_slot_dim=32`、`dense_fusion_dim=512` 的密集槽位非线性融合，总参数 ≤6.0M；
Rust/PyO3 单一批编码器贯穿 replay/precompute/shard/collator/模型；Actor-only SFT 完整生命周期；
清理旧活跃契约并同步文档/审计。PPO 不修改、不兼容、不测试，单独盘点为后续迁移项。

## Technical Context

**Language/Version**: Python ≥3.10（Conda 环境 `Mahjong-AI`，Python 3.12）；Rust workspace
（rust-toolchain.toml，rustc 1.92），PyO3 + numpy 绑定，maturin 1.14。

**Primary Dependencies**: PyTorch（GPU；CPU 正确性测试），NumPy，PyYAML，RiichiEnv
（`riichienv` + `riichi` 两个 Rust/PyO3 扩展），pytest，cargo/maturin。

**Storage**: NPZ/JSON manifest encoded shards（`actor_offsets/actor_factors/actor_numeric/...`）；
YAML 自包含配置；PyTorch Actor-only 工件；Markdown audit 证据。不生成完整
`datasets/tenhou_sft_2024_2025_encoded_60pct_v18`。

**Testing**: pytest（unit/integration/protocol）、Rust `cargo test`、真实 MJAI replay fixture
（`RiichiEnv/tests/data/126_204_0_mjai.jsonl`）、合成极端 fixture、确定性 CPU 浮点比较、
GPU SFT 冒烟（`CUDA_DEVICE=0,1`，learner_gpus=2；单卡显存用 `CUDA_DEVICE=0`）。

**Target Platform**: Linux；CPU 正确性 + CUDA Training 接口。

**Project Type**: 多包 ML 训练系统（Rust 环境/状态机 + PyO3 + Python 模型/SFT + 在线 bot）。

**Performance Goals**: 编码吞吐与分段耗时测量（CPU/Rust/PyO3）；GPU Actor-only SFT 前向/反向
吞吐、显存、tokens/s；`context_tokens=256` 严格上界 ≤256；总参数 ≤6.0M；目标上下文统计
public/Actor/Critic mean/p50/p95/p99/max（合法动作重排不变性 within `atol/rtol=1e-5`）。

**Constraints**: Actor 不读对手闭手/真实摸牌/真实牌山/里宝/事后标签/Critic token；Critic 私有
输入严格为三家真实闭手 + 未来五张；无 Q/legacy 路径；不保留旧 V18 输入兼容分支或 state 迁移；
PPO worker/rollout/learner/推理/评测/性能基线不动；不生成完整数据集、不启动正式 SFT/GRP/PPO；
归档 V16/V17 资产不可变；注释中文；删除前全仓 `rg`；冒烟后清理临时产物。

**Scale/Scope**: 涉及 `riichi_ppo_v1/model`、`riichi_ppo_v1/sft`、`riichi_ppo_v1/tools`、
`riichi_ppo_v1/tests`、`RiichiEnv/riichienv-python`（新增 `current_state_encoding.rs`）、
`RiichiEnv/riichienv-state-machine`（少量 pub 化）、docs/configs/audit；PPO 路径只盘点不改。

## Constitution Check

*GATE: Passed before research and re-checked after design.*

- **I. 目录按职责**: PASS。新增 Rust 编码器放 `riichienv-python/src/current_state_encoding.rs`
  （Observation 访问 + riichi 分析内核）；模型嵌入/架构留在 `riichi_ppo_v1/model`；SFT 仍在
  `sft/`；校验在 `tools/`；生产 CLI 不进 tests。
- **II. 单一现行版本契约**: PASS。V18 仍是唯一活跃契约（协议版本号保持 18，contract hash 更新；
  不新增 V19）；删除 history/54 行 Snapshot/旧局部 position 的活跃实现；无 legacy adapter、
  旧输入转换、双模型分支、state-dict 迁移。
- **III. 产物存储**: PASS。只新增/改写代码与文档；不删除/覆盖 checkpoint、数据集、日志；
  audit 记录写入 `audit/reports/v18/report/PROGRESS.md`，脚本/统计在 `audit/reports/v18/scripts`；
  临时产物清理。
- **IV. 评测机制**: PASS。PPO 1v3 与 SFT 3000/96 节奏不动；只记录 PPO 待迁移。
- **V. 测试基线**: PASS。性能用 `CUDA_DEVICE=0,1`/learner_gpus=2；默认 3 轮、首轮预热、
  报告耗时与指标；PPO 性能基线不跑。
- **VI. 通用性优先**: PASS。新常量（row width、dense dims、context 上限、segment/kind）收敛在
  `model/encoding_protocol.py` 与 `model/schema.py` 单点；路径/版本经 CLI/config 传入；
  v18_sft.yaml 自包含（含 dense_slot_dim/dense_fusion_dim/context/rope 等）。
- **删除门**: PASS WITH EXECUTION CONDITION。删除 `snapshot.py` 等旧文件前全仓 `rg` 零引用 +
  相关测试通过；模块若还有独立职责（action_jsons/action_id 映射、241→MJAI 解码）保留在
  `bridge.py`/`action_groups.py`。
- **文档门**: PASS WITH EXECUTION CONDITION。v18_input_protocol.md 重写、KyokuEventTupleProtocol、
  v18_sft、README/AGENTS/directory-responsibilities、PROGRESS 均为显式任务；PPO 文档只加
  「待迁移」标记。

Post-design re-check：research/contract/data-model 保持全部 gate；无需宪法修订或复杂性豁免。

## Project Structure

### Documentation (this feature)

```text
specs/010-v18-current-state-input-sft/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── v18-current-state-contract.md
├── checklists/
│   └── requirements.md
└── tasks.md
```

### Source Code (repository root)

```text
RiichiEnv/
├── riichienv-python/src/
│   ├── current_state_encoding.rs   # 新增：当前局面批编码（shared+analysis rows/numeric/offsets）
│   ├── encoding_facts.rs           # 查询事实提取（复用；少量 helper pub 化）
│   └── lib.rs                      # 注册 prepare_current_state_batch + CurrentStateBatch
└── riichienv-state-machine/src/
    ├── analysis.rs                 # suji_safe/wall_class 改 pub 供新编码器复用
    └── lib.rs                      # 保持 ANALYSIS_VERSION=4 / ENCODING_PROTOCOL_VERSION=18

riichi_ppo_v1/
├── model/
│   ├── encoding_protocol.py        # 重写：segment/kind/separator/字段 schema/query 槽位（单源）
│   ├── dense_embedding.py          # 新增：DenseSlotFusion + 共享 gated MLP + 简单 concat 嵌入
│   ├── architecture.py             # 重写：ModelConfig/新前向/双向公共掩码/结构化 Actor 掩码/
│   │                               #       Critic 掩码/RoPE 连续位置/action pair 排序映射
│   ├── current_state.py            # 新增：Python 批编码包装（Rust 编码器 + query 装配 + 校验）
│   ├── bridge.py                   # 更新：PreparedBatch(actor_*/query_*/legal/critic_*) + prepare
│   ├── semantic_validation.py      # 重写：新行布局/顺序/域/摘要/tile-state/信息边界校验
│   ├── critic_features.py          # 更新：新行宽（segment 4/5, kind 13/14）
│   ├── parameter_count.py          # 更新：≤6.0M + 分项报告 + forbidden keys
│   ├── schema.py                   # 更新：行宽/数值宽/context 常量单点
│   ├── action_query.py             # 微调（行布局常量引用）
│   ├── action_groups.py            # 保留
│   ├── native_encoding.py          # 保留（查询编码调用）
│   └── snapshot.py                 # 删除（全仓 rg 零引用后）
├── sft/
│   ├── data.py                     # 重写：EncodedSample 新字段 + encode_kyoku 用新编码器
│   ├── precompute.py               # 重写：新 shard 数组/manifest 键/读写
│   ├── trainer.py                  # 更新：collate_samples/_forward_actor/长度/指标
│   ├── actor_bc.py                 # 更新：新参数根 + forward 字段
│   ├── contract.py                 # 更新：契约 payload/版本/validate_manifest
│   ├── checkpoint.py               # 更新：契约版本常量
│   └── train.py                    # 微调
├── tools/
│   ├── v18_token_statistics.py     # 重写：actor_offsets 统计 + segment 贡献
│   └── validate.py                 # 更新
├── configs/
│   └── v18_sft.yaml                # 重写：自包含（dense/context/rope/policy_head_type 等）
├── docs/
│   ├── v18_input_protocol.md       # 重写
│   ├── v18_sft.md                  # 重写
│   └── KyokuEventTupleProtocol.md  # 更新：事件仅同步，不作模型输入
└── tests/
    ├── v18_fixtures.py             # 重写：新 actor 序列构造器
    ├── unit/test_v18_encoding_protocol.py   # 重写
    ├── unit/test_v18_architecture.py        # 重写（RoPE/mask/排序/嵌入）
    ├── unit/test_v18_dense_embedding.py     # 新增（槽位敏感/内部顺序/padding/梯度/尺度）
    ├── unit/test_v18_parameter_count.py     # 更新
    ├── unit/test_v18_actor_sft.py           # 更新
    ├── unit/test_v18_sft_contract.py        # 更新（新 hash/manifest）
    ├── unit/test_v18_critic_features.py     # 更新（新行宽）
    ├── integration/test_v18_encoding_bridge.py        # 更新
    ├── integration/test_v18_replay_bridge.py           # 更新
    ├── integration/test_v18_query_semantics.py         # 更新
    ├── integration/test_v18_information_boundaries.py  # 更新
    ├── integration/test_v18_sft_lifecycle.py           # 新增：encode→precompute→shard→collate→train
    └── integration/test_v18_validation.py              # 更新
```

**Structure Decision**: 采用现有包布局与单职责文件划分；新增独立文件
`model/dense_embedding.py`、`model/current_state.py`、`RiichiEnv/riichienv-python/src/
current_state_encoding.rs`，不把 schema/融合逻辑继续堆在 `architecture.py`。

## Phases

**Phase 0（调查）**: 已完成——Rust/PyO3 全链路、Observation 字段、测试面、PPO/文档引用盘点、
扩展从源码重建成功、基线 25 测试通过。

**Phase 1（协议与 Rust 编码器）**: encoding_protocol.py 新 schema/常量 + 语义校验骨架 +
`current_state_encoding.rs`（table/self hand/self analysis/player/rivers/summaries/melds/
tile-state/opponent analysis）+ lib.rs 注册 + Rust 单测 + 重建扩展。
依赖：Phase 0。

**Phase 2（模型嵌入/架构）**: `dense_embedding.py` + 新 `architecture.py`
（RoPE/双向公共/结构化 Actor mask/Critic 掩码/action 排序映射/参数契约）+ 模型单测。
依赖：Phase 1（用真实编码行验证）。

**Phase 3（bridge/current_state 装配）**: `current_state.py` + `bridge.py`
（PreparedBatch 新字段、prepare 用新编码器、critic_features 新行宽）+ 集成测试。
依赖：Phase 1, 2。

**Phase 4（SFT 全链路）**: `sft/data.py/precompute.py/trainer.py/actor_bc.py/contract.py/
checkpoint.py` + `configs/v18_sft.yaml` + 生命周期集成测试 + token 统计。
依赖：Phase 3。

**Phase 5（清理/文档/审计）**: 删除 snapshot.py 等（rg 零引用后）；工具更新；docs/README/
AGENTS/directory-responsibilities/PROGRESS 同步；全仓一致性复核；PPO 待迁移盘点。
依赖：Phase 4。

**Phase 6（性能与收敛）**: CPU/Rust/PyO3 编码吞吐与分段耗时、GPU SFT 冒烟（3 轮，后两轮统计）、
`speckit-converge` 补齐缺口、临时产物清理。
依赖：Phase 5。

## Complexity Tracking

无需宪法违约；无表格项。
