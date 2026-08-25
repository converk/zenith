# Implementation Plan: V18 Input Architecture

**Branch**: `008-v18-input-architecture` | **Date**: 2026-08-25 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/008-v18-input-architecture/spec.md`

## Summary

Replace the only active input/model contract with V18: Rust-derived fixed 29-row atomic Snapshot,
query metadata with real supplier seat, structurally isolated action-pair attention, a conservative
4.9M–5.1M GQA Actor-Critic without Q modules, strict public/private information boundaries, and an
exact V18 Actor-only BC interface. Reuse history/suffix, action mapping, and Critic private tokens;
retire active V16/V17 compatibility while preserving every historical artifact.

## Technical Context

**Language/Version**: Python >=3.10 (Mahjong-AI Conda environment); Rust workspace toolchain from
`RiichiEnv/rust-toolchain.toml`

**Primary Dependencies**: PyTorch >=2.2, NumPy >=1.26, PyO3/numpy Rust bindings, RiichiEnv,
`riichi` MJAI state machine, PyYAML

**Storage**: NPZ/JSON manifest encoded samples, YAML configurations, PyTorch Actor-only artifacts,
Markdown audit evidence; no new full dataset

**Testing**: pytest, cargo test, real MJAI replay fixtures, schema/contract validation, deterministic
CPU floating-point comparison

**Target Platform**: Linux; CPU correctness tests and CUDA-capable training interface; all Python
commands through `conda run -n Mahjong-AI`

**Project Type**: Multi-package ML training system with Rust environment/state machine, PyO3 bridge,
Python model/SFT framework, and online bot consumer

**Performance Goals**: Mean total sequence length 97–103 (projected exact 99.803028); 29 Snapshot
rows for every sample; Actor-Critic 4.9M–5.1M parameters; pair permutations invariant within
`atol=1e-5, rtol=1e-5`

**Constraints**: Actor cannot see opponent hands/future wall; Critic receives three hands and exactly
five future tiles but no action queries; no Q/legacy path; no full dataset, formal SFT, or PPO work;
archived V16/V17 assets are immutable

**Scale/Scope**: Existing validation metadata covers 1,439,440 decisions in 102 shards; 241 fixed
actions; 29 Snapshot fields; changes span active model/schema/bridge/SFT/bot consumers and docs

## Constitution Check

*GATE: Passed before research and re-checked after design.*

- **Responsibility layout**: PASS. Protocol/model remain in `riichi_ppo_v1/model`; SFT interface in
  `riichi_ppo_v1/sft`; Rust environment/state derivation in `RiichiEnv`; production validation in
  `riichi_ppo_v1/tools`; audit evidence under `audit/reports/v18`.
- **Single active contract**: PASS. Design replaces active V16 paths with V18 and includes no adapter,
  fallback, checkpoint migration, or dual implementation.
- **Artifact preservation**: PASS. No checkpoint, dataset, log, historical spec, or historical report
  is deleted/overwritten. The future dataset path is registered but not materialized.
- **Evaluation governance**: PASS. PPO mechanism remains 10×400 every five updates and is untouched.
- **Test baseline/observability**: PASS. No training/performance run is planned; validation commands
  and statistics are recorded reproducibly.
- **Generality**: PASS. Domain constants have one source; operational/artifact paths enter via config
  or CLI; V18 config is self-contained.
- **Deletion gate**: PASS WITH EXECUTION CONDITION. Before deleting/renaming Q or V16 active code,
  run repository-wide `rg`, update every active caller, and pass relevant tests. Historical artifacts
  are excluded from deletion.
- **Documentation gate**: PASS WITH EXECUTION CONDITION. Protocol, READMEs, SFT docs, AGENTS.md,
  directory responsibilities, active references, and PROGRESS evidence are explicit tasks.

Post-design re-check: the data model and contracts preserve all gates; no complexity exception or
constitution amendment is required.

## Project Structure

### Documentation (this feature)

```text
specs/008-v18-input-architecture/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── v18-input-contract.md
├── checklists/
│   └── requirements.md
└── tasks.md
```

### Source Code (repository root)

```text
RiichiEnv/
├── riichienv-core/src/observation/       # 可复用供牌者事件推导与原生状态
├── riichienv-python/src/                 # Observation/Action → 现行编码 facts
├── riichienv-state-machine/src/          # 向听、atomic_snapshot.rs 与域校验
├── src/riichienv/_riichienv.pyi         # PyO3 公开类型
└── tests/                                # Rust/Python 环境与真实 MJAI 回放

riichi_ppo_v1/
├── model/
│   ├── encoding_protocol.py              # V18 schema、29 字段与域的单一来源
│   ├── snapshot.py                       # 原生 V18 Snapshot 桥接
│   ├── action_query.py                   # Query 行公共对象/映射
│   ├── architecture.py                   # 3+1 Actor、2 Critic、隔离 mask、无 Q
│   ├── bridge.py                         # V18PreparedBatch 与 Rust/Python 装配
│   ├── critic_features.py                # 三家真实手牌 + future5
│   ├── semantic_validation.py            # shape/domain/order/boundary
│   └── parameter_count.py                # 统一参数统计入口
├── sft/
│   ├── contract.py                       # 纯 V18 数据/保存契约
│   ├── data.py                           # V18 sample/collator schema
│   ├── actor_bc.py                       # Actor-only BC 与 save/load
│   ├── train.py                          # 通用 V18 CLI
│   └── precompute.py                     # 现行格式/selection 统计读取
├── tools/
│   ├── validate.py                       # 合规生产校验
│   └── v18_token_statistics.py           # 不物化的 selection 长度统计
├── configs/
│   ├── sft.yaml                          # 活跃 V18 默认
│   └── v18_sft.yaml                      # 自包含 V18 示例
├── tests/
│   ├── unit/                             # Snapshot/model/SFT/参数/非法输入
│   ├── integration/                      # bridge/replay/信息边界
│   └── protocol/                         # action/query/schema 矩阵
└── docs/                                 # V18 协议及 Rust/Python/SFT 文档

riichi_lab_bot/
├── src/riichi_lab_bot/                   # 消费单一 V18 bridge/model schema
└── tests/                                # bot V18 bridge/safety

audit/reports/v18/
├── design/
├── eval/
├── report/PROGRESS.md
└── scripts/
```

**Structure Decision**: Modify each component only within its existing responsibility. New independent
parameter counting and Actor-only BC responsibilities receive self-describing modules. Active code
uses neutral reusable names where practical; historical versioned artifacts remain untouched.

## Implementation Strategy

### Phase A: Contract and Rust facts

1. Lock the protocol version, 29-field table, domains, overflow buckets, and factor widths in one
   Rust authority exported through PyO3; build Python protocol definitions directly from that export.
2. Reuse/extract the existing last supplier actor event helper. Extend current native encoding facts
   so chi/pon/daiminkan/ron receive real self-relative source and all other actions receive N/A.
3. Derive V18 Snapshot facts in Rust from native Observation/state, including four shanten modes,
   accepted/declaration states, ankan-excluding open meld count, discard counts, latest tedashi, and
   trailing tsumogiri. Return fixed factors/numeric/lengths through PyO3.

### Phase B: Python schema, model, and boundaries

1. Replace old Snapshot tensors through sample, bridge, validation, collation, inference, rollout,
   evaluation, and bot consumers with `[B,29,4]`, `[B,29,1]`, lengths 29.
2. Make QueryEmbedding consume all metadata. Validate pair identity and allow arbitrary unique pair
   order whose action-ID set equals the legal mask.
3. Refactor forward into shared-public-only then Actor-only isolated attention; preserve action-ID
   scatter. Keep Critic sequence independent and strictly validate ordered private context.
4. Remove Q definitions/APIs/configuration after reference audit. Add the unified parameter counter
   and lock the final count/range and state-key audit.

### Phase C: Actor-only BC and active migration

1. Add explicit freeze/optimizer/forward/loss/save/load behavior for pure V18 Actor-only BC.
2. Remove legacy contract acceptance and V16 model/config field filtering. Switch active CLI/default
   config/data paths to V18 without executing formal SFT.
3. Update training/evaluation/bot call sites needed to compile and test the single active schema,
   without designing or running PPO.

### Phase D: Evidence and documentation

1. Add normal/boundary/N/A/overflow/malformed Rust, Python, protocol, integration, real replay,
   permutation, information-boundary, parameter, Q-key, and Actor-only lifecycle tests.
2. Add the read-only token-statistics CLI and record the exact existing selection result.
3. Synchronize all required docs/config/CLI paths and record every command/result in PROGRESS.md.

## Complexity Tracking

No constitution violations require justification.
