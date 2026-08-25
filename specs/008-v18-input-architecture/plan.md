# Implementation Plan: V18 Input Architecture

**Branch**: `008-v18-input-architecture` | **Date**: 2026-08-25 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/008-v18-input-architecture/spec.md`

## Summary

Replace the only active input/model contract with V18: Rust-derived fixed 54-row atomic Snapshot,
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

**Performance Goals**: Mean total sequence length 121–127 (projected exact 124.803028); 54 Snapshot
rows for every sample; Actor-Critic 4.9M–5.1M parameters; pair permutations invariant within
`atol=1e-5, rtol=1e-5`

**Constraints**: Actor cannot see opponent hands/future wall; Critic receives three hands and exactly
five future tiles but no action queries; no Q/legacy path; no full dataset, formal SFT, or PPO work;
archived V16/V17 assets are immutable

**Scale/Scope**: Existing validation metadata covers 1,439,440 decisions in 102 shards; 241 fixed
actions; 54 Snapshot fields; changes span active model/schema/bridge/SFT/bot consumers and docs

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
│   ├── encoding_protocol.py              # V18 schema、54 字段与域的单一来源
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

1. Lock the protocol version, 54-field table, domains, overflow buckets, and factor widths in one
   Rust authority exported through PyO3; build Python protocol definitions directly from that export.
2. Reuse/extract the existing last supplier actor event helper. Extend current native encoding facts
   so chi/pon/daiminkan/ron receive real self-relative source and all other actions receive N/A.
3. Derive V18 Snapshot facts in Rust from native Observation/state, including four shanten modes,
   accepted/declaration states, ankan-excluding open meld count, discard counts, latest tedashi, and
   trailing tsumogiri. Return fixed factors/numeric/lengths through PyO3.

### Phase B: Python schema, model, and boundaries

1. Replace old Snapshot tensors through sample, bridge, validation, collation, inference, rollout,
   evaluation, and bot consumers with `[B,54,4]`, `[B,54,1]`, lengths 54.
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

### Phase E: 49-row Snapshot Objective Facts extension

1. Extend the Rust schema from 29 to 49 rows in place: keep the four placement rows and the
   seat-major opponent summaries, append six per-opponent facts directly after each opponent's five
   existing summary fields, and place the two global categorical facts after the four shanten rows
   and before the latest-tedashi/streak rows.
2. Derive all 20 new facts inside the native preparation loop from own hand/melds, public rivers,
   public meld tiles, and revealed dora indicators only; never read opponent concealed hands, wall
   order, or post-hoc labels. Keep the existing `opened` predicate as the sole menzen test while
   letting ankan tiles contribute to visible-meld dora/aka counts; deduplicate dora kinds for the
   global unknown-copy count; advance dora kinds from current indicators.
3. Propagate the fixed 49-row shapes through Python schema/bridge/validation/architecture without a
   copied field table, update length expectations (116–122, target 118.80) and parameter
   expectations, and re-run Rust, Python, protocol, replay, information-boundary, token-statistics,
   and forward-pass evidence.

### Phase F: Remove the estimated remaining-wall token

1. Delete the live-wall counter from the Rust state-machine state suffix (field, initializer,
   tsumo decrement, and its token row); renumber the remaining KIND_COUNTER fields to contiguous
   1..7 so no hole remains.
2. Update protocol tests to assert the absence of the removed counter and the new field numbers,
   and add a one-token-per-decision correction to the read-only token-statistics tool.
3. Rebuild/reinstall the local PyO3 extension, re-run Rust/Python/protocol suites and the
   read-only statistics, and synchronize FR-008d, Decision 13, protocol docs, and PROGRESS.md.

### Phase G: 54-row Snapshot extension: progress, riichi-trait, and turn facts

1. Replace each opponent's latest-tedashi tile with the riichi declaration tile (same 37-code
   vocabulary), add post-riichi tsumogiri counts per opponent, and keep the fixed opponent summary
   at 13 rows per seat.
2. Add self shanten-improving and tenpai winning tile counts from a normalized 13-tile shape and
   the legal known area; add the exact current turn (public discard rounds plus one) to the
   Objective Facts state suffix. All new facts are derived in Rust with no estimated values.
3. Update schema/domains/tests/docs/statistics (54 rows, mean 124.803028) and parameter
   expectations (4,940,802), then re-run Rust, Python, protocol, and forward-pass evidence.

## Complexity Tracking

No constitution violations require justification.
