# Tasks: V18 Input Architecture

**Input**: Design documents from `specs/008-v18-input-architecture/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Required by the feature. Story test tasks precede their implementation tasks.

**Organization**: Tasks are grouped by user story and executed in dependency order. Every completed
task is changed to `[X]` only after its stated validation passes.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the audited V18 work surface without modifying archived artifacts.

- [X] T001 Record the initial repository status, archived-asset preservation boundary, and V18 validation plan in `audit/reports/v18/report/PROGRESS.md`
- [X] T002 [P] Create fixed V18 audit subdirectory placeholders and validation-script location under `audit/reports/v18/eval/` and `audit/reports/v18/scripts/`
- [X] T003 Run and record repository-wide active/reference audits for V16/V17/Q/schema/snapshot/query symbols in `audit/reports/v18/report/PROGRESS.md`
- [X] T004 [P] Add the self-contained, non-executed Actor-only configuration in `riichi_ppo_v1/configs/v18_sft.yaml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Lock the single V18 protocol and native bridge contract used by every story.

**⚠️ CRITICAL**: No story implementation proceeds until protocol tests and native bindings are ready.

- [X] T005 [P] Add failing V18 protocol/domain/order tests in `riichi_ppo_v1/tests/unit/test_v18_encoding_protocol.py`
- [X] T006 [P] Add failing native Snapshot and source-seat tests in `RiichiEnv/riichienv-state-machine/src/atomic_snapshot.rs` and `RiichiEnv/riichienv-python/src/encoding_facts.rs`
- [X] T007 Define the only 29-field order/domain table and machine-readable schema export in `RiichiEnv/riichienv-state-machine/src/atomic_snapshot.rs`, then consume it without a copied table in `riichi_ppo_v1/model/encoding_protocol.py`
- [X] T008 Implement Rust atomic Snapshot derivation/schema validation and reuse the existing last-supplier event logic in `RiichiEnv/riichienv-state-machine/src/atomic_snapshot.rs`
- [X] T009 Implement native Observation/Action fact extraction including supplier source seats in `RiichiEnv/riichienv-python/src/encoding_facts.rs` and register it in `RiichiEnv/riichienv-python/src/lib.rs`
- [X] T010 Update PyO3 typing/export contracts for V18 facts in `RiichiEnv/src/riichienv/_riichienv.pyi`, `RiichiEnv/src/riichienv/__init__.py`, and `RiichiEnv/riichienv-state-machine/src/lib.rs`
- [X] T011 Reinstall both Rust extensions in `Mahjong-AI` and pass focused Rust/PyO3 protocol tests, recording commands in `audit/reports/v18/report/PROGRESS.md`

**Checkpoint**: V18 schema and native atomic facts are authoritative and callable.

---

## Phase 3: User Story 1 - Encode atomic public state (Priority: P1) 🎯 MVP

**Goal**: Every valid decision exposes exactly 29 ordered atomic Snapshot tokens with stable domains.

**Independent Test**: Representative, boundary, N/A, overflow, malformed, and replay fixtures all
produce `[B,29,4]`, `[B,29,1]`, and lengths 29; projected validation mean is 97–103.

### Tests for User Story 1

- [X] T012 [P] [US1] Add failing Python Snapshot normal/boundary/N/A/overflow/illegal-input tests in `riichi_ppo_v1/tests/unit/test_v18_snapshot.py`
- [X] T013 [P] [US1] Add failing Rust/Python encoding bridge tests in `riichi_ppo_v1/tests/integration/test_v18_encoding_bridge.py`
- [X] T014 [P] [US1] Add a real MJAI replay-to-V18 bridge test using `RiichiEnv/tests/data/126_204_0_mjai.jsonl` in `riichi_ppo_v1/tests/integration/test_v18_replay_bridge.py`

### Implementation for User Story 1

- [X] T015 [US1] Replace mixed V16 Snapshot assembly with the fixed native V18 bridge in `riichi_ppo_v1/model/snapshot.py`
- [X] T016 [US1] Replace prepared-batch Snapshot fields and enforce length 29 in `riichi_ppo_v1/model/bridge.py`
- [X] T017 [US1] Update V18 sample entities and collation shapes in `riichi_ppo_v1/sft/data.py` and `riichi_ppo_v1/sft/train.py`
- [X] T018 [US1] Update semantic shape/domain/order validation in `riichi_ppo_v1/model/semantic_validation.py` and `riichi_ppo_v1/model/validation.py`
- [X] T019 [US1] Propagate V18 Snapshot tensors through `riichi_ppo_v1/training/trajectory.py`, `riichi_ppo_v1/training/rollout_buffer.py`, and `riichi_ppo_v1/training/inference.py`
- [X] T020 [US1] Propagate the single V18 Snapshot schema through `riichi_ppo_v1/evaluation/policy_adapter.py` and `riichi_lab_bot/src/riichi_lab_bot/`
- [X] T021 [US1] Implement read-only established-selection length statistics in `riichi_ppo_v1/tools/v18_token_statistics.py` and wrapper `audit/reports/v18/scripts/validate_v18_token_statistics.py`
- [X] T022 [US1] Run US1 Rust, unit, bridge, real replay, and exact token-statistics validation and record component means in `audit/reports/v18/report/PROGRESS.md`

**Checkpoint**: Atomic V18 public input is independently usable and measured.

---

## Phase 4: User Story 2 - Score actions without candidate interference (Priority: P1)

**Goal**: Query metadata is active, source seats are correct, and action-pair order cannot affect
action-ID-aligned raw logits.

**Independent Test**: Metadata ablations affect embeddings; all supplier/non-supplier cases validate;
pair permutations preserve mapped logits at `atol=1e-5, rtol=1e-5`.

### Tests for User Story 2

- [X] T023 [P] [US2] Add failing Query metadata/source-seat/action-ID mapping tests in `riichi_ppo_v1/tests/integration/test_v18_query_semantics.py`
- [X] T024 [P] [US2] Add failing isolated-attention mask and permutation-invariance tests in `riichi_ppo_v1/tests/unit/test_v18_architecture.py`
- [X] T025 [P] [US2] Extend exhaustive action protocol tests for pair mapping and supplier domains in `riichi_ppo_v1/tests/protocol/test_protocol_matrix.py` and `riichi_ppo_v1/tests/protocol/test_action_space_exhaustive.py`

### Implementation for User Story 2

- [X] T026 [US2] Implement Atomic Snapshot embedding for `[B,29,4]` plus `[B,29,1]` and make Query embedding consume action type, 34-kind primary tile, and source seat in `riichi_ppo_v1/model/architecture.py`
- [X] T027 [US2] Validate metadata agreement, unique unordered action IDs, legal-mask set equality, and raw-logit mapping in `riichi_ppo_v1/model/action_query.py` and `riichi_ppo_v1/model/semantic_validation.py`
- [X] T028 [US2] Set the fixed 256/16/4/16/704 and 3+1+2 topology, then implement public-only shared layers and pair-isolated Actor attention/position IDs in `riichi_ppo_v1/model/architecture.py`
- [X] T029 [US2] Update native query bridge naming/contracts and all active callers in `riichi_ppo_v1/model/native_encoding.py` and `riichi_ppo_v1/model/bridge.py`
- [X] T030 [US2] Run US2 metadata, protocol, mask, and permutation tests and record tolerance/results in `audit/reports/v18/report/PROGRESS.md`

**Checkpoint**: Action scoring is metadata-aware and structurally permutation invariant.

---

## Phase 5: User Story 3 - Preserve information boundaries (Priority: P1)

**Goal**: Actor is hidden-information invariant while Critic strictly consumes three real hands and
the next five wall tiles without action Queries.

**Independent Test**: Hidden-only mutations leave Actor logits unchanged; controlled private
mutations alter Critic value; shape, order, mask, and count failures are rejected.

### Tests for User Story 3

- [X] T031 [P] [US3] Add failing Actor hidden-information invariance tests in `riichi_ppo_v1/tests/integration/test_v18_information_boundaries.py`
- [X] T032 [P] [US3] Add failing Critic private-use/order/future-five/query-exclusion tests in `riichi_ppo_v1/tests/unit/test_critic_features.py`

### Implementation for User Story 3

- [X] T033 [US3] Enforce exactly three opponent-hand segments and five future tiles in `riichi_ppo_v1/model/critic_features.py`
- [X] T034 [US3] Split Actor-only and full Actor-Critic call boundaries and keep action Queries out of Critic in `riichi_ppo_v1/model/architecture.py`
- [X] T035 [US3] Strengthen Critic private shape/order/mask validation in `riichi_ppo_v1/model/semantic_validation.py` and `riichi_ppo_v1/model/bridge.py`
- [X] T036 [US3] Run US3 isolation/private-use tests and record evidence in `audit/reports/v18/report/PROGRESS.md`

**Checkpoint**: Public/private boundaries are independently proven.

---

## Phase 6: User Story 4 - Train and persist Actor-only BC (Priority: P2)

**Goal**: A pure V18 BC lifecycle updates and persists Actor parameters only.

**Independent Test**: Forward/backward/optimizer/save/load succeeds with no Critic/value optimizer
membership, gradients, updates, or legacy load acceptance.

### Tests for User Story 4

- [X] T037 [P] [US4] Add failing Actor-only forward/backward/optimizer freeze tests in `riichi_ppo_v1/tests/unit/test_v18_actor_sft.py`
- [X] T038 [P] [US4] Add failing pure-V18 Actor save/load and legacy-rejection integration tests in `riichi_ppo_v1/tests/integration/test_v18_actor_sft_lifecycle.py`

### Implementation for User Story 4

- [X] T039 [US4] Implement explicit Actor-only freeze, parameter iteration, logits, and BC loss in `riichi_ppo_v1/sft/actor_bc.py`
- [X] T040 [US4] Implement exact V18 Actor-only save/load with no migration in `riichi_ppo_v1/sft/actor_bc.py` and `riichi_ppo_v1/sft/checkpoint.py`
- [X] T041 [US4] Replace legacy V16 contract acceptance and manifest schema with pure V18 definitions in `riichi_ppo_v1/sft/contract.py` and `riichi_ppo_v1/sft/precompute.py`
- [X] T042 [US4] Wire the active SFT CLI/config validation to Actor-only V18 without running training in `riichi_ppo_v1/sft/train.py`, `riichi_ppo_v1/configs/sft.yaml`, and `riichi_ppo_v1/configs/v18_sft.yaml`
- [X] T043 [US4] Run the Actor-only lifecycle integration test and record frozen/gradient/update evidence in `audit/reports/v18/report/PROGRESS.md`

**Checkpoint**: V18 Actor-only BC is ready for future formal SFT.

---

## Phase 7: User Story 5 - Audit one active V18 contract (Priority: P2)

**Goal**: Model, state, config, CLI, tests, and active docs agree on V18 with 4.9M–5.1M parameters
and no Q or compatibility path.

**Independent Test**: Unified validation passes and repository audit finds no active Q/V16/V17
contract claims while archived artifacts remain unchanged.

### Tests for User Story 5

- [X] T044 [P] [US5] Add failing unified parameter/state-key tests in `riichi_ppo_v1/tests/unit/test_v18_parameter_count.py`
- [X] T045 [P] [US5] Update active artifact/config/CLI contract tests in `riichi_ppo_v1/tests/unit/test_artifact_conventions.py` and `riichi_ppo_v1/tests/unit/test_config_loading.py`
- [X] T046 [P] [US5] Add production V18 validation coverage in `riichi_ppo_v1/tests/integration/test_v18_validation.py`

### Implementation for User Story 5

- [X] T047 [US5] Remove `q_scorer`, candidate-Q, dueling-Q APIs, and active Q branches after the recorded `rg` audit in `riichi_ppo_v1/model/architecture.py` and `riichi_ppo_v1/training/learner.py`
- [X] T048 [US5] Remove Q controls from active configuration/loading paths while preserving historical configs as archived evidence in `riichi_ppo_v1/configs/` and `riichi_ppo_v1/training/`
- [X] T049 [US5] Implement the unified parameter/state-key counter in `riichi_ppo_v1/model/parameter_count.py` and expose it through `riichi_ppo_v1/tools/validate.py`
- [X] T050 [US5] After per-file `rg`, replace/remove Python active V16 modules and APIs including `riichi_ppo_v1/model/v16_rust_encoding.py`, `riichi_ppo_v1/sft/train_v16.py`, their exports/callers, and active V16-named tests without compatibility aliases
- [X] T051 [US5] After per-file `rg`, replace/remove Rust active V16 encoding modules in `RiichiEnv/riichienv-python/src/v16_facts.rs` and `RiichiEnv/riichienv-state-machine/src/v16_encoding.rs`, then update bot consumers under `riichi_lab_bot/src/riichi_lab_bot/`
- [X] T052 [US5] Run focused V18 parameter, state-key, config, production-validation, and active-reference audits and record exact results in `audit/reports/v18/report/PROGRESS.md`

**Checkpoint**: One auditable active V18 contract remains.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Complete hard-gated documentation, full validation, cleanup, and traceability.

- [X] T053 [P] Add the complete V18 protocol document in `riichi_ppo_v1/docs/v18_input_protocol.md`
- [X] T054 [P] Update Rust/Python event and source-seat bridge semantics in `riichi_ppo_v1/docs/KyokuEventTupleProtocol.md`
- [X] T055 [P] Create/update the root `README.md` and update `riichi_ppo_v1/README.md` plus training framework documentation under `riichi_ppo_v1/docs/`
- [X] T056 [P] Update SFT usage, configuration examples, and CLI/default paths in `riichi_ppo_v1/docs/` and `riichi_ppo_v1/configs/`
- [X] T057 [P] Update active protocol/generation/dataset/artifact guidance and archival labels in `AGENTS.md`, `.gitignore`, and `riichi_lab_bot/README.md`
- [X] T058 [P] Update component responsibilities for new files/directories in `docs/directory-responsibilities.md`
- [X] T059 Audit and correct every active document claiming V16/V17 is current while preserving historical reports/specs in place, recording results in `audit/reports/v18/report/PROGRESS.md`
- [X] T060 Run Rust workspace, Python unit/protocol/integration/replay, RiichiEnv, and bot test suites in `Mahjong-AI` and record pass/fail counts in `audit/reports/v18/report/PROGRESS.md`
- [X] T061 Run `specs/008-v18-input-architecture/quickstart.md`, remove all smoke-generated logs/results, run `git diff --check`, and finalize the requirement-to-test evidence in `audit/reports/v18/report/PROGRESS.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup → Foundational → US1.
- US2 depends on Foundational and the US1 batch/schema propagation.
- US3 depends on the shared-public refactor in US2 but its private encoder tests can start after
  Foundational.
- US4 depends on the US2 Actor API and US1 collation.
- US5 depends on US1–US4 so the final state/key/active-reference audit reflects delivered code.
- Polish depends on all stories; documentation tasks T053–T058 may draft in parallel after interfaces
  stabilize, then T059–T061 serialize final validation.

### User Story Dependencies

- **US1 (P1)**: Foundational only; delivers the MVP atomic schema.
- **US2 (P1)**: US1 prepared-batch/model inputs.
- **US3 (P1)**: US2 shared-public/Actor separation.
- **US4 (P2)**: US1 collation plus US2 Actor-only output.
- **US5 (P2)**: All functional stories.

### Parallel Opportunities

- T002/T004, T005/T006, and each story's test files can proceed in parallel where marked `[P]`.
- Rust Snapshot derivation and Python test fixtures can proceed concurrently after T007.
- US3 Critic tests and US4 persistence tests can be prepared while US2 model work stabilizes.
- T053–T058 affect distinct documents and can run in parallel after public interfaces freeze.

## Parallel Examples

```text
US1: T012 Snapshot unit tests | T013 bridge tests | T014 real replay test
US2: T023 metadata tests | T024 attention tests | T025 protocol matrix
US3: T031 Actor isolation | T032 Critic private-use tests
US4: T037 freeze/optimizer | T038 save/load lifecycle
US5: T044 parameter/state keys | T045 config contracts | T046 production validation
Docs: T053 protocol | T054 event bridge | T055 README | T056 SFT | T057 governance | T058 directories
```

## Implementation Strategy

### MVP First

1. Complete T001–T011 to establish V18 native facts and schema.
2. Complete T012–T022 for a fixed, validated, measurable 29-token input.
3. Validate US1 independently before modifying attention or training interfaces.

### Incremental Delivery

1. Atomic input → metadata-aware isolated policy → proven information boundary.
2. Add Actor-only BC lifecycle.
3. Remove Q/legacy paths and run one-active-contract audit.
4. Synchronize documentation and run full converge validation.

## Notes

- Historical V16/V17 checkpoints, datasets, logs, reports, and specs are never task deletion targets.
- Code comments added or modified during implementation are Chinese.
- No task authorizes full dataset generation, formal SFT, PPO design/run, or evaluation changes.
