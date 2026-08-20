# Tasks: RiichiLab Bot V16 输入适配

**Input**: Design documents from `/specs/005-riichi-lab-bot-v16-input/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Required by the user and constitution because this is a protocol/semantics change.

## Phase 1: Setup

- [X] T001 Create feature governance artifacts in `specs/005-riichi-lab-bot-v16-input/`

---

## Phase 2: Foundational

- [X] T002 Add V16 actor input semantic validator in `riichi_ppo_v1/model/semantic_validation.py`
- [X] T003 Refactor `PreparedDecision` data shape in `riichi_lab_bot/src/riichi_lab_bot/bridge.py`

---

## Phase 3: User Story 1 - V16/V17 checkpoint 可加载并推理 (Priority: P1)

**Goal**: V16 SFT 与 V17 PPO checkpoint 可用同一 bot runtime 推理。

**Independent Test**: `riichi_lab_bot/tests/test_checkpoint.py` loads both checkpoints and runs warmup/infer.

- [X] T004 [US1] Update checkpoint tests in `riichi_lab_bot/tests/test_checkpoint.py`
- [X] T005 [US1] Implement V16 strict checkpoint loading in `riichi_lab_bot/src/riichi_lab_bot/policy.py`
- [X] T006 [US1] Implement V16 warmup and inference in `riichi_lab_bot/src/riichi_lab_bot/policy.py`

---

## Phase 4: User Story 2 - 在线 observation 生成训练等价 V16 输入 (Priority: P1)

**Goal**: Single-seat online bridge output matches training `prepare_v16()`.

**Independent Test**: `riichi_lab_bot/tests/test_bridge_integration.py` compares every V16 segment over a fixed half-game.

- [X] T007 [US2] Update bridge equivalence tests in `riichi_lab_bot/tests/test_bridge_integration.py`
- [X] T008 [US2] Replace old v13 token assembly with training `prepare_v16()` reuse in `riichi_lab_bot/src/riichi_lab_bot/bridge.py`
- [X] T009 [US2] Update safety fake prepared construction in `riichi_lab_bot/tests/test_safety.py`

---

## Phase 5: User Story 3 - 缺失线上字段可重建并语义校验 (Priority: P1)

**Goal**: Missing online snapshot fields are reconstructed and tested against full local observations.

**Independent Test**: `riichi_lab_bot/tests/test_bridge_semantics.py` deletes missing fields across a local half-game and compares V16 inputs.

- [X] T010 [US3] Extend online snapshot tracker in `riichi_lab_bot/src/riichi_lab_bot/observation.py`
- [X] T011 [US3] Apply reconstructed fields before V16 prepare in `riichi_lab_bot/src/riichi_lab_bot/bridge.py`
- [X] T012 [US3] Update missing-field and edge-window tests in `riichi_lab_bot/tests/test_bridge_semantics.py`

---

## Final Phase: Polish & Cross-Cutting

- [X] T013 Update V16/V17 runtime docs in `riichi_lab_bot/README.md`
- [X] T014 Run semantic and bot tests from `specs/005-riichi-lab-bot-v16-input/quickstart.md`
- [X] T015 Run local three-game smoke with V16/V17 checkpoint when CUDA is available

## Dependencies & Execution Order

- Phase 1 before all work.
- Phase 2 blocks all user stories.
- US1 and US2 both depend on the new `PreparedDecision` shape.
- US3 depends on US2 bridge path.
- Polish depends on US1-US3.

## Implementation Strategy

Complete V16 input shape and validator first, then checkpoint inference, then online field semantics. Keep changes scoped to bot runtime and tests, with training-side V16 encoders as the single source of truth.
