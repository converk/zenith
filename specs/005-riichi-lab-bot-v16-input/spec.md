# Feature Specification: RiichiLab Bot V16 输入适配

**Feature Branch**: `005-riichi-lab-bot-v16-input`

**Created**: 2026-08-20

**Status**: Draft

**Input**: 将 `riichi_lab_bot` 从旧 V13 actor 输入迁移到现行 V16 actor 输入协议,支持 V16 SFT 与 V17 PPO checkpoint。V16 与 V17 actor 结构完全一致,V17 仅是在 V16 SFT 基础上 PPO 训练,因此 bot 只实现一套 V16 输入与推理逻辑。删除 bot 中旧版本与 token schema 限制,重点验证每个输入段与当前局面、历史事件、snapshot、query 语义一致。线上阶段仅跑 validation,不接入 ranked。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - V16/V17 checkpoint 可加载并推理 (Priority: P1)

作为 bot 维护者,我希望同一个客户端可以加载 V16 SFT 和 V17 PPO checkpoint,并用 V16 actor 前向选择合法动作。

**Why this priority**: 如果模型无法加载或仍走旧 `isolated_action_query` 前向,线上 validation 无法开始。

**Independent Test**: 分别加载本地 V16 SFT 与 V17 PPO checkpoint,执行 warmup 与首个合法局面的 CPU 推理,输出动作必须在 legal mask 内。

**Acceptance Scenarios**:

1. **Given** V16 SFT checkpoint, **When** bot 加载并 warmup, **Then** 模型按 `symmetric_action_query` strict load 并走 `forward_v16(policy_only=True)`。
2. **Given** V17 PPO checkpoint, **When** bot 加载并推理, **Then** 使用同一套 V16 输入与推理逻辑,不因 PPO format 或历史 token schema 闸门失败。

---

### User Story 2 - 在线 observation 生成训练等价 V16 输入 (Priority: P1)

作为训练与线上一致性维护者,我希望单席在线 bridge 生成的 history、snapshot、query 与训练侧 `BatchedStateBridge.prepare_v16()` 完全一致。

**Why this priority**: V16 的策略输入语义来自当前真实局面;任何 history 时间线、snapshot 或 query 偏差都会让线上行为偏离训练分布。

**Independent Test**: 本地半庄逐请求比较 online bridge 与训练 bridge 的 V16 三段数组、长度、legal mask 和 action id。

**Acceptance Scenarios**:

1. **Given** 任一 request_action observation, **When** online bridge prepare, **Then** history 表示只含已发生且当前可见的客观事件与自身状态。
2. **Given** 当前局面 snapshot, **When** 编码 compact snapshot, **Then** 场况、分数、剩余牌、宝牌、对手 7 项摘要与 observation/current event stream 一致。
3. **Given** 当前 legal actions, **When** 编码 query rows, **Then** 每个合法 action id 恰有相邻 offense/defense 一对 query,slot 编码在声明基数内。

---

### User Story 3 - 缺失线上字段可重建并语义校验 (Priority: P1)

作为线上 bot 维护者,我希望 RiichiLab 缺少新 snapshot 字段时,bot 能从事件流和 observation 重建字段,并在语义不一致时失败而不是默默喂错输入。

**Why this priority**: V16 query 使用 `tiles_left`、`tsumogiri_flags`、`riichi_*`、`missed_agari_*` 等事实;旧默认值会污染振听、对手摘要和局面时间。

**Independent Test**: 在本地完整半庄中删除线上可能缺失字段,逐请求重建后与完整 observation 的 V16 输入完全一致。

**Acceptance Scenarios**:

1. **Given** start_kyoku, **When** 新局开始, **Then** riichi、tsumogiri、missed-agari、tiles-left tracker 全部 reset。
2. **Given** 牌河事件, **When** 发生摸切/手切/立直声明/立直通过, **Then** `tsumogiri_flags`、`last_tedashis`、`riichi_declaration_indices`、`riichi_sutehais` 与当前局面一致。
3. **Given** 见逃或同巡振听状态, **When** 后续 query 计算 offense/furiten slot, **Then** `missed_agari_doujun` 与 `missed_agari_riichi` 与 RiichiEnv 完整 observation 一致。

## Edge Cases

- `start_kyoku` 必须清空上一局的 ron target、riichi、tsumogiri、missed-agari、tiles-left 残留。
- 吃、碰、大明杠、暗杠、加杠、九种九牌、自摸、荣和窗口都必须能 prepare、decode、通过安全响应校验。
- 红五在 action template、query primary tile 和安全响应比较中必须保持区分。
- 服务器提供空 `tsumogiri_flags`、缺少 `tiles_left` 或 stale `riichi_sutehais` 时,tracker 重建值优先。
- 如果当前 observation 与 tracker 产生不可调和的合法动作或字段矛盾,bot 应拒绝该决策并记录错误。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Bot MUST load any checkpoint whose `model_config.policy_head_type` is `symmetric_action_query` and whose `model` tensors strict match `KyokuTransformerActorCritic(ModelConfig)`.
- **FR-002**: Bot MUST NOT enforce legacy `token_schema_version`、`feature_schema_sha256`、`rust_analysis_version`、`decision_analysis_version`、PPO v2 或 V13 SFT policy-head restrictions.
- **FR-003**: Bot MUST prepare V16 actor input as separate history、snapshot、query arrays with lengths and legal mask, matching training-side `prepare_v16()`.
- **FR-004**: Bot MUST call a V16 semantic validator before every model inference.
- **FR-005**: Bot MUST decode model action ids via the existing state machine and preserve existing safety validation against `Observation.select_action_from_mjai()` and server `possible_actions`.
- **FR-006**: Bot MUST reconstruct or override online-missing snapshot fields used by V16 semantics, including riichi, tsumogiri, tiles-left and missed-agari state.
- **FR-007**: Bot README and tests MUST describe V16/V17-only support and validation-only online rollout.

### Key Entities

- **PreparedDecision**: Single decision payload containing V16 actor input segments, legal action metadata and event context.
- **OnlineStateBridge**: Stateful per-seat MJAI state bridge plus online field tracker.
- **PolicyEngine**: Checkpoint loader and deterministic inference wrapper for V16 actor forward.
- **V16SemanticValidator**: Assertion layer for history visibility, snapshot/query cardinalities and legal-mask alignment.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: V16 SFT and V17 PPO local checkpoints both load and warm up on CPU without legacy schema errors.
- **SC-002**: Online bridge and training bridge produce byte-identical V16 input arrays for a full fixed-seed local half-game.
- **SC-003**: Missing-field replay tests complete with zero semantic mismatches across at least one fixed-seed local half-game.
- **SC-004**: Local bot run completes three fixed-seed games with zero fallback and zero withheld actions, excluding user-requested online token validation.

## Assumptions

- V13/V14/V15 bot checkpoint support is out of scope and may be removed from bot code/tests.
- V16 and V17 actor topology and input protocol are identical.
- Online ranked remains disabled until validation passes with a user-provided token.
