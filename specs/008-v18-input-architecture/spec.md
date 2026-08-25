# Feature Specification: V18 Input Architecture

**Feature Branch**: `008-v18-input-architecture`
**Created**: 2026-08-25
**Status**: Draft
**Input**: Upgrade Zenith to one V18 input/model contract with atomic snapshots, isolated query
attention, strict Actor/Critic information boundaries, and Actor-only BC SFT readiness.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Encode atomic public state (Priority: P1)

As a training-data producer, I need every decision to expose a fixed, unambiguous set of atomic
public-state facts so semantically different mahjong situations remain distinguishable.

**Why this priority**: The V18 model and all later training data depend on one trustworthy input
contract.

**Independent Test**: Encode representative and boundary fixtures and verify every sample has
exactly 29 ordered Snapshot tokens, valid domains, and the required batch shapes.

**Acceptance Scenarios**:

1. **Given** any valid decision state, **When** encoded, **Then** its Snapshot has exactly 29 tokens
   in canonical order and no fact already represented by the state suffix.
2. **Given** open/closed hands, all riichi phases, red fives, count limits, and winning shanten,
   **When** encoded, **Then** every value uses its specified categorical, tile, numeric, or N/A domain.
3. **Given** the established 2024/2025 60% validation selection or a representative non-materialized
   pass, **When** V18 lengths are measured, **Then** mean total length is 97–103 and component
   statistics are reported.

---

### User Story 2 - Score actions without candidate interference (Priority: P1)

As a policy consumer, I need each legal action's Offense/Defense pair to use its actual action
metadata while remaining structurally isolated from every other candidate pair.

**Why this priority**: Candidate order must not change policy meaning or raw action scores.

**Independent Test**: Permute legal action pairs, map outputs by action ID, and verify raw logits
agree within normal floating-point tolerance.

**Acceptance Scenarios**:

1. **Given** multiple legal actions, **When** a pair is processed, **Then** it sees the public prefix
   and its partner but no token from another pair.
2. **Given** action queries, **When** public tokens are processed, **Then** public tokens cannot
   attend back to any query.
3. **Given** the same actions in different orders, **When** logits are mapped by action ID,
   **Then** corresponding values are invariant within declared tolerance.
4. **Given** chi, pon, daiminkan, or ron, **When** metadata is produced, **Then** source seat is the
   real relative supplier seat; actions without a supplier use N/A.

---

### User Story 3 - Preserve information boundaries (Priority: P1)

As a model reviewer, I need proof that the Actor uses only observable/self information while the
Critic actually uses three opponents' concealed hands and the next five wall tiles.

**Why this priority**: Leakage invalidates policy training, while missing private Critic facts
weakens the intended value estimator.

**Independent Test**: Change only concealed hands and future wall, then verify Actor logits are
unchanged and Critic values respond to valid private-input changes.

**Acceptance Scenarios**:

1. **Given** identical Actor-visible input, **When** concealed opponent hands and future wall change,
   **Then** Actor logits remain unchanged within tolerance.
2. **Given** valid private information, **When** Critic evaluation runs, **Then** three ordered hands,
   five future tiles, and a value query are accepted with validated shapes and masks.
3. **Given** legal-action Query tokens, **When** Critic evaluation runs, **Then** no action Query is
   part of Critic input.

---

### User Story 4 - Train and persist Actor-only BC (Priority: P2)

As an SFT operator, I need a V18-only behavior-cloning interface that updates the Actor without
computing or modifying Critic/value parameters.

**Why this priority**: It prepares the architecture for later SFT without starting formal training.

**Independent Test**: Execute a small forward/backward/optimizer/save/load cycle and verify loss,
gradients, updated parameters, frozen parameters, and restored outputs.

**Acceptance Scenarios**:

1. **Given** a labeled V18 batch, **When** BC runs, **Then** only legal-action Actor logits and BC
   loss are computed.
2. **Given** an optimizer step, **When** parameters are inspected, **Then** only Actor parameters
   participate; Critic/value parameters remain frozen, gradient-free, and unchanged.
3. **Given** an Actor-only artifact, **When** loaded under V18, **Then** outputs are restored and
   historical schema/checkpoint inputs are rejected.

---

### User Story 5 - Audit one active V18 contract (Priority: P2)

As a maintainer, I need configuration, validation, documentation, and state artifacts to agree on
one V18 contract with no active Q scorer or historical compatibility path.

**Why this priority**: A version upgrade is incomplete if stale paths silently remain usable.

**Independent Test**: Run contract, parameter, state-key, configuration, replay, and active-reference
audits and compare their evidence with V18 documentation.

**Acceptance Scenarios**:

1. **Given** the Actor-Critic model, **When** parameters are counted under the documented scope,
   **Then** the total is 4.9M–5.1M and excludes every Q module.
2. **Given** V18 state/configuration, **When** audited, **Then** no Q scorer, candidate-Q, Q-boosting,
   fallback, migration, or legacy field exists.
3. **Given** active documentation, **When** version references are audited, **Then** V18 is current
   and every V16/V17 asset reference identifies cold storage.

### Edge Cases

- Winning states distinguish AGARI from numeric shanten zero.
- Chiitoitsu/kokushi use N/A for open hands and their specified ranges otherwise.
- Latest tedashi preserves red-five identity and uses N/A if absent.
- Tsumogiri streaks above four use `4+`; riichi turns above 24 use `25+`.
- Stable count domains handle zero and overflow without sample-dependent scaling.
- Actions without a supplier cannot retain a stale source seat; supplier actions cannot use N/A.
- Variable prefix/action counts preserve isolation, padding, action-ID mapping, and Snapshot length.
- Invalid order/domain/shape, duplicate IDs, incomplete pairs, wrong private-hand order, and a future
  count other than five are rejected.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST expose one active V18 schema/version contract and MUST NOT provide
  V16/V17 loading, adaptation, conversion, fallback, compatibility flags, dual models, legacy maps,
  or state migration.
- **FR-002**: Every valid sample MUST contain exactly 29 atomic Snapshot tokens ordered as four
  placement/pressure facts, fifteen opponent-summary facts, and ten derived facts.
- **FR-003**: Placement facts MUST contain own rank followed by score pressure relative to each
  opponent in fixed relative-seat order, using one sample-independent normalization.
- **FR-004**: For each opponent in fixed order, summary facts MUST be riichi status, riichi turn,
  open meld count excluding concealed kans, cumulative tedashi count, and cumulative tsumogiri count.
- **FR-005**: Derived facts MUST be overall, standard, chiitoitsu, and kokushi shanten, followed by
  each opponent's latest tedashi tile and consecutive-tsumogiri length.
- **FR-006**: Overall/standard shanten MUST support AGARI or 0..6; chiitoitsu MUST support N/A for
  open hands or AGARI/0..6; kokushi MUST support N/A for open hands or AGARI/0..13.
- **FR-007**: Latest tedashi MUST support N/A or 37 red-aware tiles; streak MUST use 0/1/2/3/4+;
  riichi status MUST use NONE/DECLARED/ACCEPTED; riichi turn MUST use N/A/1..24/25+.
- **FR-008**: Rank, score pressure, open meld count, tedashi count, and tsumogiri count MUST each
  have one stable, documented, test-locked domain or bucket scheme established in design.
- **FR-009**: Snapshot MUST NOT duplicate round wind, hand number, dealer, honba, riichi sticks,
  remaining tiles, dora indicators, absolute scores, own hand, or drawn tile from the state suffix.
- **FR-010**: Snapshot factors MUST be `[B,29,4]` ordered as field ID, relative seat, categorical
  value, tile value; numeric values MUST be `[B,29,1]`; every valid length MUST equal 29.
- **FR-011**: Field order/domains MUST have one authoritative definition shared by encoding,
  validation, collation, model consumption, and protocol checks.
- **FR-012**: Existing history, current-state suffix, and one Offense/Defense pair per legal action
  MUST remain in the public-input organization.
- **FR-013**: Query embeddings MUST materially consume action type, primary tile, and source seat.
- **FR-014**: Chi, pon, daiminkan, and ron MUST carry the real relative supplier seat through the
  bridge; all other actions MUST carry N/A.
- **FR-015**: The contract MUST validate a one-to-one action ID, token pair, and raw-logit mapping.
- **FR-016**: Each pair MUST see the public prefix and itself, MUST NOT see another pair, and MUST
  reuse pair-local positions; public tokens MUST NOT see query tokens.
- **FR-017**: Permuting pairs MUST preserve action-ID-aligned logits within declared tolerance.
- **FR-018**: The model MUST retain rotary positions, RMS normalization, causal grouped-query
  attention, and gated feed-forward processing with width 256, 16 query heads, 4 key/value heads,
  head dimension 16, and feed-forward width 704.
- **FR-019**: Actor processing MUST have three shared-public layers and one Actor-only layer;
  Critic processing MUST have two layers after the shared representation.
- **FR-020**: Unified parameter counting MUST include embeddings, Actor, Critic, and value head,
  exclude Q modules, and total 4.9M–5.1M.
- **FR-021**: Active code/configuration/APIs/state keys MUST contain no Q scorer, candidate-Q,
  Q-boosting, or effective Q branch; removal requires prior repository-wide reference audit and tests.
- **FR-022**: Actor inputs MUST be limited to public observations, self information, and action
  queries; changing only concealed opponent hands/future wall MUST NOT change Actor logits.
- **FR-023**: Critic input MUST combine shared public representation with three real opponent hands
  in fixed order, exactly five future wall tiles, and a value query; it MUST NOT receive action Queries,
  and validation MUST prove private facts affect computation.
- **FR-024**: Actor-only BC MUST compute only Actor legal-action logits/loss, use no auxiliary loss,
  freeze Critic/value, omit them from its optimizer, and leave them gradient-free and unchanged.
- **FR-025**: Actor-only forward/backward/step/V18-only save/load MUST pass integration testing;
  non-V18 artifacts MUST be rejected rather than migrated.
- **FR-026**: A self-contained V18 configuration MUST supply all version-specific values without
  overlay inheritance; reusable implementation MUST not hardcode experiment/artifact/schema paths.
- **FR-027**: Validation MUST cover unit, protocol, schema, integration, replay, normal, boundary,
  N/A, stable-bucket, malformed-input, isolation, state-key, and parameter-count cases.
- **FR-028**: A non-materializing pass over the established 60% selection, or documented
  representative V18 validation if it cannot be fully traversed, MUST report 97–103 mean tokens and
  component statistics.
- **FR-029**: V18 protocol, bridge docs, root/framework READMEs, training/SFT docs, config/default
  paths, AGENTS.md, directory responsibilities when changed, and active references MUST be synchronized.
- **FR-030**: `audit/reports/v18/report/PROGRESS.md` MUST record implementation, tests, parameter
  count, Q audit, token statistics, and reproducible commands.
- **FR-031**: V16/V17 checkpoints, datasets, logs, and reports MUST remain unmodified/undeleted;
  active references MUST label them archival.
- **FR-032**: This feature MUST NOT materialize the full V18 dataset, run formal SFT, design/run PPO,
  or change PPO evaluation mechanism, frequency, scale, or governance.

### Key Entities *(include if feature involves data)*

- **Atomic Snapshot Token**: One canonical public-state fact with field identity, relative-seat
  applicability, categorical/tile representation, and optional normalized numeric value.
- **Action Query Pair**: Offense and Defense tokens tied to one action ID and actual metadata.
- **Public Prefix**: History, current-state suffix, and Snapshot facts visible to all action pairs.
- **Critic Private Context**: Three ordered concealed hands plus exactly five future wall tiles.
- **V18 Training Sample**: V18-only inputs, actions, labels, lengths, masks, and schema identity.
- **Actor-only Artifact**: Persisted V18 Actor parameters and contract metadata.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of valid samples contain 29 Snapshot tokens with `[B,29,4]` factors,
  `[B,29,1]` numeric values, and length 29.
- **SC-002**: Representative mean total length is 97–103, target near 99.80, with separate history,
  Snapshot, and action-pair contributions.
- **SC-003**: Parameter count is 4,900,000–5,100,000 and 0 state/config keys belong to Q scoring.
- **SC-004**: 100% of action-ID-aligned logits in permutation fixtures agree within tolerance.
- **SC-005**: Hidden-only changes cause 0 Actor changes beyond tolerance, while a controlled private
  change demonstrably changes Critic output.
- **SC-006**: Actor-only BC forward/backward/step/save/load completes with 0 Critic/value gradients,
  optimizer parameters, or parameter changes.
- **SC-007**: All affected Python, Rust, protocol, integration, and replay tests pass, including
  specified boundary and malformed-input cases.
- **SC-008**: Required active docs/config/CLI paths agree with V18; active-reference audit finds 0
  unlabeled claims that V16/V17 are current.
- **SC-009**: Delivery produces no full V18 dataset, formal SFT, PPO work, evaluation change, or
  modification/deletion of archived V16/V17 artifacts.

## Assumptions

- The established 2024/2025 60% selection remains authoritative and can be streamed or sampled.
- Historical-event and current-state suffix semantics remain valid except required source-seat data.
- Fixed opponent order is the repository's established self-relative order and will be explicit.
- Delegated stable domains use global limits/overflow buckets, never sample-dependent scaling.
- Floating-point tolerances are fixed before implementation and shared by validations.
- Archived V16/V17 artifacts remain physically in place; archival moves are out of scope.
