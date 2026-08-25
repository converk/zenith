# Data Model: V18 Input Architecture

## AtomicSnapshotField

Authoritative immutable Rust definition for one of 54 positions, exported through PyO3 and consumed
without a copied Python field table.

| Attribute | Type | Validation |
|---|---|---|
| index | integer | 0..48, unique and contiguous |
| field_id | integer | 1..54, unique schema ID |
| name | string | stable protocol name |
| relative_seat | integer | 0 for self/global or 1..3 opponent |
| categorical_domain | named domain | fixed range/bucket table |
| tile_domain | named domain | N/A or red-aware 37 identity |
| numeric_rule | named rule | zero or fixed score-pressure normalization |

Canonical order:

1. own rank
2. score pressure versus seats 1, 2, 3
3. for each seat 1, 2, 3: riichi status, riichi turn, open meld count, tedashi count,
   tsumogiri count, first-six-discard man/pin/sou/terminal-or-honor counts, confirmed open-meld
   yakuhai han, and visible-meld dora-plus-aka han
4. overall, standard, chiitoitsu, kokushi shanten
5. fully-visible tile-kind count and unknown copies across distinct dora kinds
6. latest tedashi for seats 1, 2, 3
7. consecutive tsumogiri for seats 1, 2, 3

## SnapshotBatch

| Attribute | Shape/type | Validation |
|---|---|---|
| snapshot_factors | integer `[B,54,4]` | columns are field, relative seat, category, tile |
| snapshot_numeric | float `[B,54,1]` | finite and rule-valid |
| snapshot_lengths | integer `[B]` | every value equals 54 |

State transition: native Observation → Rust-derived 54 rows → Python batch with no variable-length
Snapshot padding. Invalid native facts fail before model forward.

## ActionQueryPair

| Attribute | Type | Validation |
|---|---|---|
| action_id | integer | unique among valid pairs; 0..240; present in legal mask |
| offense_row | 15 integers | query type Offense and matching metadata/action ID |
| defense_row | 15 integers | query type Defense and matching metadata/action ID |
| action_type | category | supported canonical action code |
| primary_tile | category | N/A or one of 34 tile kinds |
| source_seat | category | 1..3 only for supplier actions, otherwise N/A |

Relationships: one legal action owns exactly one pair and one raw logit. Pair storage order is not
semantic; `action_id` is the identity and output mapping key.

## ActorInput

Contains history, current-state suffix, Atomic Snapshot, legal mask, and Action Query pairs. It has
no fields for concealed opponent hands or future wall. Shared-public tokens precede all query pairs.

## CriticPrivateContext

| Segment | Order | Validation |
|---|---|---|
| opponent concealed hands | relative seats 1, 2, 3 | exactly three seats; red identity retained |
| future wall | positions 1..5 | exactly five valid physical tiles |
| value query | final | exactly one learned token |

Relationships: appended only to shared public representation; never appended to Actor/query input.

## V18EncodedSample

Contains ActorInput, optional CriticPrivateContext, target action, provenance, and encoding version
18. Full SFT dataset materialization is deferred, but fixtures and streaming validation use the same
schema.

## ActorOnlyCheckpoint

| Attribute | Validation |
|---|---|
| contract_version | exact active V18 SFT contract |
| encoding_protocol_version | exactly 18 |
| model_config | exact conservative V18 topology |
| actor_state | only Actor/public/embedding/policy parameters |

Load has one transition: valid V18 artifact → restored Actor. Historical or malformed artifacts
transition to a hard error; there is no migration state.
