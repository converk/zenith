# Research: V18 Input Architecture

## Decision 1: Atomic Snapshot representation

- **Decision**: Replace the variable V16 mixed-kind Snapshot with exactly 49 rows. Each row uses
  four integer factors `(field_id, relative_seat, categorical_value, tile_value)` and one numeric
  channel. Field IDs are 1..29; factor value 0 is reserved for N/A/unused. The authoritative table
  lives once in Rust and is exported as machine-readable schema metadata; the Python protocol,
  validators, collators, and model construct their definitions directly from that export.
- **Rationale**: A nonzero field ID keeps every valid row distinguishable even when its categorical
  value is legitimately zero. Direct consumption of one native table prevents cross-language drift.
- **Alternatives considered**: One embedding class per field was rejected as repetitive; retaining
  V16 kinds was rejected because it preserves heterogeneous multi-fact tokens.

## Decision 2: Stable domains and normalization

- **Decision**: Use fixed relative seats 1=shimocha, 2=toimen, 3=kamicha; own rank 1..4;
  riichi 1=NONE, 2=DECLARED, 3=ACCEPTED; riichi turn 0=N/A, 1..24 exact, 25=25+;
  open meld count 0..4 excluding ankan; cumulative discard-type counts 0..24 exact, 25=25+;
  shanten 0=N/A, 1=AGARI, and `n+2` for numeric n; tile 0=N/A and 1..37 for red-aware identity;
  tsumogiri streak 0..3 exact and 4=4+. Score pressure is `clip(score_delta / 100000,-1,1)`.
- **Rationale**: All domains are sample-independent, testable, compact, and cover legal four-player
  play with explicit overflow. A 100,000-point divisor represents a full starting-table score mass
  and gives one stable signed normalization.
- **Alternatives considered**: V16 per-field scales and `clip(delta/25000,-5,5)` are mathematically
  similar but less immediately bounded; quantile/data-dependent buckets violate the contract.

## Decision 3: Rust/Python Snapshot boundary

- **Decision**: Rust owns derivation and validation from native Observation: four shanten values,
  accepted/declaration status, declaration turn, open-meld count, tedashi/tsumogiri counts, latest
  tedashi with red identity, and trailing tsumogiri streak. Python only invokes the native batch
  bridge and assembles/pads fixed-shape arrays.
- **Rationale**: The current native Observation already exposes hands, melds, discards,
  `riichi_accepted`, declaration indices, `last_tedashis`, and flags. Rust already owns shanten.
- **Alternatives considered**: Computing facts in Python was rejected because it would duplicate
  state semantics and weaken the Rust/Python protocol test.

## Decision 4: Query metadata and source seat

- **Decision**: Keep the 15-column query row. Embed columns 2..4 using independent action-type,
  34-kind primary-tile, and relative-source-seat tables, in addition to action ID, query type,
  and answer slots. Rust derives the supplier from the latest applicable public event for chi, pon,
  daiminkan, and ron; every other action is N/A. Both tokens in a pair must have identical metadata.
- **Rationale**: This preserves storage width while making all promised metadata semantically active.
- **Alternatives considered**: Splitting metadata into more query tokens violates the requirement;
  using `Action.actor` is wrong because it identifies the responding actor, not the supplier.

## Decision 5: Action-pair isolated attention

- **Decision**: Run the three shared layers over public tokens only. For the one Actor-only layer,
  concatenate shared public states with query embeddings and build a structural mask: each public
  row sees only causal public keys; each query row sees every valid public key and exactly the two
  rows of its own pair. All pairs reuse position IDs `(public_capacity, public_capacity+1)`.
- **Rationale**: It directly realizes three shared-public layers plus one Actor layer, allows pair
  bidirectionality, blocks candidate leakage, and makes pair-order invariance structural.
- **Alternatives considered**: Ordinary causal masking fails because Offense cannot see Defense and
  later pairs see earlier pairs; data augmentation offers only approximate invariance.

## Decision 6: Actor/Critic boundary

- **Decision**: Actor methods accept no Critic-private arguments. Full Actor-Critic forward builds
  shared public states once, sends public states plus action pairs to Actor, and independently sends
  public states plus ordered opponent-hand tokens, exactly five ordered future-wall tokens, and a
  value query to the two-layer Critic. Private validation rejects absent/incomplete context when a
  full value is requested.
- **Rationale**: Separate call signatures make leakage difficult and testable. Existing critic token
  encoding already preserves relative hand order, red fives, and future position.
- **Alternatives considered**: Passing one omnibus batch into policy-only code was rejected because
  hidden information could be accidentally consumed.

## Decision 7: Conservative model size and parameter accounting

- **Decision**: Fix `d_model=256`, 16 query heads, 4 KV heads, head dimension 16, FFN 704,
  3 shared + 1 Actor-only + 2 Critic layers. Count all model parameters (including frozen ones),
  embeddings, Actor, Critic, and value head/query; Q modules do not exist. Expose one production
  counter and test 4.9M–5.1M.
- **Rationale**: With current non-Q components the topology is 4,911,106 parameters before the small
  V18 metadata/Snapshot embedding delta, leaving safe margin inside the requested range.
- **Alternatives considered**: Larger widths/layers exceed the target; retaining a dormant Q scorer
  violates state-dict and API requirements.

## Decision 8: Pure V18 Actor-only SFT contract

- **Decision**: Provide an explicit Actor-only BC wrapper/API that freezes Critic/value parameters,
  returns legal-action logits plus cross-entropy loss, exposes only trainable Actor parameters to the
  optimizer, and saves/loads a V18-tagged Actor-only state and model configuration. Loading requires
  exact V18 contract/schema values and performs no migration.
- **Rationale**: Callers no longer reproduce fragile parameter-name filtering, and integration tests
  can prove no Critic/value gradients or updates.
- **Alternatives considered**: Reusing the legacy resume loader was rejected because it accepts old
  contracts and saves full training state.

## Decision 9: Token-length validation without dataset generation

- **Decision**: Add a production statistics CLI that reads the established validation selection's
  archived offset metadata or streams representative V18 re-encoding without writing samples.
  Existing exact validation metadata contains 1,439,440 decisions: history 53.837822, V16 Snapshot
  6.124897, query pairs 8.482603. The initial 29-row Snapshot projected 99.803028; the active
  49-row Snapshot projects 118.803028 without materializing a dataset; the live-wall estimator was removed from the state suffix, one token per decision.
- **Rationale**: This is exact for sequence length because V18 preserves history/query organization
  and fixes Snapshot length; it satisfies the non-materialization constraint.
- **Alternatives considered**: Generating the full V18 selection is explicitly out of scope; a tiny
  synthetic-only fixture would provide weaker evidence than the existing full-selection offsets.

## Decision 10: Active-contract migration and archive policy

- **Decision**: Rename active V16-specific public code/tests to neutral or V18 names, make default
  config/docs/CLI V18, remove active Q and compatibility branches after repository-wide reference
  audits, and keep historical specs, reports, checkpoints, datasets, logs, and explicitly archival
  configs unchanged unless an active label must be corrected outside the historical artifact itself.
- **Rationale**: Constitution v1.8.0 allows only one active V18 contract while preserving evidence.
- **Alternatives considered**: Parallel V16/V18 implementations and compatibility flags are both
  explicitly prohibited.

## Decision 11: 49-row public Snapshot extension

- **Decision**: Keep the four placement rows first. For each relative opponent, append the six new
  public facts directly after the existing five opponent summary fields: four exact `0..6`
  first-six-river composition counts, `0..5,6+` confirmed open-meld yakuhai han, and
  `0..7,8+` visible-meld dora-plus-aka han. Place the two global categorical facts after the four
  shanten rows and before latest-tedashi/streak rows. The global domains are `0..24,25+` fully
  visible tile kinds and `0..15,16+` unknown copies of distinct dora kinds.
- **Rationale**: Per-opponent facts stay in the established seat-major summary region; global facts
  stay with derived state. All 20 rows remain fixed categorical tokens and preserve query isolation.
- **Alternatives considered**: Adding them to per-action Query rows would repeat state facts and let
  candidate count alter the public prefix; continuous count features would weaken the categorical
  field contract.

## Decision 12: Legal information and meld semantics

- **Decision**: Rust derives the extension using only own hand/melds, public rivers, public meld
  tiles, and dora indicators. It never reads opponent hand vectors or wall order. Yakuhai examines
  only opened melds; dora/aka uses all legally represented meld tiles, including ankan, while the
  existing `opened` predicate remains the sole menzen test. Dora kinds are advanced from indicators
  and deduplicated before unknown-copy counting.
- **Rationale**: This proves the Actor's public boundary while retaining actual wind-yakuhai han and
  visible dora information. Keeping all work in the native preparation loop avoids a Python hot path.
- **Alternatives considered**: Deriving counts from offline labels, opponent concealed tiles, or
  true wall state is forbidden; treating ankan as opened would corrupt menzen/shanten semantics.

## Decision 13: Remove the estimated remaining-wall token

- **Decision**: The Objective Facts state suffix no longer contains a live-wall (剩余牌山数) counter.
  The previous estimator initialized 70 and decremented only on tsumo events, which cannot reproduce
  kan-driven dead-wall changes or other exact wall sizes; since the MJAI event stream carries no
  reliable remaining-wall field, the token is dropped and the remaining counter fields are
  renumbered to a contiguous 1..7. The per-decision sequence projection becomes 118.803028.
- **Rationale**: An estimated feature is worse than no feature for learning; removing it costs
  exactly one fixed token per decision and keeps the public-input organization unchanged.
- **Alternatives considered**: Keeping the estimator with a documented caveat, or routing
  RiichiEnv-core `tiles_left` into the state-machine path, both break the single-source event-stream
  derivation that the state machine guarantees.

## Decision 14: Progress, riichi-trait, and turn facts

- **Decision**: Snapshot is extended from 49 to 54 rows. Each opponent's latest-tedashi tile is
  replaced by the riichi declaration tile (same 37-code vocabulary), a post-riichi tsumogiri count
  keeps the opponent summary at 13 rows per seat, and two self facts join the global region: the
  shanten-improving tile count and the tenpai winning tile count, both computed from a normalized
  13-tile shape against remaining copies in the legal known area. The exact current turn (public
  discard rounds + 1) joins the Objective Facts state suffix as a plain counter.
- **Rationale**: Progress facts are the strongest missing signal (the model previously had to derive
  improve/win counts through a long chain from raw hand + rivers); per-opponent riichi traits are
  direct defense cues; the turn replaces the removed estimated wall count with an exact one.
- **Alternatives considered**: Keeping latest-tedashi as a duplicated event-prefix fact; adding
  estimated opponent danger or open-meld turn via event JSON parsing into the hot path; both are
  rejected (redundancy or estimation vs. the exact derivation rule).
