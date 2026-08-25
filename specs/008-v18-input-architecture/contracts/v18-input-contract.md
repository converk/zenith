# Contract: V18 Input, Query, and Model Boundary

## Version

- Encoding protocol: 18
- Encoded format: `riichi-sft-encoded-v18`
- Only this contract is accepted by active save/load and validation paths.

## Snapshot

- `snapshot_factors`: `[B,54,4]`, integer columns
  `(field_id, relative_seat, categorical_value, tile_value)`.
- `snapshot_numeric`: `[B,54,1]`, finite float.
- `snapshot_lengths`: `[B]`, identically 54.
- Field positions and value domains are defined once in the Rust active protocol table and exported
  as machine-readable metadata consumed by every Python layer.
- Per-opponent rows are seat-major: base five summary fields, first-six discard man/pin/sou/
  terminal-or-honor counts (`0..6`), open-meld yakuhai (`0..5,6+`), and visible-meld dora-plus-aka
  (`0..7,8+`). Global rows after shanten are fully visible kinds (`0..24,25+`) and unknown copies
  across distinct dora kinds (`0..15,16+`). Zero is the valid non-missing value for all 20 additions.

## Query pairs

- `query_rows`: `[B,2*A,15]`; rows `(2*i,2*i+1)` are Offense/Defense for one action.
- `query_action_ids`: `[B,A]`; valid prefixes contain unique IDs whose set equals legal-mask IDs.
- Both rows contain the same action ID/type/primary tile/source seat.
- Supplier actions chi/pon/daiminkan/ron require relative source 1..3; all others require N/A.
- `raw_policy_logits[B,action_id]` is the sole raw-logit mapping.

## Attention

- Three shared layers consume public prefix only.
- Actor layer public rows use causal public attention and cannot see Query rows.
- Each Query row sees every valid public row and both rows of its own pair only.
- Every pair shares the same two local position IDs.

## Information boundary

- Actor accepts public/self facts and legal-action Queries only. The new rows use no opponent
  concealed-hand, true-wall, or post-hoc information; duplicate dora kinds are de-duplicated for
  the global unknown-copy field.
- Critic accepts shared public states, private hands in relative order 1/2/3, future wall positions
  1..5, then value query. It receives no action Query.

## Model and persistence

- Topology: `d_model=256`, Q heads 16, KV heads 4, head dimension 16, FFN 704,
  shared/Actor/Critic layers 3/1/2.
- Actor-Critic parameter scope is 4.9M–5.1M and contains no Q module or key.
- Actor-only BC saves and loads exact V18 Actor artifacts only; no legacy fallback or migration.
