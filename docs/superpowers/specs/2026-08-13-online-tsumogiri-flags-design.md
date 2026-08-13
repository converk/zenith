# Online `tsumogiri_flags` authoritative reconstruction

## Goal

Make the online bot's actor input use the same `tsumogiri_flags` semantics as
local PPO rollout and 1v3 evaluation, even when the server includes the field
but serializes it as four empty lists.

## Scope

Change only `riichi_lab_bot` observation normalization at the
`OnlineStateBridge` boundary. Do not change RiichiEnv, PPO training, model
weights, the actor feature schema, or unrelated reconstructed fields.

## Design

`ThreatSnapshotTracker` remains the source of truth. It resets all four rows on
`start_kyoku` and appends each public `dahai.tsumogiri` value to the discarding
player's row.

Add `tsumogiri_flags` to the fields that `OnlineStateBridge.prepare()` always
overrides from the tracker. This is intentionally stronger than the existing
"fill only when absent" compatibility behavior: a present but semantically
incomplete server value must not suppress event reconstruction.

The reconstructed value is applied before `DecisionAnalysisBatch.state_tokens`
encodes the three opponent threat rows. No token layout, action mapping, legal
mask, or query construction changes.

## Error and lifecycle behavior

- `start_kyoku` clears prior-kyoku flags, preventing cross-kyoku residue.
- An empty flag table at the start of a kyoku remains valid and is overwritten
  with the same empty tracker value.
- Event deltas remain the authoritative input; malformed or unsupported events
  retain the tracker's existing ignore behavior.
- Reconnection and missing-delta behavior are outside this narrowly scoped fix.

## Tests

Use test-driven development:

1. Add a regression test whose serialized Observation already contains
   `tsumogiri_flags=[[], [], [], []]`, while its event delta contains real
   `dahai.tsumogiri` values. Before the fix, the prepared Observation must
   incorrectly retain the empty rows.
2. After the fix, assert that the prepared Observation contains the tracker
   reconstruction.
3. Assert that the segment-6 threat categorical and numeric streak slots encode
   the reconstructed trailing tsumogiri streak.
4. Run the focused regression test, the full bot bridge/client suite, and the
   training bridge integration tests.

## Acceptance criteria

- A present but empty online `tsumogiri_flags` field is always replaced by the
  event-derived value.
- The resulting actor threat token matches the local training semantics.
- Existing online observation compatibility and training bridge tests pass.
- No production file outside the online bot bridge is modified.
