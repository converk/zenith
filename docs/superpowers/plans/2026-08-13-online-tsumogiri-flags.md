# Online Tsumogiri Flags Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the online bot always replace the server's potentially empty `tsumogiri_flags` snapshot with the authoritative value reconstructed from MJAI event deltas.

**Architecture:** Keep `ThreatSnapshotTracker` unchanged and extend the existing authoritative override set in `OnlineStateBridge.prepare()`. Add one focused semantic regression test that starts with a present-but-empty server field, feeds a trailing tsumogiri streak through events, and verifies both the prepared Observation and segment-6 threat token.

**Tech Stack:** Python 3.12, pytest, NumPy, RiichiEnv native Observation, `OnlineStateBridge`.

## Global Constraints

- Do not change RiichiEnv, PPO training, model weights, actor schema, legal masks, or candidate-query construction.
- Reset behavior remains owned by `ThreatSnapshotTracker.apply_event(start_kyoku)`.
- Modify only the online bridge production code and its focused tests.
- Use the Conda environment `Mahjong-AI` for Python and test commands.

---

### Task 1: Regress present-but-empty server flags

**Files:**
- Modify: `riichi_lab_bot/tests/test_bridge_semantics.py`
- Modify: `riichi_lab_bot/src/riichi_lab_bot/bridge.py:145-153`

**Interfaces:**
- Consumes: `ThreatSnapshotTracker.fields() -> dict[str, Any]` and `ObservationView.set_fields(fields: dict[str, Any]) -> None`.
- Produces: `OnlineStateBridge.prepare(observation)` whose returned `PreparedDecision.observation.tsumogiri_flags` is always event-derived.

- [ ] **Step 1: Write the failing regression test**

Add a test that serializes a valid local Observation with the server field explicitly present as four empty rows, while the event delta contains one tedashi followed by two tsumogiri discards from player 1:

```python
def test_present_empty_server_tsumogiri_flags_are_overridden() -> None:
    from riichienv import Observation, RiichiEnv

    observation = RiichiEnv(game_mode="4p-red-half", seed=42).reset()[0]
    data = json.loads(
        base64.b64decode(observation.serialize_to_base64()).decode("utf-8")
    )
    data["tsumogiri_flags"] = [[], [], [], []]
    data["events"].extend([
        json.dumps({"type": "dahai", "actor": 1, "pai": "1p", "tsumogiri": False}),
        json.dumps({"type": "dahai", "actor": 1, "pai": "2p", "tsumogiri": True}),
        json.dumps({"type": "dahai", "actor": 1, "pai": "3p", "tsumogiri": True}),
    ])
    encoded = base64.b64encode(
        json.dumps(data, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    server_observation = ObservationView(
        Observation.deserialize_from_base64(encoded),
        missing_fields=missing_observation_fields(encoded),
    )

    prepared = OnlineStateBridge(0).prepare(server_observation)

    assert prepared.observation.tsumogiri_flags[1] == [False, True, True]
    threat = prepared.token_factors[
        (prepared.token_factors[:, 0] == 6)
        & (prepared.token_factors[:, 1] == 4)
        & (prepared.token_factors[:, 2] == 1)
    ]
    threat_numeric = prepared.token_numeric[
        (prepared.token_factors[:, 0] == 6)
        & (prepared.token_factors[:, 1] == 4)
        & (prepared.token_factors[:, 2] == 1)
    ]
    assert int(threat[0, 6]) == 2
    assert float(threat_numeric[0, 3]) == pytest.approx(2 / 12)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
conda run -n Mahjong-AI pytest -q \
  riichi_lab_bot/tests/test_bridge_semantics.py::test_present_empty_server_tsumogiri_flags_are_overridden
```

Expected: FAIL because `prepared.observation.tsumogiri_flags[1]` remains empty.

- [ ] **Step 3: Implement the minimal authoritative override**

In `OnlineStateBridge.prepare()`, add the reconstructed field to the existing always-applied override dictionary:

```python
riichi_overrides = {
    "riichi_declared": derived_fields["riichi_declared"],
    "riichi_accepted": derived_fields["riichi_accepted"],
    "riichi_declaration_indices": derived_fields[
        "riichi_declaration_indices"
    ],
    "riichi_sutehais": derived_fields["riichi_sutehais"],
    "tsumogiri_flags": derived_fields["tsumogiri_flags"],
}
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
conda run -n Mahjong-AI pytest -q \
  riichi_lab_bot/tests/test_bridge_semantics.py::test_present_empty_server_tsumogiri_flags_are_overridden \
  riichi_lab_bot/tests/test_bridge_semantics.py::test_server_observation_with_missing_snapshot_fields_prepares
```

Expected: `2 passed`.

- [ ] **Step 5: Run the relevant regression suites**

Run:

```bash
conda run -n Mahjong-AI pytest -q \
  riichi_lab_bot/tests/test_bridge_semantics.py \
  riichi_lab_bot/tests/test_bridge_integration.py \
  riichi_lab_bot/tests/test_client.py \
  riichi_ppo_v1/tests/integration/test_bridge_integration.py \
  riichi_ppo_v1/tests/unit/test_feature_schema_v13.py
```

Expected: all tests pass without warnings or errors.

- [ ] **Step 6: Verify diff scope and commit**

Run:

```bash
git diff --check
git status --short
git diff -- riichi_lab_bot/src/riichi_lab_bot/bridge.py \
  riichi_lab_bot/tests/test_bridge_semantics.py
```

Expected: only the bridge, focused test, and this implementation-plan document differ from the design commit.

Commit:

```bash
git add riichi_lab_bot/src/riichi_lab_bot/bridge.py \
  riichi_lab_bot/tests/test_bridge_semantics.py \
  docs/superpowers/plans/2026-08-13-online-tsumogiri-flags.md
git commit -m "fix: reconstruct online tsumogiri flags"
```
