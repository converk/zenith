# Quickstart: Validate V18 Input Architecture

Run from the repository root. Every Python command uses the `Mahjong-AI` Conda environment. These
checks do not generate a complete dataset, start formal SFT, or run PPO.

## 1. Contract and unit validation

```bash
conda run -n Mahjong-AI python -m pytest \
  riichi_ppo_v1/tests/unit/test_v18_snapshot.py \
  riichi_ppo_v1/tests/unit/test_v18_architecture.py \
  riichi_ppo_v1/tests/unit/test_v18_actor_sft.py
```

Expected: exactly 54 Snapshot rows; field/domain boundary cases pass; parameter count is 4.9M–5.1M;
state keys contain no Q names; pair permutations preserve action-ID-aligned logits; Actor-only BC
leaves Critic/value frozen and gradient-free.

## 2. Rust and bridge validation

```bash
conda run -n Mahjong-AI cargo test \
  --manifest-path RiichiEnv/riichienv-state-machine/Cargo.toml
conda run -n Mahjong-AI python -m pytest \
  riichi_ppo_v1/tests/integration/test_v18_encoding_bridge.py \
  riichi_ppo_v1/tests/integration/test_v18_replay_bridge.py \
  riichi_ppo_v1/tests/protocol
```

Expected: shanten/status/count/tile/streak derivation, real supplier seat, malformed-input rejection,
protocol mapping, and replay behavior all pass.

## 3. Information isolation

```bash
conda run -n Mahjong-AI python -m pytest \
  riichi_ppo_v1/tests/integration/test_v18_information_boundaries.py
```

Expected: hidden-only changes do not change Actor logits; valid private changes are consumed by the
Critic; private ordering, five-tile wall boundary, and masks are verified.

## 4. Non-materializing token statistics and production validation

```bash
conda run -n Mahjong-AI python -m riichi_ppo_v1.tools.v18_token_statistics \
  --dataset datasets/tenhou_sft_2024_2025_encoded_60pct_v16
conda run -n Mahjong-AI python -m riichi_ppo_v1.tools.validate \
  --games 1 --output audit/reports/v18/eval/v18_protocol_coverage.json
```

Expected: the established selection is verified, mean V18 total length is 116–122 (target about
118.80), parameter/state/schema checks pass, and any explicit output is placed under
`audit/reports/v18/` or `logs/v18/`.

## 5. Full affected suites

```bash
conda run -n Mahjong-AI python -m pytest riichi_ppo_v1/tests RiichiEnv/tests riichi_lab_bot/tests
```

Expected: all affected tests pass. Remove temporary smoke logs/results after validation and record
the reproducible commands/results in `audit/reports/v18/report/PROGRESS.md`.
