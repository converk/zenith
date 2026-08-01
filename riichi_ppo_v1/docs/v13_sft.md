# v13 isolated-action-query SFT

Schema 13 is the corrected successor to the incomplete schema 12. It keeps the
4-layer, 192-wide decoder, fixed 241-action protocol, isolated two-token action
queries, and the existing multi-task loss. Schema 12 is deliberately rejected.

## Frozen feature contract

Categorical zero means N/A unless the row kind or another field in the same row
explicitly marks the value as known. Shanten categories use `value + 2`, so agari is 1,
tenpai is 2, and N/A is 0. Numeric scaling and bit allocations are defined by
`model/feature_schema.py`; its canonical JSON hash is stored in every cache and
checkpoint together with decision-analysis version 16 and the Rust analysis
version.

The six state rows are current hand, current tenpai/value, placement, and the
right/across/left opponent threats. They are computed from the current hand and
public event history, never from the best candidate. Every legal action has an
adjacent offense/defense query pair. Terminal actions carry explicit N/A.
Current-hand ukeire is applicable only to a normalized 13-tile post-action
shape; a 14-tile self-draw state is explicitly N/A. All schema-13 numeric fields
are clipped to their declared range and encoding aborts if a value escapes it.
Equal indicators retain their multiplicity: two indicators selecting the same
tile type make every copy count as two dora. State hand/exposed-dora numerics
clip at 8, while the categorical preserved-dora query clips at 6 before its
one-based encoding.
Complete suji requires every applicable anchor (both sides for middle tiles),
and kabe strength comes from related suited tiles rather than the candidate.

## Matched v11/v13 ablation caches

Use identical subset parameters for both encodings. Their manifests must have
the same `selection_manifest_sha256`. Encoded chunks retain
`year/game_id/kyoku_index/seat/decision_index` for every sample. Each manifest
hashes the complete flat identity sequence, expert actions, legal masks, and
exact chunk layout. At training startup every ablation config requires the
paired cache and rejects any selection, supervision, identity-order, count, or
chunk-layout mismatch. The configs
also disable length-based reordering, so identical seeds produce identical
sample batches despite different v11/v13 token lengths. Schema 11 records a
frozen legacy-encoder hash over the concrete Rust replay/public-observation,
hand-analysis, scoring, and Python encoding sources. Its cache loader also
checks the replay-semantics version exported by the installed Rust extension.
Subset assignment uses a domain-separated game-id hash, so every kyoku from
one game stays in the same remainder and the v11/v13 game lists are identical.
The independent canary hash cannot accidentally correlate with the source
train/validation split or with the production subset remainder.

```bash
conda run -n Mahjong-AI python -m riichi_ppo_v1.sft.precompute \
  --source datasets/tenhou_sft_2024_2025 \
  --output datasets/tenhou_sft_ablation_v11 \
  --subset-denominator 5 --subset-remainders 0,1 \
  --game-sample-denominator 8 --game-sample-remainder 0 \
  --token-schema 11 --workers 16 --require-identity-contract

conda run -n Mahjong-AI python -m riichi_ppo_v1.sft.precompute \
  --source datasets/tenhou_sft_2024_2025 \
  --output datasets/tenhou_sft_ablation_v13 \
  --subset-denominator 5 --subset-remainders 0,1 \
  --game-sample-denominator 8 --game-sample-remainder 0 \
  --token-schema 13 --workers 16 --require-identity-contract
```

The four configs are `sft_ablation_v11_global_ce.yaml`,
`sft_ablation_v13_global_ce.yaml`, `sft_ablation_v13_isolated_ce.yaml`, and
`sft_ablation_v13_isolated_multitask.yaml`.

## 0.4% semantic canary

Run this before the ablations or full cache. Its manifest records per-slot
N/A/absolute-saturation/out-of-range counts plus legal/expert coverage for every action
group and all 241 action ids.

```bash
conda run -n Mahjong-AI python -m riichi_ppo_v1.sft.precompute \
  --source datasets/tenhou_sft_2024_2025 \
  --output datasets/tenhou_sft_canary_v13 \
  --subset-denominator 5 --subset-remainders 0,1 \
  --game-sample-denominator 100 --game-sample-remainder 0 \
  --token-schema 13 --workers 16 --require-complete-action-coverage
```

Do not accept the canary unless every legal and expert action group, including
`ryukyoku`, is non-zero and `numeric_abs_gt_1_by_slot` is all zero. The verified
2026-08-01 run contains 951,793 decisions, including 40 legal and 36 expert
action-240 rows.

## Full cache and performance checks

```bash
conda run -n Mahjong-AI python -m riichi_ppo_v1.sft.precompute \
  --source datasets/tenhou_sft_2024_2025 \
  --output datasets/tenhou_sft_2024_2025_encoded_40pct_v13 \
  --subset-denominator 5 --subset-remainders 0,1 \
  --token-schema 13 --workers 16

CUDA_DEVICE=0,3 conda run -n Mahjong-AI python -m riichi_ppo_v1.tools.benchmark_sft \
  --dataset datasets/tenhou_sft_2024_2025_encoded_40pct_v13 \
  --config riichi_ppo_v1/configs/sft.yaml --steps 3
```

Run three performance iterations and treat the first as warm-up. PPO smoke
tests use `target_kl=0.0`, `update_epochs=4`, and `kyokus_per_worker=1`; long
training keeps `kyokus_per_worker=16`.

The expected full-cache totals are 94,009,417 training decisions and 951,475
validation decisions. Stop before training if the actual totals or selection
manifest differ from the matched v11/v13 preparation run.
