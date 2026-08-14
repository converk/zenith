# v13 isolated-action-query SFT

V13 is the sole supported preprocessing and training contract. It keeps the
4-layer, 192-wide decoder, fixed 241-action protocol, isolated two-token action
queries, and the existing multi-task loss. Schema 12 is deliberately rejected.

## Frozen feature contract

Categorical zero means N/A unless the row kind or another field in the same row
explicitly marks the value as known. Shanten categories use `value + 2`, so agari is 1,
tenpai is 2, and N/A is 0. Numeric scaling and bit allocations are defined by
`model/feature_schema.py`. New caches and checkpoints bind these semantics with
the single `riichi-sft-v13-1` contract identifier. The immutable formal cache's
older exact metadata tuple is recognized only at the contract boundary.

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

v11 兼容已移除:历史 v11 checkpoint 仅作冷存储保留,不提供权重评测或训练路径;
当前唯一契约是 v13。

## 0.4% semantic canary

Run this before the full cache. Its manifest records per-slot
N/A/absolute-saturation/out-of-range counts plus legal/expert coverage for every action
group and all 241 action ids.

```bash
conda run -n Mahjong-AI python -m riichi_ppo_v1.sft.precompute \
  --source datasets/tenhou_sft_2024_2025 \
  --output datasets/tenhou_sft_canary_v13 \
  --subset-denominator 5 --subset-remainders 0,1 \
  --game-sample-denominator 100 --game-sample-remainder 0 \
  --workers 16 --require-complete-action-coverage
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
  --workers 16
```

性能验证使用 PPO smoke：`conda run -n Mahjong-AI riichi-ppo-smoke --device cuda`，
跑三轮并单独报告第 2–3 轮。PPO smoke 使用 `target_kl=0.0`、`update_epochs=4`、
`kyokus_per_worker=1`；长期训练保持 `kyokus_per_worker=16`。

The expected full-cache totals are 94,009,417 training decisions and 951,475
validation decisions. Stop before training if the actual totals or selection
manifest differ from the audited V13 preparation run.
