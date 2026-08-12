# Step 2.5 — High-Budget Teacher Override Audit (Full-Hanchan MC)

This directory implements GOAL_PROMPT_STEP2.5.md: auditing the Step-2
high-confidence Teacher overrides (rollout-to-kyoku-end + GRP V2) against a
stronger, GRP-independent ground truth — paired full-hanchan Monte Carlo
expected utility with the actual final rank as reward.

## Pipeline

```text
select_audit_samples.py -> results/audit_samples.csv (30 overrides + 30 matched keeps)
fullmc_audit.py         -> results/audit_summary_shard*.csv,
                           branch_outcomes_shard*.csv, paired_deltas_shard*.csv,
                           stability_shard*.csv, fullmc_summary_shard*.json
correctness_audit.py    -> results/fullmc_correctness_audit.json
analyze.py              -> override_audit_results.csv, keep_control_audit_results.csv,
                           teacher_fullmc_calibration.csv, calibration_buckets.csv,
                           rank4_override_audit.csv, gap_override_audit.csv,
                           threshold_sweep.csv, experiment_summary.json, report.md
```

## Commands

```bash
# 1. Sample selection (fixed audit set, before any simulation)
conda run -n Mahjong-AI python select_audit_samples.py

# 2. Full-hanchan paired MC on two GPUs (physical 0 and 3); two processes per
#    GPU (4 shards) speeds up this CPU-bound pipeline.
for s in 0 1 2 3; do
  if [ $((s % 2)) -eq 0 ]; then dev=0; else dev=2; fi
  CUDA_DEVICE=$dev conda run -n Mahjong-AI python fullmc_audit.py \
    --device cuda:0 --shard-id $s --num-shards 4 --resume --append &
done
wait

# 3. Runtime correctness audit (renchan/ryuukyoku/honba/kyotaku/south/final)
CUDA_DEVICE=0 conda run -n Mahjong-AI python correctness_audit.py \
  --device cuda:0 --limit 8 --worlds 0,1,2,3,4

# 4. Analysis + report
conda run -n Mahjong-AI python analyze.py
```

## Key design decisions

1. **Audit set**: all 30 Step-2 high-confidence overrides
   (Teacher Best Rank2/3/4 = 11/9/10) plus 30 conservative-keep controls
   matched on gap bucket (exact), kyoku stage, π1, Step2 |ΔGRP|
   (rank-transformed), Step2 world budget and action type. The set is fixed
   before any full-hanchan simulation.
2. **Counterfactual**: A = Policy Top1, B = Teacher Best (overrides) or the
   strongest Step-2 challenger (keeps). Both branches share every sampled
   world and only the forced first action differs.
3. **Continuation**: PPO v2 (`checkpoint_00100.pt`), greedy/argmax, for every
   player until the whole hanchan ends (`4p-red-half`, including renchan,
   ryuukyoku, honba, kyotaku, south entry, tobi and the env's 30000-target
   extension rule). The GRP V2 leaf value is never used.
4. **World sampling**: the Step-2 uniform baseline (tile-type conservation,
   legal hand sizes, fixed dora slots, MJAI/env consistency). Each world is
   constructed with a wall seed so branch clones use the same future-shuffle
   RNG stream (matched randomness); walls match exactly whenever the branches
   have identical public tile multisets at the same round.
5. **Reward**: the target seat's actual final rank utility
   (1st=+10, 2nd=+4, 3rd=-4, 4th=-10). Paired differences D = R_B - R_A are
   the primary statistic.
6. **Adaptive budget**: 64 → 128 → 256 worlds for overrides and keep
   controls (cap 256), stopping when the z=1.96 CI of D excludes 0 for
   two consecutive waves; at the cap, CI-includes-0 decisions are
   UNRESOLVED (near-tie when |mean ΔFull| ≤ 0.5).
7. **Statistics**: the decision is the unit; per-world rollouts only estimate
   each decision's paired ΔFull. Override Precision uses a Wilson CI;
   correlations use Pearson/Spearman on decision-level predicted ΔGRP vs
   observed ΔFull.

> Note: the 2026-08-12 run's `results/audit_samples.csv` reused four keep
> decisions during greedy matching (the same A/B pair appears twice, matched
> to different overrides). `analyze.py` deduplicates keep controls by
> `decision_id` (keeping the closer match); `select_audit_samples.py` now
> enforces strictly one-to-one matching for future runs.

## Reproducibility

All random seeds (sample selection, world seeds, wall seeds), decision
identities, checkpoints, adaptive stopping parameters and per-world branch
outcomes are recorded under `results/`. `fullmc_config.json` stores the run
configuration; `audit_summary_shard*.csv` stores every audited decision's
Step-2 features together with the full-hanchan statistics.

## Step-2 data dependency

`select_audit_samples.py` reads
`audit/reports/grp_ranker_20260811/step2_top4_rollout/results/decision_summary.csv`
and never re-runs the Step-2 kyoku-ending rollout.
