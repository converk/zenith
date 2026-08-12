# Step 2 — Top4 Paired Counterfactual Rollout + GRP V2 Teacher

This directory implements GOAL_PROMPT_STEP2.md: validating whether an
expensive counterfactual teacher (paired rollout to kyoku end + GRP V2) can
rerank the Policy Top4 candidates better than Policy probability alone.

## Pipeline

```text
select_candidates.py  -> policy_candidates.csv, selected_decisions.csv
paired_rollout.py     -> decision_summary_shard*.csv, stability_shard*.csv,
                         candidate_values_shard*.csv, rollout_summary_shard*.json
semantic_audit.py     -> semantic_audit.json
analyze.py            -> merged summaries, gap/top3-vs-top4/threshold tables,
                         experiment_summary.json, report.md
```

## Commands

```bash
# 1. Candidate sweep + stratified sampling (50k validation decisions, seed 20260811)
CUDA_DEVICE=0 python select_candidates.py --device cuda:0 --max-samples 50000 --seed 20260811

# 2. Paired rollout on two GPUs (physical 0 and 3)
CUDA_DEVICE=0 python paired_rollout.py --device cuda:0 --shard-id 0 --num-shards 2
CUDA_DEVICE=2 python paired_rollout.py --device cuda:0 --shard-id 1 --num-shards 2

# 3. Semantic audit (optional but recommended)
CUDA_DEVICE=0 python semantic_audit.py --device cuda:0

# 4. Analysis + report
python analyze.py
```

## Key design decisions

1. **Candidate policy**: v13 SFT (`best_heuristic.pt`); Top4 probabilities are
   normalized over legal actions only, exactly like the recall analysis in the
   goal prompt.
2. **Continuation policy**: PPO v2 (`checkpoint_00100.pt`), greedy (argmax),
   which removes sampling noise from branch comparisons.
3. **World sampling (baseline)**: the remaining tile multiset is sampled
   uniformly into the three opponents' hands and the wall, with revealed dora
   slots fixed. The sampled world is materialized by rewriting the MJAI event
   stream (per-player draws) so that the RiichiEnv and the batched policy
   state machine replay exactly the same world. This sampler is a baseline:
   it does not condition on behavioral cues (defense, riichi, call patterns).
4. **Policy features**: the batched state machine is fed the sampled world's
   MJAI event history directly from the replayed environment (self-consistent
   env/bridge pairing).  A fidelity audit (`semantic_audit.py`) additionally
   re-encodes real decision states through the training-time `Kyoku.steps`
   streams; the remaining differences are near-tie reshuffles only.
   Legal-but-edge hands (kan-containing, 15-17 physical tiles) that the
   project's strict 13/14-tile analyzer cannot represent get a neutral
   analysis fallback so rollouts can continue.
5. **Paired comparison**: all four candidates share every sampled world;
   only the forced first action differs. The teacher statistic is the
   world-matched paired difference vs Policy Top1.
6. **Adaptive budget**: 16 -> 32 -> ... worlds, stopping when every challenger
   vs Top1 pair is resolved at z=1.96 or the cap (64; 128 for the stability
   subset) is reached. Verdicts: `keep_top1` / `override_topN` / `uncertain`.

## Reproducibility

All random seeds (policy sweep, world sampling), decision identities, policy /
GRP checkpoints, continuation configuration, and per-world per-candidate GRP
values are recorded in the output files. `selected_decisions.csv` contains the
full policy features (π1-π4, gaps, entropy, top4 cumulative probability).

## Step-2 final run (2026-08-12)

* 420 stratified decisions, ~109k paired rollouts (Top4 branches × adaptive
  worlds), ~5.8M continuation policy decisions on two L20 GPUs
  (physical GPUs 0 and 3), ~2.2h wall time, ~355-360 decisions/s per shard.
* Conclusion: **CONDITIONAL GO** — the GRP teacher is stable and finds
  low-gap overrides (9.2% in gap<0.05 vs 5.0% in gap>=0.70), but the reliable
  high-confidence override region has limited coverage; a future reranker must
  gate on |mean ΔGRP| and policy gap. Details in `results/report.md`.
