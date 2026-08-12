"""Step-2 candidate selection: v13 SFT policy sweep + stratified sampling.

This mirrors the recall analysis in GOAL_PROMPT_STEP2.md:

* sample 50,000 validation decisions (seed=20260811, no replacement);
* run the v13 SFT policy (best_heuristic.pt) and record Top4 actions,
  probabilities, gaps, entropy and expert recall;
* stratify-sample a few hundred decisions by the Top1-Top2 probability gap,
  oversampling the low-confidence buckets.

Outputs (under ``results/``):
* ``policy_candidates.csv``         all 50k decisions with policy features
* ``policy_sweep_summary.json``     recall + gap-bucket reproduction stats
* ``selected_decisions.csv``        stratified sample for the teacher
* ``selection_summary.json``        per-bucket sampling counts
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from riichi_ppo_v1.sft.checkpoint import load_v13_weights_only  # noqa: E402
from riichi_ppo_v1.sft.data import SftSample, iter_split_samples  # noqa: E402
from riichi_ppo_v1.sft.train import length_bucketed_batches  # noqa: E402


VALIDATION_SEED = 20260811
SWEEP_SAMPLES = 50_000

GAP_BUCKETS: list[tuple[str, float, float]] = [
    ("lt005", -float("inf"), 0.05),
    ("05_20", 0.05, 0.20),
    ("20_50", 0.20, 0.50),
    ("50_70", 0.50, 0.70),
    ("ge70", 0.70, float("inf")),
]

# High sampling for low gaps; keep some high-gap samples for confident-but-wrong.
DEFAULT_BUCKET_TARGETS: dict[str, int] = {
    "lt005": 120,
    "05_20": 100,
    "20_50": 100,
    "50_70": 60,
    "ge70": 40,
}


def gap_bucket(gap: float) -> str:
    for name, lo, hi in GAP_BUCKETS:
        if lo <= gap < hi:
            return name
    return "ge70"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _sample_rows(
    sample: SftSample,
    logits: torch.Tensor,
) -> dict[str, Any]:
    """Extract one policy row (identity + Top4 + gap features)."""
    legal = sample.legal_mask
    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()  # masked logits
    probs = np.maximum(probs, 0.0)
    order = np.argsort(-probs, kind="stable")
    top_ids = [int(a) for a in order if legal[int(a)]][:4]
    while len(top_ids) < 4:
        top_ids.append(-1)
    pi = [float(probs[a]) if a >= 0 else 0.0 for a in top_ids]
    logpi = [float(np.log(p)) if p > 0 else float("-inf") for p in pi]
    legal_probs = probs[legal]
    entropy = float(-np.sum(legal_probs * np.log(np.maximum(legal_probs, 1e-12))))
    expert = int(sample.action)
    expert_in_top4 = expert in top_ids
    return {
        "year": int(sample.year),
        "game_id": str(sample.game_id),
        "kyoku_index": int(sample.kyoku_index),
        "seat": int(sample.seat),
        "decision_index": int(sample.decision_index),
        "expert_action": expert,
        "top1_action": top_ids[0],
        "top2_action": top_ids[1],
        "top3_action": top_ids[2],
        "top4_action": top_ids[3],
        "pi1": pi[0],
        "pi2": pi[1],
        "pi3": pi[2],
        "pi4": pi[3],
        "logpi1": logpi[0],
        "logpi2": logpi[1],
        "logpi3": logpi[2],
        "logpi4": logpi[3],
        "gap12": pi[0] - pi[1],
        "gap13": pi[0] - pi[2],
        "gap14": pi[0] - pi[3],
        "log_gap12": logpi[0] - logpi[1],
        "log_gap13": logpi[0] - logpi[2],
        "log_gap14": logpi[0] - logpi[3],
        "policy_entropy": entropy,
        "top4_cum_prob": sum(pi),
        "expert_in_top4": bool(expert_in_top4),
        "expert_is_top1": bool(expert == top_ids[0]),
        "gap_bucket": gap_bucket(pi[0] - pi[1]),
    }


def sweep(
    dataset: Path,
    checkpoint: Path,
    device: torch.device,
    *,
    max_samples: int = SWEEP_SAMPLES,
    seed: int = VALIDATION_SEED,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    model = load_v13_weights_only(checkpoint, device=device)
    model.eval()
    samples = iter_split_samples(
        dataset,
        "validation",
        seed=seed,
        shuffle=True,
        include_critic=False,
    )
    rows: list[dict[str, Any]] = []
    batches = length_bucketed_batches(
        itertools.islice(samples, max_samples),
        batch_size,
        window_batches=32,
    )
    with torch.inference_mode():
        for batch in batches:
            max_tokens = max(sample.token_length for sample in batch)
            factors = torch.zeros((len(batch), max_tokens, 10), dtype=torch.uint8)
            numeric = torch.zeros((len(batch), max_tokens, 8), dtype=torch.float32)
            lengths = torch.empty(len(batch), dtype=torch.long)
            legal = torch.empty((len(batch), 241), dtype=torch.bool)
            for row, sample in enumerate(batch):
                factors[row, : sample.token_length] = torch.from_numpy(sample.token_factors)
                numeric[row, : sample.token_length] = torch.from_numpy(sample.token_numeric)
                lengths[row] = sample.token_length
                legal[row] = torch.from_numpy(sample.legal_mask)
            factors = factors.to(device)
            numeric = numeric.to(device)
            lengths = lengths.to(device)
            legal = legal.to(device)
            output = model.forward_policy(factors, numeric, legal, lengths)
            for sample, logits in zip(batch, output["policy_logits"], strict=True):
                rows.append(_sample_rows(sample, logits))
    return rows


def stratify(
    rows: list[dict[str, Any]],
    targets: dict[str, int] | None = None,
    *,
    seed: int = VALIDATION_SEED + 1,
) -> list[dict[str, Any]]:
    targets = targets or DEFAULT_BUCKET_TARGETS
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = {name: [] for name, _lo, _hi in GAP_BUCKETS}
    for row in rows:
        by_bucket[row["gap_bucket"]].append(row)
    selected: list[dict[str, Any]] = []
    for name in ("lt005", "05_20", "20_50", "50_70", "ge70"):
        pool = list(by_bucket[name])
        # Representative uniform sample within each gap bucket (no expert
        # label look-ahead; high-gap samples still get a small share).
        rng.shuffle(pool)
        take = min(int(targets.get(name, 0)), len(pool))
        chosen = pool[:take]
        selected.extend(chosen)
    selected.sort(key=lambda row: row["gap12"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "checkpoints/train_riichi_v13_sft/best_heuristic.pt"),
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=SWEEP_SAMPLES)
    parser.add_argument("--seed", type=int, default=VALIDATION_SEED)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bucket-targets", default=None, help="JSON dict of bucket->count")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or (Path(__file__).resolve().parent / "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    targets = json.loads(args.bucket_targets) if args.bucket_targets else None

    print(f"[select] sweeping {args.max_samples} validation decisions seed={args.seed}")
    rows = sweep(
        Path(args.dataset),
        Path(args.checkpoint),
        device,
        max_samples=args.max_samples,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    _write_csv(out_dir / "policy_candidates.csv", rows)
    print(f"[select] policy sweep done: {len(rows)} decisions")

    # Reproduce the recall / gap-bucket statistics from section 2.1.
    n = len(rows)
    summary: dict[str, Any] = {
        "seed": args.seed,
        "samples": n,
        "recall_at1": float(np.mean([r["expert_is_top1"] for r in rows])),
        "recall_at3": float(np.mean([r["expert_in_top4"] and (
            r["expert_action"] in (r["top1_action"], r["top2_action"], r["top3_action"])
        ) for r in rows])),
        "recall_at4": float(np.mean([r["expert_in_top4"] for r in rows])),
        "top3_miss": int(sum(not (
            r["expert_action"] in (r["top1_action"], r["top2_action"], r["top3_action"])
        ) for r in rows)),
        "top4_miss": int(sum(not r["expert_in_top4"] for r in rows)),
        "top4_of_top3_miss": int(sum(
            not (r["expert_action"] in (r["top1_action"], r["top2_action"], r["top3_action"]))
            and r["expert_action"] == r["top4_action"]
            for r in rows
        )),
        "gap_buckets": {},
        "pi1_mean": float(np.mean([r["pi1"] for r in rows])),
        "pi1_median": float(np.median([r["pi1"] for r in rows])),
        "gap12_mean": float(np.mean([r["gap12"] for r in rows])),
        "gap12_median": float(np.median([r["gap12"] for r in rows])),
    }
    for name, lo, hi in GAP_BUCKETS:
        subset = [r for r in rows if r["gap_bucket"] == name]
        if not subset:
            continue
        summary["gap_buckets"][name] = {
            "range": [None if lo == -float("inf") else lo, None if hi == float("inf") else hi],
            "n": len(subset),
            "top1_accuracy": float(np.mean([r["expert_is_top1"] for r in subset])),
            "top3_accuracy": float(np.mean([
                r["expert_action"] in (r["top1_action"], r["top2_action"], r["top3_action"])
                for r in subset
            ])),
            "top4_accuracy": float(np.mean([r["expert_in_top4"] for r in subset])),
        }
    (out_dir / "policy_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print("[select] sweep summary:", json.dumps(summary, indent=2, ensure_ascii=False))

    selected = stratify(rows, targets, seed=args.seed + 1)
    _write_csv(out_dir / "selected_decisions.csv", selected)
    sel_summary = {
        "seed": args.seed + 1,
        "selected": len(selected),
        "per_bucket": {name: 0 for name, _lo, _hi in GAP_BUCKETS},
        "expert_not_top1": int(sum(not r["expert_is_top1"] for r in selected)),
    }
    for row in selected:
        sel_summary["per_bucket"][row["gap_bucket"]] += 1
    (out_dir / "selection_summary.json").write_text(
        json.dumps(sel_summary, indent=2, ensure_ascii=False)
    )
    print("[select] selection:", json.dumps(sel_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
