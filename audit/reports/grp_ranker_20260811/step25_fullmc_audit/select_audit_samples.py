"""Step 2.5 sample selection: 30 high-confidence overrides + matched keeps.

Reads the Step 2 ``decision_summary.csv`` and emits the fixed audit set:

* all ``final_verdict_resolved95`` high-confidence overrides (30);
* ~30 conservative-keep controls matched on the variables listed in
  GOAL_PROMPT_STEP2.5.md section 6 (policy gap, kyoku stage, Top1
  probability, Step2 |mean ΔGRP|, simulation budget, action type).

The audit set is fixed before any full-hanchan simulation: no decision is
added/removed afterwards based on results.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
STEP2_RESULTS = (
    REPO_ROOT
    / "audit/reports/grp_ranker_20260811/step2_top4_rollout/results"
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _i(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def strongest_challenger_delta(row: dict[str, Any]) -> tuple[int, float]:
    """Return (rank, mean ΔGRP) of the strongest Step2 challenger vs Top1."""
    best_rank = 1
    best_delta = -float("inf")
    for rank, key in ((2, "delta_ba_mean"), (3, "delta_ca_mean"), (4, "delta_da_mean")):
        delta = _f(row.get(key))
        if math.isfinite(delta) and delta > best_delta:
            best_rank, best_delta = rank, delta
    return best_rank, best_delta


def teacher_best_delta(row: dict[str, Any]) -> float:
    rank = _i(row.get("teacher_best"))
    key = {2: "delta_ba_mean", 3: "delta_ca_mean", 4: "delta_da_mean"}.get(rank)
    return _f(row.get(key)) if key else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--n-keep-controls", type=int, default=30,
        help="target number of matched conservative-keep controls",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir or Path(__file__).resolve().parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(STEP2_RESULTS / "decision_summary.csv")
    overrides = [r for r in rows if str(r.get("verdict95", "")).startswith("override")]
    keeps = [r for r in rows if str(r.get("verdict95", "")) == "keep_top1"]
    overrides.sort(key=lambda r: (str(r.get("decision_id", ""))))
    keeps.sort(key=lambda r: (str(r.get("decision_id", ""))))

    if len(overrides) != 30:
        raise RuntimeError(
            f"expected exactly 30 high-confidence overrides, got {len(overrides)}"
        )

    # Matching features (all available before any full-hanchan simulation).
    def features_raw(row: dict[str, Any]) -> dict[str, float] | None:
        challenger_rank, challenger_delta = strongest_challenger_delta(row)
        if challenger_rank == 1 and not math.isfinite(challenger_delta):
            # No finite Step-2 challenger data: no meaningful audited B exists.
            return None
        teacher_rank = _i(row.get("teacher_best"))
        effect = (
            abs(teacher_best_delta(row))
            if teacher_rank > 1
            else abs(challenger_delta)
        )
        if not math.isfinite(effect):
            effect = 0.0
        return {
            "gap12": _f(row.get("gap12")),
            "kyoku_index": float(_i(row.get("kyoku_index"))),
            "pi1": _f(row.get("pi1")),
            "max_abs_delta_grp": effect,
            "n_worlds": float(_i(row.get("n_worlds"))),
            "top1_tsumogiri": float(_i(row.get("top1_action")) % 2 == 0),
            "expert_is_top1": float(str(row.get("expert_is_top1", "")).lower() == "true"),
            "seat": float(_i(row.get("seat"))),
        }

    keep_feats = {}
    for keep in keeps:
        feat = features_raw(keep)
        if feat is not None:
            keep_feats[keep["decision_id"]] = feat
    keeps = [r for r in keeps if r["decision_id"] in keep_feats]
    skipped_keeps = 143 - len(keeps)
    if len(keeps) < args.n_keep_controls:
        raise RuntimeError(
            f"not enough matchable keep controls: {len(keeps)} < {args.n_keep_controls}"
        )
    # Rank-transform |ΔGRP| against the keep pool: the keep pool simply does
    # not contain effects as large as the strongest overrides (that is itself
    # a Step-2 finding), and raw z-distances would let that one variable
    # dominate every match.  The transform preserves ordering while keeping
    # the matched controls representative of the pool's effect distribution.
    effect_values = sorted(feat["max_abs_delta_grp"] for feat in keep_feats.values())

    def effect_rank(value: float) -> float:
        return float(
            sum(1 for other in effect_values if other < value)
            + 0.5 * sum(1 for other in effect_values if other == value)
        ) / float(len(effect_values))

    def features(row: dict[str, Any]) -> dict[str, float] | None:
        feat = features_raw(row)
        if feat is None:
            return None
        feat["max_abs_delta_grp"] = effect_rank(feat["max_abs_delta_grp"])
        return feat

    names = sorted(keep_feats[next(iter(keep_feats))])
    # z-score normalization against the keep pool (the matching population).
    std = {
        name: float(np.std([feat[name] for feat in keep_feats.values()]) or 1.0)
        for name in names
    }
    weights = {
        "gap12": 1.0,
        "kyoku_index": 0.8,
        "pi1": 1.0,
        "max_abs_delta_grp": 1.2,
        "n_worlds": 0.4,
        "top1_tsumogiri": 0.3,
        "expert_is_top1": 0.3,
        "seat": 0.3,
    }

    def distance(override: dict[str, Any], keep: dict[str, Any]) -> float:
        a, b = features(override), features(keep)
        total = 0.0
        for name in names:
            diff = (a[name] - b[name]) / std[name]
            total += weights[name] * diff * diff
        return math.sqrt(total)

    # Exact gap-bucket alignment + greedy nearest keep, strictly one-to-one:
    # each keep control is audited once (the keep pools are large enough in
    # every gap bucket, so replacement is never needed).
    matched: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    used: set[str] = set()
    for override in overrides:
        bucket = str(override.get("gap_bucket"))
        candidates = []
        for keep in keeps:
            if (
                str(keep.get("gap_bucket")) != bucket
                or keep["decision_id"] in used
            ):
                continue
            candidates.append((keep, distance(override, keep)))
        candidates.sort(key=lambda pair: pair[1])
        if not candidates:
            raise RuntimeError(
                f"no keep control in gap bucket {bucket} for {override['decision_id']}"
            )
        keep, dist = candidates[0]
        used.add(keep["decision_id"])
        matched.append((override, keep, dist))

    samples: list[dict[str, Any]] = []
    for override, keep, dist in matched:
        tb_rank = _i(override.get("teacher_best"))
        b_rank = tb_rank
        samples.append(
            {
                **override,
                "group": "override",
                "audit_b_rank": b_rank,
                "audit_b_action": override.get(f"top{b_rank}_action"),
                "matched_override_id": override["decision_id"],
                "match_distance": round(dist, 4),
            }
        )
        challenger_rank, _challenger_delta = strongest_challenger_delta(keep)
        if challenger_rank == 1:
            raise RuntimeError(
                f"matched keep {keep['decision_id']} has no challenger data"
            )
        samples.append(
            {
                **keep,
                "group": "keep_control",
                "audit_b_rank": challenger_rank,
                "audit_b_action": keep.get(f"top{challenger_rank}_action"),
                "matched_override_id": override["decision_id"],
                "match_distance": round(dist, 4),
            }
        )

    samples.sort(key=lambda r: (r["group"], r["decision_id"]))
    _write_csv(out_dir / "audit_samples.csv", samples)

    n_keep = sum(1 for r in samples if r["group"] == "keep_control")
    n_ov = sum(1 for r in samples if r["group"] == "override")
    summary = {
        "n_override": n_ov,
        "n_keep_control": n_keep,
        "override_teacher_best": dict(
            sorted(Counter(_i(r.get("teacher_best")) for r in overrides).items())
        ),
        "override_gap_buckets": dict(
            sorted(Counter(str(r.get("gap_bucket")) for r in overrides).items())
        ),
        "keep_gap_buckets": dict(
            sorted(
                Counter(
                    str(r.get("gap_bucket"))
                    for r in samples
                    if r["group"] == "keep_control"
                ).items()
            )
        ),
        "mean_match_distance": round(
            float(np.mean([d for _, _, d in matched])), 4
        ),
        "max_match_distance": round(
            float(np.max([d for _, _, d in matched])), 4
        ),
        "skipped_keeps_no_challenger": skipped_keeps,
    }
    (out_dir / "selection_summary.json").write_text(
        __import__("json").dumps(summary, indent=2, ensure_ascii=False)
    )
    print("audit samples:", n_ov, "overrides +", n_keep, "keeps")
    print("override teacher best:", summary["override_teacher_best"])
    print("keep gap buckets:", summary["keep_gap_buckets"])


if __name__ == "__main__":
    main()
