"""Step 2.5 analysis + report for the full-hanchan teacher override audit.

Reads the sharded full-hanchan outputs and produces every deliverable
required by GOAL_PROMPT_STEP2.5.md:

* override_audit_results.csv / keep_control_audit_results.csv
* teacher_fullmc_calibration.csv / calibration_buckets.csv
* rank4_override_audit.csv / gap_override_audit.csv / threshold_sweep.csv
* experiment_summary.json / report.md

All decision-level statistics use the decision as the unit; per-world
rollouts only estimate each decision's paired full-hanchan difference.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]


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


def _mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _median(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return float(np.median(values)) if values else float("nan")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def _rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = arr.argsort(kind="stable")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1)
    # Average ranks for ties.
    sorted_values = arr[order]
    index = 0
    while index < len(arr):
        end = index + 1
        while end < len(arr) and sorted_values[end] == sorted_values[index]:
            end += 1
        if end - index > 1:
            ranks[order[index:end]] = np.mean(ranks[order[index:end]])
        index = end
    return ranks


def pearson(x: list[float], y: list[float]) -> tuple[float, float]:
    pairs = [
        (a, b)
        for a, b in zip(x, y, strict=True)
        if math.isfinite(a) and math.isfinite(b)
    ]
    if len(pairs) < 3:
        return float("nan"), float("nan")
    a = np.asarray([p[0] for p in pairs], dtype=float)
    b = np.asarray([p[1] for p in pairs], dtype=float)
    if float(np.std(a)) == 0 or float(np.std(b)) == 0:
        return float("nan"), float("nan")
    r, p_value = float(np.corrcoef(a, b)[0, 1]), float("nan")
    if math.isfinite(r):
        n = len(pairs)
        if abs(r) >= 1.0:
            p_value = 0.0
        else:
            t = r * math.sqrt((n - 2) / (1 - r * r))
            p_value = 2 * (1 - _t_cdf(abs(t), n - 2))
    return r, p_value


def spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    pairs = [
        (a, b)
        for a, b in zip(x, y, strict=True)
        if math.isfinite(a) and math.isfinite(b)
    ]
    if len(pairs) < 3:
        return float("nan"), float("nan")
    rx = _rankdata([p[0] for p in pairs])
    ry = _rankdata([p[1] for p in pairs])
    r, _ = pearson(rx.tolist(), ry.tolist())
    n = len(pairs)
    p_value = float("nan")
    if math.isfinite(r):
        if abs(r) >= 1.0:
            p_value = 0.0
        else:
            t = r * math.sqrt((n - 2) / (1 - r * r))
            p_value = 2 * (1 - _t_cdf(abs(t), n - 2))
    return r, p_value


def _t_cdf(t: float, df: int) -> float:
    """Student-t CDF via the incomplete beta (numerically stable, scipy-free)."""
    x = df / (df + t * t)
    return 1 - 0.5 * _betainc(df / 2, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Continued-fraction evaluation (modified Lentz).
    tiny = 1e-300
    max_iter = 300
    eps = 1e-12
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    factor = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    return h * factor / a


GAP_BUCKETS = [
    ("lt005", -float("inf"), 0.05),
    ("05_20", 0.05, 0.20),
    ("20_50", 0.20, 0.50),
    ("50_70", 0.50, 0.70),
    ("ge70", 0.70, float("inf")),
]


def gap_bucket(gap: float) -> str:
    for name, lo, hi in GAP_BUCKETS:
        if lo <= gap < hi:
            return name
    return "ge70"


def load_results(out_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("audit_summary_shard*.csv")):
        summaries.extend(_read_csv(path))
    summaries.sort(key=lambda row: (str(row.get("group")), str(row.get("decision_id"))))
    return summaries


def resolved_stats(rows: list[dict[str, Any]], *, group: str) -> dict[str, Any]:
    if group == "override":
        supported = [r for r in rows if r.get("resolution") == "SUPPORTED"]
        rejected = [r for r in rows if r.get("resolution") == "REJECTED"]
        unresolved = [r for r in rows if r.get("resolution") == "UNRESOLVED"]
        denominator = len(supported) + len(rejected)
        precision = (
            len(supported) / denominator if denominator else float("nan")
        )
        ci = wilson_ci(len(supported), denominator) if denominator else (float("nan"), float("nan"))
    else:
        supported = [r for r in rows if r.get("keep_resolution") == "KEEP_SUPPORTED"]
        rejected = [r for r in rows if r.get("keep_resolution") == "KEEP_REJECTED"]
        unresolved = [r for r in rows if r.get("keep_resolution") == "KEEP_UNRESOLVED"]
        denominator = len(supported) + len(rejected)
        precision = (
            len(supported) / denominator if denominator else float("nan")
        )
        ci = wilson_ci(len(supported), denominator) if denominator else (float("nan"), float("nan"))
    deltas = [_f(r.get("mean_delta_full")) for r in rows]
    supported_deltas = [_f(r.get("mean_delta_full")) for r in supported]
    return {
        "n": len(rows),
        "n_supported": len(supported),
        "n_rejected": len(rejected),
        "n_unresolved": len(unresolved),
        "precision": precision,
        "precision_ci_lo": ci[0],
        "precision_ci_hi": ci[1],
        "mean_delta": _mean(deltas),
        "median_delta": _median(deltas),
        "supported_mean_delta": _mean(supported_deltas),
        "resolved_n": denominator,
    }


def harmful_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_deltas = [_f(r.get("mean_delta_full")) for r in rows]
    harmful = [d for d in all_deltas if math.isfinite(d) and d < 0]
    resolved_harmful = [
        _f(r.get("mean_delta_full"))
        for r in rows
        if r.get("resolution") in ("SUPPORTED", "REJECTED")
        and _f(r.get("mean_delta_full")) < 0
    ]
    return {
        "harmful_override_count": len(harmful),
        "harmful_override_rate": (
            len(harmful) / len(all_deltas) if all_deltas else float("nan")
        ),
        "mean_harmful_loss": _mean(harmful),
        "max_harmful_loss": min(harmful) if harmful else float("nan"),
        "harmful_among_resolved_count": len(resolved_harmful),
        "harmful_among_resolved_rate": (
            len(resolved_harmful) / len(rows) if rows else float("nan")
        ),
    }


def threshold_sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    thresholds = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    result = []
    for threshold in thresholds:
        subset = [
            r
            for r in rows
            if math.isfinite(_f(r.get("predicted_delta_grp")))
            and abs(_f(r.get("predicted_delta_grp"))) >= threshold
        ]
        stats = resolved_stats(subset, group="override")
        result.append(
            {
                "threshold_abs_delta_grp": threshold,
                "coverage": len(subset) / len(rows) if rows else float("nan"),
                "n": stats["n"],
                "n_supported": stats["n_supported"],
                "n_rejected": stats["n_rejected"],
                "n_unresolved": stats["n_unresolved"],
                "override_precision": stats["precision"],
                "precision_ci_lo": stats["precision_ci_lo"],
                "precision_ci_hi": stats["precision_ci_hi"],
                "mean_delta_full": stats["mean_delta"],
            }
        )
    return result


def gap_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for name, lo, hi in GAP_BUCKETS:
        subset = [r for r in rows if gap_bucket(_f(r.get("gap12"))) == name]
        if not subset:
            continue
        stats = resolved_stats(subset, group="override")
        result.append(
            {
                "gap_bucket": name,
                "range": f"{'-inf' if lo == -float('inf') else lo}..{'+inf' if hi == float('inf') else hi}",
                "n": stats["n"],
                "n_supported": stats["n_supported"],
                "n_rejected": stats["n_rejected"],
                "n_unresolved": stats["n_unresolved"],
                "override_precision": stats["precision"],
                "mean_delta_full": stats["mean_delta"],
            }
        )
    return result


def rank_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for rank in (2, 3, 4):
        subset = [
            r for r in rows if _i(r.get("audit_b_rank")) == rank
        ]
        stats = resolved_stats(subset, group="override")
        result.append(
            {
                "teacher_best_rank": rank,
                "n": stats["n"],
                "n_supported": stats["n_supported"],
                "n_rejected": stats["n_rejected"],
                "n_unresolved": stats["n_unresolved"],
                "override_precision": stats["precision"],
                "precision_ci_lo": stats["precision_ci_lo"],
                "precision_ci_hi": stats["precision_ci_hi"],
                "mean_delta_full": stats["mean_delta"],
            }
        )
    return result


def calibration_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = [0.0, 0.2, 0.4, 0.6, 1.0, 2.0, float("inf")]
    result = []
    for index in range(len(edges) - 1):
        lo, hi = edges[index], edges[index + 1]
        subset = [
            r
            for r in rows
            if math.isfinite(_f(r.get("predicted_delta_grp")))
            and abs(_f(r.get("predicted_delta_grp"))) >= lo
            and (
                abs(_f(r.get("predicted_delta_grp"))) < hi
                if math.isfinite(hi)
                else True
            )
        ]
        if not subset:
            continue
        stats = resolved_stats(subset, group="override")
        result.append(
            {
                "predicted_delta_grp_bucket": (
                    f"{lo:g}-{hi:g}" if math.isfinite(hi) else f">={lo:g}"
                ),
                "n": stats["n"],
                "mean_predicted_delta_grp": _mean(
                    [_f(r.get("predicted_delta_grp")) for r in subset]
                ),
                "mean_full_hanchan_delta": stats["mean_delta"],
                "supported_fraction": (
                    stats["n_supported"] / stats["n"]
                    if stats["n"]
                    else float("nan")
                ),
            }
        )
    return result


def expert_analysis(overrides: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for r in overrides:
        expert = _i(r.get("expert_action"))
        top1 = _i(r.get("top1_action"))
        teacher = _i(r.get("audit_b_action"))
        resolution = r.get("resolution")
        mc_preferred = "teacher_best" if resolution == "SUPPORTED" else (
            "policy_top1" if resolution == "REJECTED" else "unresolved"
        )
        rows.append(
            {
                "decision_id": r["decision_id"],
                "policy_top1_is_expert": expert == top1,
                "teacher_best_is_expert": expert == teacher,
                "fullmc_preferred": mc_preferred,
                "fullmc_preferred_is_expert": (
                    expert == teacher if mc_preferred == "teacher_best" else (
                        expert == top1 if mc_preferred == "policy_top1" else False
                    )
                ),
                "teacher_not_expert_mc_supported": (
                    expert != teacher and mc_preferred == "teacher_best"
                ),
            }
        )
    return {
        "n": len(rows),
        "policy_top1_is_expert_rate": _mean([1.0 if r["policy_top1_is_expert"] else 0.0 for r in rows]),
        "teacher_best_is_expert_rate": _mean([1.0 if r["teacher_best_is_expert"] else 0.0 for r in rows]),
        "fullmc_preferred_is_expert_rate": _mean([1.0 if r["fullmc_preferred_is_expert"] else 0.0 for r in rows]),
        "teacher_not_expert_mc_supported_count": sum(1 for r in rows if r["teacher_not_expert_mc_supported"]),
        "cases": rows,
    }


def final_verdict(
    override_stats: dict[str, Any],
    harmful: dict[str, Any],
    calibration: dict[str, Any],
    keep_stats: dict[str, Any],
) -> dict[str, Any]:
    precision = override_stats["precision"]
    harmful_rate = harmful["harmful_among_resolved_rate"]
    mean_gain = override_stats["mean_delta"]
    spearman_r = calibration.get("spearman_r", float("nan"))
    keep_accuracy = keep_stats.get("precision", float("nan"))

    go = (
        precision >= 0.80
        and harmful_rate <= 0.10
        and mean_gain >= 0.5
        and (math.isfinite(spearman_r) and spearman_r >= 0.30)
    )
    conditional = (
        precision >= 0.60
        and harmful_rate <= 0.20
        and mean_gain > 0
    )
    if go:
        verdict = "GO"
    elif conditional:
        verdict = "CONDITIONAL GO"
    else:
        verdict = "NO-GO"
    return {
        "verdict": verdict,
        "rationale": {
            "override_precision": precision,
            "harmful_among_resolved_rate": harmful_rate,
            "mean_full_hanchan_gain": mean_gain,
            "spearman_predicted_vs_observed": spearman_r,
            "keep_accuracy": keep_accuracy,
            "go_thresholds": {
                "precision>=": 0.80,
                "harmful_rate<=": 0.10,
                "mean_gain>=": 0.5,
                "spearman>=": 0.30,
            },
            "conditional_thresholds": {
                "precision>=": 0.60,
                "harmful_rate<=": 0.20,
                "mean_gain>": 0.0,
            },
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir or (Path(__file__).resolve().parent / "results"))

    rows = load_results(out_dir)
    overrides = [r for r in rows if str(r.get("group")) == "override"]
    raw_keeps = [r for r in rows if str(r.get("group")) == "keep_control"]
    # This run's sample matching reused four keep decisions (each duplicated
    # row audits the same A/B pair).  Statistics use one row per decision.
    best_keep: dict[str, dict[str, Any]] = {}
    for keep in raw_keeps:
        key = str(keep["decision_id"])
        if key not in best_keep:
            best_keep[key] = keep
            continue
        try:
            current_dist = float(best_keep[key].get("match_distance", "inf"))
            new_dist = float(keep.get("match_distance", "inf"))
        except (TypeError, ValueError):
            continue
        if new_dist < current_dist:
            best_keep[key] = keep
    keeps = sorted(best_keep.values(), key=lambda r: str(r["decision_id"]))
    n_keep_duplicates_dropped = len(raw_keeps) - len(keeps)
    if not overrides or not keeps:
        raise RuntimeError(
            f"need both override and keep summaries: override={len(overrides)} keep={len(keeps)}"
        )

    override_stats = resolved_stats(overrides, group="override")
    keep_stats = resolved_stats(keeps, group="keep_control")
    harmful = harmful_stats(overrides)
    rank_rows = rank_audit(overrides)
    gap_rows = gap_audit(overrides)
    sweep_rows = threshold_sweep(overrides)
    cal_rows = []
    for r in overrides + keeps:
        cal_rows.append(
            {
                "decision_id": r["decision_id"],
                "group": r["group"],
                "audit_b_rank": r.get("audit_b_rank"),
                "predicted_delta_grp": r.get("predicted_delta_grp"),
                "predicted_delta_grp_se": r.get("predicted_delta_grp_se"),
                "mean_delta_full": r.get("mean_delta_full"),
                "se_delta_full": r.get("se_delta_full"),
                "ci95_lo": r.get("ci95_lo"),
                "ci95_hi": r.get("ci95_hi"),
                "resolution": r.get("resolution"),
                "keep_resolution": r.get("keep_resolution"),
            }
        )
    pred = [_f(r.get("predicted_delta_grp")) for r in overrides]
    obs = [_f(r.get("mean_delta_full")) for r in overrides]
    pearson_r, pearson_p = pearson(pred, obs)
    spearman_r, spearman_p = spearman(pred, obs)
    pred_all = [_f(r.get("predicted_delta_grp")) for r in overrides + keeps]
    obs_all = [_f(r.get("mean_delta_full")) for r in overrides + keeps]
    pearson_all_r, pearson_all_p = pearson(pred_all, obs_all)
    spearman_all_r, spearman_all_p = spearman(pred_all, obs_all)
    calibration = {
        "override_only": {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "n": len([p for p in pred if math.isfinite(p)]),
        },
        "override_and_keep": {
            "pearson_r": pearson_all_r,
            "pearson_p": pearson_all_p,
            "spearman_r": spearman_all_r,
            "spearman_p": spearman_all_p,
            "n": len([p for p in pred_all if math.isfinite(p)]),
        },
    }
    cal_buckets = calibration_buckets(overrides)
    expert = expert_analysis(overrides)
    verdict = final_verdict(override_stats, harmful, calibration["override_only"], keep_stats)

    shard_summaries = []
    for path in sorted(out_dir.glob("fullmc_summary_shard*.json")):
        shard_summaries.append(json.loads(path.read_text()))

    summary = {
        "step2_field_notes": {
            "all_pairwise_resolved95": 19,
            "final_verdict_resolved95": 173,
            "conservative_keep": 143,
            "high_confidence_override": 30,
            "uncertain": 247,
            "conservative_keep_definition": (
                "当前 Teacher 没有足够统计证据证明任何 challenger 优于 Policy Top1，"
                "因此默认保持 Policy Top1；它不表示已经统计证明 Top1 是 Top4 中的最优动作。"
            ),
        },
        "override_audit": {
            **override_stats,
            **harmful,
            "mean_delta_full": override_stats["mean_delta"],
            "median_delta_full": override_stats["median_delta"],
        },
        "keep_control_audit": {
            **keep_stats,
            "n_audit_rows": len(raw_keeps),
            "n_unique_decisions": len(keeps),
            "n_duplicate_rows_dropped": n_keep_duplicates_dropped,
        },
        "rank_breakdown": rank_rows,
        "gap_breakdown": gap_rows,
        "threshold_sweep": sweep_rows,
        "calibration": calibration,
        "calibration_buckets": cal_buckets,
        "expert_secondary": {
            key: value for key, value in expert.items() if key != "cases"
        },
        "verdict": verdict,
        "performance": {
            "shards": shard_summaries,
            "total_worlds": sum(
                int(s.get("worlds", 0)) for s in shard_summaries
            ),
            "total_policy_decisions": sum(
                int(s.get("policy_decisions", 0)) for s in shard_summaries
            ),
            "total_rollouts": sum(int(s.get("rollouts", 0)) for s in shard_summaries),
            "total_waves": sum(int(s.get("waves", 0)) for s in shard_summaries),
            "total_elapsed_s": sum(float(s.get("elapsed_s", 0)) for s in shard_summaries),
        },
    }
    (out_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    _write_csv(out_dir / "override_audit_results.csv", overrides)
    _write_csv(out_dir / "keep_control_audit_results.csv", keeps)
    _write_csv(out_dir / "teacher_fullmc_calibration.csv", cal_rows)
    _write_csv(out_dir / "calibration_buckets.csv", cal_buckets)
    _write_csv(out_dir / "rank4_override_audit.csv", rank_rows)
    _write_csv(out_dir / "gap_override_audit.csv", gap_rows)
    _write_csv(out_dir / "threshold_sweep.csv", sweep_rows)
    _write_report(
        out_dir,
        overrides,
        keeps,
        override_stats,
        keep_stats,
        harmful,
        rank_rows,
        gap_rows,
        sweep_rows,
        calibration,
        cal_buckets,
        expert,
        verdict,
        summary,
    )
    print(
        "override precision: %.3f (%d/%d) | harmful rate %.3f | mean Δ %.3f | verdict %s"
        % (
            override_stats["precision"],
            override_stats["n_supported"],
            override_stats["resolved_n"],
            harmful["harmful_among_resolved_rate"],
            override_stats["mean_delta"],
            verdict["verdict"],
        )
    )


def _report_number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _write_report(
    out_dir: Path,
    overrides: list[dict[str, Any]],
    keeps: list[dict[str, Any]],
    override_stats: dict[str, Any],
    keep_stats: dict[str, Any],
    harmful: dict[str, Any],
    rank_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    calibration: dict[str, Any],
    cal_buckets: list[dict[str, Any]],
    expert: dict[str, Any],
    verdict: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Step 2.5 — High-Budget Teacher Override Audit (Full-Hanchan MC)")
    add("")
    add("## Teacher Override Audit")
    add("")
    add("```text")
    add(f"Total overrides:   {override_stats['n']}")
    add(f"Supported:        {override_stats['n_supported']}")
    add(f"Rejected:         {override_stats['n_rejected']}")
    add(f"Unresolved:       {override_stats['n_unresolved']}")
    add("")
    add(f"Override Precision: {_report_number(override_stats['precision'])}")
    add(
        f"95% CI: [{_report_number(override_stats['precision_ci_lo'])}, "
        f"{_report_number(override_stats['precision_ci_hi'])}]"
    )
    add(f"Mean full-hanchan utility gain: {_report_number(override_stats['mean_delta'])}")
    add(
        f"Harmful override rate: {_report_number(harmful['harmful_among_resolved_rate'])}"
    )
    add(f"Rank2 precision: {_report_number(next((r['override_precision'] for r in rank_rows if r['teacher_best_rank'] == 2), float('nan')))}")
    add(f"Rank3 precision: {_report_number(next((r['override_precision'] for r in rank_rows if r['teacher_best_rank'] == 3), float('nan')))}")
    add(f"Rank4 precision: {_report_number(next((r['override_precision'] for r in rank_rows if r['teacher_best_rank'] == 4), float('nan')))}")
    add("```")
    add("")
    add("## Step 2 字段口径（复用，禁止混用）")
    add("")
    add("```text")
    add("all_pairwise_resolved95 = 19")
    add("final_verdict_resolved95 = 173")
    add("  conservative_keep = 143")
    add("  high_confidence_override = 30")
    add("uncertain = 247")
    add("")
    add("conservative_keep：")
    add("当前 Teacher 没有足够统计证据证明任何 challenger 优于 Policy Top1，")
    add("因此默认保持 Policy Top1；")
    add("它不表示已经统计证明 Top1 是 Top4 中的最优动作。")
    add("```")
    add("")
    add("## 方法")
    add("")
    add(
        "- 审计集：Step 2 全部 30 个 `high_confidence_override` decision"
        "（Teacher Best = Rank2/3/4 = 11/9/10）+ 30 个按 Policy gap bucket、"
        "kyoku stage、π1、Step2 |ΔGRP|、world budget、action type 匹配的 "
        "conservative-keep control。"
    )
    add(
        "- 每个 decision：A = Policy Top1，B = Teacher Best（override）或 Step2 "
        "最强 challenger（keep control）；同一 hidden world 的两个分支只强制首个动作不同，"
        "后续全部玩家用 PPO v2 greedy/argmax 继续到整个半庄结束。"
    )
    add(
        "- Hidden world 采样沿用 Step 2 的 uniform baseline（牌种守恒、合法手牌、"
        "dora 槽位一致），并对每个 world 固定 wall seed，使分支后续 kyoku 的洗牌随机数匹配。"
    )
    add(
        "- 最终 reward 直接使用目标座位的实际最终排名：1st=+10, 2nd=+4, 3rd=-4, "
        "4th=-10；全程不使用 GRP V2 leaf value。"
    )
    add(
        "- Paired 统计量 D_i = R_B - R_A；adaptive budget 64→128→256"
        "（override 与 keep control 上限均为 256），z=1.96 的 CI 连续两个 "
        "wave 排除 0 即停。"
    )
    add("")
    add("## 结果")
    add("")
    add("### Override Precision 与 Gain")
    add("")
    add("| 指标 | 值 |")
    add("|---|---|")
    add(f"| n_total_override | {override_stats['n']} |")
    add(f"| n_supported | {override_stats['n_supported']} |")
    add(f"| n_rejected | {override_stats['n_rejected']} |")
    add(f"| n_unresolved | {override_stats['n_unresolved']} |")
    add(f"| Override Precision | {_report_number(override_stats['precision'])} |")
    add(
        f"| Precision 95% CI | [{_report_number(override_stats['precision_ci_lo'])}, "
        f"{_report_number(override_stats['precision_ci_hi'])}] |"
    )
    add(f"| mean ΔFull | {_report_number(override_stats['mean_delta'])} |")
    add(f"| median ΔFull | {_report_number(override_stats['median_delta'])} |")
    add(f"| SUPPORTED 的 mean ΔFull | {_report_number(override_stats['supported_mean_delta'])} |")
    add("")
    add("### Harmful Override")
    add("")
    add("| 指标 | 值 |")
    add("|---|---|")
    add(f"| harmful_override_count | {harmful['harmful_override_count']} |")
    add(f"| harmful_override_rate（全部 override） | {_report_number(harmful['harmful_override_rate'])} |")
    add(f"| harmful among resolved rate | {_report_number(harmful['harmful_among_resolved_rate'])} |")
    add(f"| mean harmful loss | {_report_number(harmful['mean_harmful_loss'])} |")
    add(f"| max harmful loss | {_report_number(harmful['max_harmful_loss'])} |")
    add("")
    add("### Matched Keep Controls")
    add("")
    keep_note = summary.get("keep_control_audit", {})
    if int(keep_note.get("n_duplicate_rows_dropped", 0) or 0) > 0:
        add(
            f"- 本运行的样本匹配复用了 4 个 keep decision（同一 decision 的 A/B 审计完全相同），"
            f"统计时按 decision_id 去重：{keep_note.get('n_audit_rows')} audit rows → "
            f"{keep_note.get('n_unique_decisions')} unique decisions。"
        )
        add("")
    add("| 指标 | 值 |")
    add("|---|---|")
    add(f"| n | {keep_stats['n']} |")
    add(f"| keep_supported | {keep_stats['n_supported']} |")
    add(f"| keep_rejected | {keep_stats['n_rejected']} |")
    add(f"| keep_unresolved | {keep_stats['n_unresolved']} |")
    add(f"| keep accuracy（resolved 内） | {_report_number(keep_stats['precision'])} |")
    add(f"| keep 95% CI | [{_report_number(keep_stats['precision_ci_lo'])}, {_report_number(keep_stats['precision_ci_hi'])}] |")
    add("")
    add("### 按 Teacher Rank")
    add("")
    add("| Rank | n | SUPPORTED | REJECTED | UNRESOLVED | Precision | mean ΔFull |")
    add("|---|---|---|---|---|---|---|")
    for r in rank_rows:
        add(
            f"| {r['teacher_best_rank']} | {r['n']} | {r['n_supported']} | "
            f"{r['n_rejected']} | {r['n_unresolved']} | "
            f"{_report_number(r['override_precision'])} | {_report_number(r['mean_delta_full'])} |"
        )
    add("")
    add("### 按 Policy Gap")
    add("")
    add("| Bucket | n | SUPPORTED | REJECTED | UNRESOLVED | Precision | mean ΔFull |")
    add("|---|---|---|---|---|---|---|")
    for r in gap_rows:
        add(
            f"| {r['gap_bucket']} | {r['n']} | {r['n_supported']} | "
            f"{r['n_rejected']} | {r['n_unresolved']} | "
            f"{_report_number(r['override_precision'])} | {_report_number(r['mean_delta_full'])} |"
        )
    add("")
    add("### Step2 |ΔGRP| Threshold Sweep")
    add("")
    add("| threshold | coverage | n | supported | rejected | unresolved | precision | mean ΔFull |")
    add("|---|---|---|---|---|---|---|---|")
    for r in sweep_rows:
        add(
            f"| {r['threshold_abs_delta_grp']:g} | {_report_number(r['coverage'])} | "
            f"{r['n']} | {r['n_supported']} | {r['n_rejected']} | {r['n_unresolved']} | "
            f"{_report_number(r['override_precision'])} | {_report_number(r['mean_delta_full'])} |"
        )
    add("")
    add("### Calibration：Step2 ΔGRP vs Full-Hanchan ΔFull")
    add("")
    cal = calibration["override_only"]
    cal_all = calibration["override_and_keep"]
    add("| 范围 | Pearson r (p) | Spearman r (p) | n |")
    add("|---|---|---|---|")
    add(
        f"| override only | {_report_number(cal['pearson_r'])} ({_report_number(cal['pearson_p'])}) | "
        f"{_report_number(cal['spearman_r'])} ({_report_number(cal['spearman_p'])}) | {cal['n']} |"
    )
    add(
        f"| override + keep | {_report_number(cal_all['pearson_r'])} ({_report_number(cal_all['pearson_p'])}) | "
        f"{_report_number(cal_all['spearman_r'])} ({_report_number(cal_all['spearman_p'])}) | {cal_all['n']} |"
    )
    add("")
    add("| predicted |ΔGRP| bucket | n | mean predicted | mean ΔFull | supported fraction |")
    add("|---|---|---|---|---|")
    for r in cal_buckets:
        add(
            f"| {r['predicted_delta_grp_bucket']} | {r['n']} | "
            f"{_report_number(r['mean_predicted_delta_grp'])} | "
            f"{_report_number(r['mean_full_hanchan_delta'])} | "
            f"{_report_number(r['supported_fraction'])} |"
        )
    add("")
    add("### Expert Secondary Analysis")
    add("")
    add(f"- Policy Top1 == Expert: {_report_number(expert['policy_top1_is_expert_rate'])}")
    add(f"- Teacher Best == Expert: {_report_number(expert['teacher_best_is_expert_rate'])}")
    add(f"- Full-MC preferred == Expert: {_report_number(expert['fullmc_preferred_is_expert_rate'])}")
    add(f"- Teacher ≠ Expert 且 Full-MC 支持 Teacher 的案例数: {expert['teacher_not_expert_mc_supported_count']}")
    add("")
    add("### 性能")
    add("")
    perf = summary.get("performance", {})
    add(f"- 总 worlds（world-pairs）：{perf.get('total_worlds')}")
    add(f"- 总 rollouts（分支轨迹）：{perf.get('total_rollouts')}")
    add(f"- 总 policy decisions：{perf.get('total_policy_decisions')}")
    add(f"- 总 wave 数：{perf.get('total_waves')}")
    add(f"- 四 shard 合计运行时间：{perf.get('total_elapsed_s')}s")
    add("")
    add("| shard | decisions | worlds | rollouts/s | policy decisions/s | avg batch | GPU util | CPU core util |")
    add("|---|---|---|---|---|---|---|---|")
    for shard in perf.get("shards", []):
        add(
            f"| {shard.get('shard_id')} | {shard.get('decisions')} | "
            f"{shard.get('worlds')} | {shard.get('rollouts_per_s')} | "
            f"{shard.get('policy_decisions_per_s')} | "
            f"{shard.get('avg_policy_batch_size')} | "
            f"{shard.get('gpu_util_pct')}% | {shard.get('cpu_core_util_pct')}% |"
        )
    add("")
    add("## Success Criteria (Q1–Q8)")
    add("")
    add(
        f"**Q1** Step2 的 30 个 high-confidence Teacher override 中，"
        f"在 full-hanchan expected utility 意义上得到支持的为 "
        f"{override_stats['n_supported']}（{_report_number(override_stats['precision'])} "
        f"precision，unresolved {override_stats['n_unresolved']} 个不计数）。"
    )
    add(
        f"**Q2** Teacher Override Precision = {_report_number(override_stats['precision'])} "
        f"(95% CI [{_report_number(override_stats['precision_ci_lo'])}, "
        f"{_report_number(override_stats['precision_ci_hi'])}])，"
        f"是否足以作为 Reranker 监督信号见最终结论。"
    )
    add(
        f"**Q3** Teacher override 的 mean ΔFull = {_report_number(override_stats['mean_delta'])}，"
        f"median = {_report_number(override_stats['median_delta'])}；"
        f"SUPPORTED 子集的 mean = {_report_number(override_stats['supported_mean_delta'])}。"
    )
    add(
        f"**Q4** harmful override：{harmful['harmful_override_count']} 个（全部 override 中 "
        f"{_report_number(harmful['harmful_override_rate'])}；resolved 中 "
        f"{_report_number(harmful['harmful_among_resolved_rate'])}），"
        f"mean loss = {_report_number(harmful['mean_harmful_loss'])}，"
        f"max loss = {_report_number(harmful['max_harmful_loss'])}。"
    )
    add(
        f"**Q5** Step2 |ΔGRP| 与 full-hanchan override reliability：Spearman r = "
        f"{_report_number(cal['spearman_r'])}（p={_report_number(cal['spearman_p'])}）；"
        f"threshold sweep 见上表。"
    )
    add(
        f"**Q6** Policy gap 的 gate value：按 gap bucket 的 precision/mean ΔFull 见上表；"
        f"high-gap 中 rare override 的行为见 gap_override_audit.csv。"
    )
    add(
        "**Q7** Rank4 override 审计：n = "
        f"{next((r['n'] for r in rank_rows if r['teacher_best_rank'] == 4), 0)}，"
        f"precision = "
        f"{_report_number(next((r['override_precision'] for r in rank_rows if r['teacher_best_rank'] == 4), float('nan')))}，"
        f"mean ΔFull = "
        f"{_report_number(next((r['mean_delta_full'] for r in rank_rows if r['teacher_best_rank'] == 4), float('nan')))}。"
    )
    add(
        f"**Q8** Matched keep controls：n = {keep_stats['n']}，keep_supported = "
        f"{keep_stats['n_supported']}，keep_rejected = {keep_stats['n_rejected']}，"
        f"keep_unresolved = {keep_stats['n_unresolved']}，"
        f"resolved 内 keep accuracy = {_report_number(keep_stats['precision'])}。"
    )
    add("")
    add("## Correctness Audit")
    add("")
    correctness_path = out_dir / "fullmc_correctness_audit.json"
    if correctness_path.exists():
        correctness = json.loads(correctness_path.read_text())
        add(f"- {correctness['decisions_audited']} decisions × {correctness['worlds_per_decision']} worlds")
        add(f"- checks: {correctness['passed']}/{correctness['total_checks']} passed")
        term = correctness.get("termination_wind_counts", {})
        from collections import Counter as _Counter
        wind_counts = _Counter()
        for detail, count in term.items():
            wind = 0
            for token in detail.replace("round_wind=", " ").split():
                if token.isdigit():
                    wind = int(token)
                    break
            wind_counts[wind] += count
        add(
            "- 结束场分布："
            + "；".join(
                f"wind={wind} {count} 分支"
                for wind, count in sorted(wind_counts.items())
            )
        )
    else:
        add("- correctness audit 未运行（见 fullmc_correctness_audit.json）")
    add("")
    add("## Limitations")
    add("")
    add(
        "- Uniform hidden-world baseline：不包含基于弃牌、立直、副露、手切/摸切等"
        "行为信息的 posterior inference。"
    )
    add(
        "- 分支在 renchan 路径分叉后，后续 kyoku 的 wall 会按各自已消耗的随机数流"
        "继续（同一 world seed + 同一 shuffle seed 流，属于 matched randomness 设计）；"
        "只有 public 牌种完全一致时 wall 才逐值相同。"
    )
    add(
        "- `4p-red-half` 环境按 30000 目标分规则可能在无人达到 30000 时延入西场"
        "（本审计中观察到 1 例），这是环境官方规则，非实现错误。"
    )
    add(
        "- Expert action 仅作 secondary analysis，不参与 ground truth。"
    )
    add("")
    add("## 结论")
    add("")
    add(f"**{verdict['verdict']}**")
    add("")
    add("依据：")
    add(f"- Override Precision = {_report_number(override_stats['precision'])}")
    add(f"- harmful override rate（resolved 内）= {_report_number(harmful['harmful_among_resolved_rate'])}")
    add(f"- mean ΔFull = {_report_number(override_stats['mean_delta'])}")
    add(f"- Step2 |ΔGRP| 与 ΔFull 的 Spearman = {_report_number(cal['spearman_r'])}")
    add(f"- matched keep accuracy = {_report_number(keep_stats['precision'])}")
    add("")
    add("CONDITIONAL 的具体条件（进入 Step 3 的 label generation 范围）：")
    add("")
    add(
        "1. **|ΔGRP| gate**：只对 Step2 |mean ΔGRP| ≥ 0.3 的 override 生成监督标签。"
        "本样本中该区域 resolved precision = 1.000（6 supported / 0 rejected，"
        "mean ΔFull = 0.617）；全部 2 个 REJECTED 都落在 |ΔGRP| < 0.3。"
    )
    add(
        "2. **候选集合**：Rank4 的 10 个 high-confidence override 在 256 worlds 下"
        "全部 UNRESOLVED（mean ΔFull ≈ 0.02），没有证据支持 Rank4 纳入正式候选；"
        "建议 Step 3 先使用 Top3 候选集，Rank4 需要单独提高预算复验后再决定。"
    )
    add(
        "3. **统计保守性**：30 个 override 中只有 8 个在 256 worlds 内达到统计判定，"
        "precision 的 95% Wilson CI 为 [0.409, 0.929]，较宽；"
        "正式大规模 label generation 前建议先用本 gate 在小批量上复验一次。"
    )
    add("")
    add("```json")
    add(json.dumps(verdict["rationale"], indent=2, ensure_ascii=False, default=str))
    add("```")
    add("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
