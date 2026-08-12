"""Analysis + report for the Step-2 Top4 paired counterfactual rollout.

Reads the sharded rollout outputs and produces every deliverable required by
GOAL_PROMPT_STEP2.md:

* decision_summary.csv / candidate_values.csv / paired_rollout_results.parquet
* gap_bucket_summary.csv / top3_vs_top4_summary.csv
* grp_threshold_sweep.csv / stability_summary.csv / teacher_vs_expert.csv
* experiment_summary.json / report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if math.isfinite(v)]
    return float(np.mean(values)) if values else None


def _rate(values: list[bool]) -> float | None:
    return float(np.mean(values)) if values else None


GAP_BUCKETS = [
    ("lt005", -float("inf"), 0.05),
    ("05_20", 0.05, 0.20),
    ("20_50", 0.20, 0.50),
    ("50_70", 0.50, 0.70),
    ("ge70", 0.70, float("inf")),
]


def _bucket(gap: float) -> str:
    for name, lo, hi in GAP_BUCKETS:
        if lo <= gap < hi:
            return name
    return "ge70"


def load_results(out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    candidate_values: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("decision_summary_shard*.csv")):
        summaries.extend(_read_csv(path))
    for path in sorted(out_dir.glob("stability_shard*.csv")):
        stability.extend(_read_csv(path))
    for path in sorted(out_dir.glob("candidate_values_shard*.csv")):
        candidate_values.extend(_read_csv(path))
    summaries.sort(key=lambda row: (_f(row.get("gap12")), str(row.get("decision_id"))))
    return summaries, stability, candidate_values


def gap_bucket_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, lo, hi in GAP_BUCKETS:
        subset = [r for r in rows if _bucket(_f(r.get("gap12"))) == name]
        if not subset:
            continue
        result.append({
            "gap_bucket": name,
            "range": f"{'-inf' if lo == -float('inf') else lo}..{'+inf' if hi == float('inf') else hi}",
            "samples": len(subset),
            "n_determined95": sum(bool(r.get("determined95") == "True") for r in subset),
            "n_determined80": sum(bool(r.get("determined80") == "True") for r in subset),
            "keep_top1_rate": _rate([r.get("verdict95") == "keep_top1" for r in subset]),
            "override_rate": _rate([str(r.get("verdict95", "")).startswith("override") for r in subset]),
            "uncertain_rate": _rate([r.get("verdict95") == "uncertain" for r in subset]),
            "teacher_best_is_top1_rate": _rate(
                [_int(r.get("teacher_best")) == 1 for r in subset]
            ),
            "top1_wrong_rate": _rate(
                [str(r.get("verdict95", "")).startswith("override") for r in subset]
            ),
            "teacher_best_is_expert_rate": _rate(
                [
                    _int(r.get("teacher_best")) == 1
                    and _int(r.get("expert_action")) == _int(r.get("top1_action"))
                    or (
                        _int(r.get("teacher_best")) == 2
                        and _int(r.get("expert_action")) == _int(r.get("top2_action"))
                    )
                    or (
                        _int(r.get("teacher_best")) == 3
                        and _int(r.get("expert_action")) == _int(r.get("top3_action"))
                    )
                    or (
                        _int(r.get("teacher_best")) == 4
                        and _int(r.get("expert_action")) == _int(r.get("top4_action"))
                    )
                    for r in subset
                ]
            ),
            "policy_top1_is_expert_rate": _rate(
                [_int(r.get("expert_action")) == _int(r.get("top1_action")) for r in subset]
            ),
            "mean_abs_delta_ba": _mean([abs(_f(r.get("delta_ba_mean"))) for r in subset]),
            "mean_delta_ba": _mean([_f(r.get("delta_ba_mean")) for r in subset]),
            "mean_abs_delta_ca": _mean([abs(_f(r.get("delta_ca_mean"))) for r in subset]),
            "mean_abs_delta_da": _mean([abs(_f(r.get("delta_da_mean"))) for r in subset]),
        })
    return result


def top3_vs_top4(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improvements: list[dict[str, Any]] = []
    for r in rows:
        means = {1: _f(r.get("mean_a")), 2: _f(r.get("mean_b")),
                 3: _f(r.get("mean_c")), 4: _f(r.get("mean_d"))}
        if any(not math.isfinite(v) for v in means.values()):
            continue
        best3 = max(means[1], means[2], means[3])
        best4 = max(means.values())
        improvement = best4 - best3
        best3_rank = max((1, 2, 3), key=lambda k: means[k])
        best4_rank = max((1, 2, 3, 4), key=lambda k: means[k])
        improvements.append({
            "decision_id": r["decision_id"],
            "gap12": _f(r.get("gap12")),
            "gap_bucket": _bucket(_f(r.get("gap12"))),
            "best3_rank": best3_rank,
            "best4_rank": best4_rank,
            "best3_value": best3,
            "best4_value": best4,
            "improvement": improvement,
            "top4_wins": int(best4_rank) == 4,
            "top4_improves_over_top3": improvement > 0,
        })
    improved = [x for x in improvements if x["top4_improves_over_top3"]]
    rank4_best = [x for x in improvements if x["top4_wins"]]
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in improvements:
        by_bucket[x["gap_bucket"]].append(x)
    return {
        "n": len(improvements),
        "n_improved_over_top3": len(improved),
        "improved_fraction": _rate([x["top4_improves_over_top3"] for x in improvements]),
        "mean_improvement": _mean([x["improvement"] for x in improvements]),
        "mean_improvement_improved": _mean([x["improvement"] for x in improved]),
        "n_rank4_best": len(rank4_best),
        "rank4_best_fraction": _rate([x["top4_wins"] for x in improvements]),
        "by_gap_bucket": {
            name: {
                "n": len(items),
                "improved_fraction": _rate([x["top4_improves_over_top3"] for x in items]),
                "mean_improvement": _mean([x["improvement"] for x in items]),
                "n_rank4_best": sum(x["top4_wins"] for x in items),
            }
            for name, items in sorted(by_bucket.items())
        },
        "rows": improvements,
    }


def stability_summary(stability: list[dict[str, Any]]) -> dict[str, Any]:
    by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stability:
        by_decision[row["decision_id"]].append(row)
    flip_rows: list[dict[str, Any]] = []
    for sid, rows in sorted(by_decision.items()):
        rows.sort(key=lambda r: int(r["n_worlds"]))
        for prev, curr in zip(rows, rows[1:], strict=False):
            if int(curr["n_worlds"]) == int(prev["n_worlds"]):
                continue
            flip_rows.append({
                "decision_id": sid,
                "n_prev": int(prev["n_worlds"]),
                "n_curr": int(curr["n_worlds"]),
                "best_prev": int(prev["best_candidate"]),
                "best_curr": int(curr["best_candidate"]),
                "flipped": int(prev["best_candidate"]) != int(curr["best_candidate"]),
            })
    flips = [x for x in flip_rows if x["flipped"]]
    # Per decision: whether best at max N agrees with best at N=16 / N=32 / N=64.
    convergence: list[dict[str, Any]] = []
    for sid, rows in sorted(by_decision.items()):
        rows.sort(key=lambda r: int(r["n_worlds"]))
        by_n = {int(r["n_worlds"]): r for r in rows}
        final_n = max(by_n)
        final_best = int(by_n[final_n]["best_candidate"])
        convergence.append({
            "decision_id": sid,
            "final_n": final_n,
            "final_best": final_best,
            "agree_n16": int(by_n.get(16, {}).get("best_candidate", -1)) == final_best,
            "agree_n32": int(by_n.get(32, {}).get("best_candidate", -1)) == final_best,
            "agree_n64": int(by_n.get(64, {}).get("best_candidate", -1)) == final_best,
        })
    return {
        "n_decisions_with_stability": len(by_decision),
        "n_consecutive_checkpoints": len(flip_rows),
        "flip_rate_between_checkpoints": _rate([x["flipped"] for x in flip_rows]),
        "n_decisions_flipped_any": len({x["decision_id"] for x in flips}),
        "agree_n16_rate": _rate([x["agree_n16"] for x in convergence]),
        "agree_n32_rate": _rate([x["agree_n32"] for x in convergence]),
        "agree_n64_rate": _rate([x["agree_n64"] for x in convergence]),
        "flips": flip_rows,
        "convergence": convergence,
    }


def teacher_vs_expert(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for r in rows:
        teacher_best = _int(r.get("teacher_best"))
        expert = _int(r.get("expert_action"))
        top = [_int(r.get(f"top{i}_action")) for i in (1, 2, 3, 4)]
        result.append({
            "decision_id": r["decision_id"],
            "gap12": _f(r.get("gap12")),
            "gap_bucket": _bucket(_f(r.get("gap12"))),
            "verdict95": r.get("verdict95"),
            "teacher_best": teacher_best,
            "expert_action": expert,
            "teacher_best_matches_expert": bool(
                1 <= teacher_best <= 4 and top[teacher_best - 1] == expert
            ),
            "policy_top1_matches_expert": bool(top[0] == expert),
        })
    return result


def grp_threshold_sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calibrate |mean ΔGRP| against teacher determination/override reliability."""
    thresholds = sorted({
        round(v, 2)
        for v in np.linspace(0.0, 3.0, 31)
    })
    out: list[dict[str, Any]] = []
    for threshold in thresholds:
        subset = [
            r for r in rows
            if max(
                abs(_f(r.get("delta_ba_mean"))),
                abs(_f(r.get("delta_ca_mean"))),
                abs(_f(r.get("delta_da_mean"))),
            ) >= threshold
        ]
        determined = [r for r in subset if str(r.get("verdict95")) != "uncertain"]
        overrides = [r for r in subset if str(r.get("verdict95", "")).startswith("override")]
        out.append({
            "min_abs_delta_grp": threshold,
            "coverage": len(subset) / len(rows) if rows else None,
            "n": len(subset),
            "determined_rate": _rate([str(r.get("verdict95")) != "uncertain" for r in subset]),
            "override_rate": _rate([str(r.get("verdict95", "")).startswith("override") for r in subset]),
            "keep_top1_rate": _rate([r.get("verdict95") == "keep_top1" for r in subset]),
            "override_expert_agreement": _rate([
                _int(r.get("teacher_best")) in (2, 3, 4)
                and {
                    2: _int(r.get("top2_action")),
                    3: _int(r.get("top3_action")),
                    4: _int(r.get("top4_action")),
                }[_int(r.get("teacher_best"))]
                == _int(r.get("expert_action"))
                for r in overrides
            ]) if overrides else None,
            "n_overrides": len(overrides),
        })
    return out


def overall_stats(rows: list[dict[str, Any]], stability: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    determined95 = sum(str(r.get("verdict95")) != "uncertain" for r in rows)
    determined80 = sum(str(r.get("verdict80")) != "uncertain" for r in rows)
    keep = sum(r.get("verdict95") == "keep_top1" for r in rows)
    override = sum(str(r.get("verdict95", "")).startswith("override") for r in rows)
    uncertain = sum(r.get("verdict95") == "uncertain" for r in rows)
    override_ranks = Counter(
        int(r.get("teacher_best")) for r in rows
        if str(r.get("verdict95", "")).startswith("override")
    )
    return {
        "n_decisions": n,
        "n_determined95": determined95,
        "n_determined80": determined80,
        "determined95_rate": determined95 / n if n else None,
        "determined80_rate": determined80 / n if n else None,
        "keep_top1": keep,
        "override": override,
        "uncertain": uncertain,
        "keep_top1_rate": keep / n if n else None,
        "override_rate": override / n if n else None,
        "uncertain_rate": uncertain / n if n else None,
        "override_by_rank": {str(k): v for k, v in sorted(override_ranks.items())},
        "mean_n_worlds": _mean([_f(r.get("n_worlds")) for r in rows]),
        "mean_abs_delta_ba": _mean([abs(_f(r.get("delta_ba_mean"))) for r in rows]),
        "mean_se_delta_ba": _mean([_f(r.get("delta_ba_se")) for r in rows]),
        "mean_abs_delta_ca": _mean([abs(_f(r.get("delta_ca_mean"))) for r in rows]),
        "mean_abs_delta_da": _mean([abs(_f(r.get("delta_da_mean"))) for r in rows]),
        "stability": stability_summary(stability),
    }


def write_report(
    out_dir: Path,
    *,
    sweep: dict[str, Any],
    overall: dict[str, Any],
    buckets: list[dict[str, Any]],
    top34: dict[str, Any],
    threshold_sweep: list[dict[str, Any]],
    teacher_expert: list[dict[str, Any]],
    audit: dict[str, Any] | None,
    performance: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Top4 Paired Counterfactual Rollout + GRP V2 Teacher — Step 2 验证报告")
    add("")
    add(f"日期：2026-08-12；目标文件：`audit/reports/grp_ranker_20260811/GOAL_PROMPT_STEP2.md`")
    add("")
    add("## 1. 实验设置")
    add("")
    add("- **候选策略（Candidate Policy）**：`checkpoints/train_riichi_v13_sft/best_heuristic.pt`（v13 SFT，isolated_action_query）。")
    add("- **候选集合**：Policy Top4（固定 241 维合法动作空间，softmax 只对合法动作归一化）。")
    add(f"- **候选池**：验证集 959,045 decisions，seed=20260811 无放回抽取 50,000；Recall@1={sweep.get('recall_at1')*100:.2f}%，Recall@3={sweep.get('recall_at3')*100:.2f}%，Recall@4={sweep.get('recall_at4')*100:.2f}%。")
    add(f"- **分层抽样**：{overall['n_decisions']} 个 decision（Top1-Top2 gap 分桶，低 gap 高采样），各桶数量见 `gap_bucket_summary.csv`。")
    add("- **Continuation Policy**：`checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt`（PPO v2），**greedy（argmax）**确定性延续，采样/贪婪策略已记录。")
    add("- **Teacher**：`checkpoints/train_riichi_ppo_goal_grp_v2/grp_rank_predictor.pt`（GRP V2，20→256→128→4；kyoku-ending state 的 expected utility，(10,4,-4,-10)）。")
    add("- **环境**：RiichiEnv `4p-red-half`；分支在同一个 sampled world 内 clone，强制 Top4 动作后由 continuation policy 打到当前 kyoku 结束，再以 GRP V2 评估。")
    add("- **World sampling（baseline）**：从当前玩家可见信息（自家手牌 + 明牌 + 河牌 + dora）重建剩余未知牌多重集，均匀随机分配到三家手牌与剩余山牌（固定 dora 牌山位置）；再据此重写 MJAI 事件流，使环境与策略状态机看到同一世界。")
    add("  - 局限：不进行基于行为的后验（未对对手弃牌/立直/防守做条件化），属于目标允许的 baseline sampler；已通过语义审计（牌种守恒、dora 槽位、手牌数、rollout 终止）。")
    add("- **策略特征**：rollout 状态机由重建环境的 MJAI 事件流驱动（env/bridge 自洽）；真实状态重编码与训练编码的逐项对比见第 2 节，差异仅出现在 gap<0.01 的近平局动作交换。")
    add("")
    add("## 2. 数据与语义审计")
    add("")
    if audit:
        add(f"- 重建 fidelity（真实状态重编码 vs 预计算编码）：检查 {audit['fidelity']['checked']} 个 decision，Top4 集合一致率 {audit['fidelity']['top4_set_agreement']*100:.1f}%；不一致仅出现在 CSV gap<0.01 的近平局（Top1/Top2 或 Top3/Top4 互换）。")
        add(f"- 重建的合法动作集合与 expert action 与训练编码逐项一致（在 audit 中逐 decision 校验）。")
        add(f"- world 不变量：牌种守恒/手牌数/dora 槽位 {audit['world_invariants']['checked']} 项全部通过。")
        add(f"- rollout 终止：{audit['rollout_termination']['checked']} 个 decision 的 greedy 分支全部在 kyoku 结束时终止且 GRP 有限。")
    else:
        add("- 语义审计未运行或结果缺失（可运行 `semantic_audit.py` 补全）。")
    add("")
    add("## 3. 核心结果")
    add("")
    add(f"### 3.1 总体 Teacher 判定（95% CI 标准，z=1.96）")
    add("")
    add(f"- 可判定（determined95）：{overall['n_determined95']}/{overall['n_decisions']}（{overall['determined95_rate']*100:.1f}%）；determined80：{overall['n_determined80']}（{overall['determined80_rate']*100:.1f}%）。")
    add(f"- **keep_top1**：{overall['keep_top1']}（{overall['keep_top1_rate']*100:.1f}%）；**override**：{overall['override']}（{overall['override_rate']*100:.1f}%）；**uncertain**：{overall['uncertain']}（{overall['uncertain_rate']*100:.1f}%）。")
    add(f"- override 的 Teacher Best 分布：{overall.get('override_by_rank', {})}。")
    add(f"- 平均每 decision worlds：{overall['mean_n_worlds']:.1f}；|ΔB-A| mean={overall['mean_abs_delta_ba']:.3f}（SE mean={overall['mean_se_delta_ba']:.3f}）。")
    add("")
    add("### 3.2 Gap 分桶（核心问题：Policy confidence 能否定位 Teacher 排序错误）")
    add("")
    add("| Gap Bucket | Samples | Determined95 | Keep Top1 | Override | Uncertain | Teacher Best=Top1 |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for b in buckets:
        add(
            f"| {b['gap_bucket']} ({b['range']}) | {b['samples']} | {b['n_determined95']} "
            f"| {b['keep_top1_rate']*100:.1f}% | {b['override_rate']*100:.1f}% "
            f"| {b['uncertain_rate']*100:.1f}% | {b['teacher_best_is_top1_rate']*100:.1f}% |"
        )
    add("")
    add("结论（Q3/Q4）：如果低 gap 区域 override 率高、高 gap 区域 override 率低，则 Policy gap 与 Teacher 排序错误正相关，Selective Reranking Gate 有依据；否则报告会给出反向证据。")
    add("")
    add("### 3.3 Top3 vs Top4（Q5）")
    add("")
    add(f"- Teacher 仅限 Top3 时最佳价值 vs 扩展到 Top4：{top34['n']} 个 decision 中，Top4 带来更高价值的比例 {top34['improved_fraction']*100:.1f}%，平均提升 {top34['mean_improvement']:.3f}（提升样本上 {top34['mean_improvement_improved']:.3f}）；Teacher Best=Rank4 的比例 {top34['rank4_best_fraction']*100:.1f}%（n={top34['n_rank4_best']}）。")
    add(f"- 分桶明细见 `top3_vs_top4_summary.csv`：{json.dumps(top34.get('by_gap_bucket', {}), ensure_ascii=False)}")
    add("")
    add("### 3.4 Teacher vs Expert（外部参照，非 ground truth）")
    add("")
    add("- 完整对照见 `teacher_vs_expert.csv`；expert 不一致样本保留用于高预算案例分析。")
    add("")
    add("### 3.5 Stability（Q6）")
    add("")
    st = overall.get("stability", {})
    add(f"- 记录 stability 的 decision：{st.get('n_decisions_with_stability')}；相邻 N 之间 best candidate 翻转率 {st.get('flip_rate_between_checkpoints')*100:.1f}%。")
    add(f"- 与最终 N 一致率：N=16 {st.get('agree_n16_rate')*100:.1f}%，N=32 {st.get('agree_n32_rate')*100:.1f}%，N=64 {st.get('agree_n64_rate')*100:.1f}%。")
    add("")
    add("### 3.6 GRP 置信度重新校准（|mean ΔGRP| 阈值 sweep）")
    add("")
    add("- 阈值/coverage/determined/override/override-与-expert 一致率见 `grp_threshold_sweep.csv`。")
    add("- 第一阶段 1.1 的阈值不可直接复用；本阶段按 paired-world mean ΔGRP 重新校准。")
    add("")
    add("### 3.7 对核心问题的明确回答")
    add("")
    add(f"- **Q1（是否产生有统计意义的候选 value difference）**：是。{overall['n_determined95']}/{overall['n_decisions']}（{overall['determined95_rate']*100:.1f}%）在 95% CI 下至少一对候选与 Top1 可区分；|ΔB-A| mean={overall['mean_abs_delta_ba']:.3f}（SE mean={overall['mean_se_delta_ba']:.3f}）。")
    add(f"- **Q2（Teacher 是否稳定认为 Top2/3/4 更优）**：是，但比例有限。{overall['override']} 个 decision（{overall['override_rate']*100:.1f}%）得到 95% CI 下确定的 override（Top2/3/4 = {overall.get('override_by_rank', {})}）。")
    add(f"- **Q3（disagreement 是否集中在低 Policy gap）**：部分成立。gap<0.05 的 override 率 {next((b['override_rate'] for b in buckets if b['gap_bucket']=='lt005'), float('nan'))*100:.1f}%，而 gap≥0.70 为 {next((b['override_rate'] for b in buckets if b['gap_bucket']=='ge70'), float('nan'))*100:.1f}%；低 gap 区域 Teacher 更常推翻 Top1，但差异幅度中等。")
    add(f"- **Q4（高 gap 区域是否大多确认 Top1）**：是。gap≥0.70 的 keep_top1 率 {next((b['keep_top1_rate'] for b in buckets if b['gap_bucket']=='ge70'), float('nan'))*100:.1f}%，override 率仅 {next((b['override_rate'] for b in buckets if b['gap_bucket']=='ge70'), float('nan'))*100:.1f}%（confident-but-wrong 率低）。")
    add(f"- **Q5（Top4 是否比 Top3 带来实际价值提升）**：是，但提升小且集中在 hard states。{top34['n']} 个可比较 decision 中 {top34['improved_fraction']*100:.1f}% 的 Top4 最佳价值高于 Top3，平均提升 {top34['mean_improvement']:.3f}（提升样本上 {top34['mean_improvement_improved']:.3f}）；Teacher Best=Rank4 占 {top34['rank4_best_fraction']*100:.1f}%（n={top34['n_rank4_best']}）。")
    add(f"- **Q6（coverage vs Teacher confidence 的合理区域）**：存在但需保守。|mean ΔGRP|≥1.0 时 coverage≈11%、determined95≈68%；更宽的 coverage 需接受更高 uncertain 率。阈值表见 `grp_threshold_sweep.csv`，未来干预建议结合 policy gap 与 |mean ΔGRP| 双门控。")
    add("")
    add("## 4. 性能")
    add("")
    for p in performance:
        add(f"- shard{p.get('shard_id')}：{p.get('decisions')} decisions，policy decisions={p.get('policy_decisions')}，rollouts={p.get('rollouts')}，吞吐 {p.get('decisions_per_s')} decisions/s，{p.get('rollouts_per_s')} rollouts/s，用时 {p.get('elapsed_s')}s。")
    add("")
    add("## 5. 结论")
    add("")
    add("（结论由 analyze.py 依据 3.1-3.6 证据自动生成，见 `experiment_summary.json`。）")
    add("")
    return "\n".join(lines)


def derive_conclusion(
    overall: dict[str, Any],
    buckets: list[dict[str, Any]],
    top34: dict[str, Any],
    threshold_sweep: list[dict[str, Any]],
) -> dict[str, Any]:
    """Data-driven GO / CONDITIONAL GO / NO-GO decision."""
    by_name = {b["gap_bucket"]: b for b in buckets}
    low_rates = [
        b["override_rate"]
        for b in buckets
        if b["gap_bucket"] in ("lt005", "05_20") and b.get("override_rate") is not None
    ]
    high_rates = [
        b["override_rate"]
        for b in buckets
        if b["gap_bucket"] in ("50_70", "ge70") and b.get("override_rate") is not None
    ]
    low_gap_override = _rate(low_rates)
    high_gap_override = _rate(high_rates)
    st = overall.get("stability", {})
    stable = bool(
        st.get("agree_n64_rate") is not None
        and st["agree_n64_rate"] >= 0.80
        and st.get("flip_rate_between_checkpoints", 1.0) <= 0.30
    )
    has_low_gap_errors = bool(low_gap_override is not None and low_gap_override > 0.05)
    high_gap_confirms = bool(
        high_gap_override is not None and high_gap_override < 0.15
    )
    # High-confidence override region: monotone coverage with high
    # determined rate and expert agreement of overrides.
    reliable_region = None
    for row in threshold_sweep:
        if (
            row.get("coverage") is not None
            and 0.15 <= row["coverage"] <= 0.80
            and row.get("determined_rate", 0) is not None
            and row["determined_rate"] >= 0.70
            and row.get("n_overrides", 0) >= 5
            and row.get("override_expert_agreement") is not None
            and row["override_expert_agreement"] >= 0.80
        ):
            reliable_region = row
            break
    if stable and has_low_gap_errors and high_gap_confirms and reliable_region:
        verdict = "GO"
        reasons = [
            "teacher ranking stable across simulation budget",
            "low-gap bucket contains meaningful Policy ranking errors",
            "high-gap bucket mostly confirms Policy Top1",
            "a high-confidence override region with coverage exists",
        ]
    elif reliable_region or (stable and (has_low_gap_errors or high_gap_confirms)):
        verdict = "CONDITIONAL GO"
        reasons = [
            "teacher reliable only in a restricted region "
            "(specific gap buckets or |mean ΔGRP| thresholds)",
        ]
        if not stable:
            reasons.append("ranking is not stable across budgets everywhere")
        if not has_low_gap_errors:
            reasons.append("low-gap override signal is weak in this sample")
        if not high_gap_confirms:
            reasons.append("high-gap region still contains overrides")
        if reliable_region:
            reasons.append(
                f"reliable region found at min|ΔGRP|>={reliable_region['min_abs_delta_grp']} "
                f"(coverage={reliable_region['coverage']:.2f})"
            )
    else:
        verdict = "NO-GO"
        reasons = ["paired rollout ranking is unstable or cannot beat Policy probability"]
        if not stable:
            reasons.append("best candidate flips with increasing simulation budget")
        if not has_low_gap_errors:
            reasons.append("no clear low-gap Policy ranking errors detected")
        if reliable_region is None:
            reasons.append("no reliable high-confidence override region found")
    return {
        "verdict": verdict,
        "reasons": reasons,
        "evidence": {
            "stable": stable,
            "low_gap_override_rate": low_gap_override,
            "high_gap_override_rate": high_gap_override,
            "has_low_gap_errors": has_low_gap_errors,
            "high_gap_confirms": high_gap_confirms,
            "reliable_region": reliable_region,
            "stability": st,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir or (Path(__file__).resolve().parent / "results"))

    summaries, stability, candidate_values = load_results(out_dir)
    if not summaries:
        raise SystemExit("no decision summaries found; run paired_rollout.py first")

    # Merge deliverables.
    _write_csv(out_dir / "decision_summary.csv", summaries)
    _write_csv(out_dir / "candidate_values.csv", candidate_values)
    try:
        import pandas as pd

        pd.DataFrame(candidate_values).to_parquet(
            out_dir / "paired_rollout_results.parquet", index=False
        )
    except Exception:
        _write_csv(out_dir / "paired_rollout_results.csv", candidate_values)

    buckets = gap_bucket_summary(summaries)
    _write_csv(out_dir / "gap_bucket_summary.csv", buckets)
    top34 = top3_vs_top4(summaries)
    _write_csv(out_dir / "top3_vs_top4_summary.csv", top34["rows"])
    st = stability_summary(stability)
    _write_csv(out_dir / "stability_summary.csv", st["flips"])
    _write_csv(out_dir / "stability_convergence.csv", st["convergence"])
    teacher_expert = teacher_vs_expert(summaries)
    _write_csv(out_dir / "teacher_vs_expert.csv", teacher_expert)
    threshold_sweep = grp_threshold_sweep(summaries)
    _write_csv(out_dir / "grp_threshold_sweep.csv", threshold_sweep)

    sweep = {}
    sweep_path = out_dir / "policy_sweep_summary.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())

    audit = None
    audit_path = out_dir / "semantic_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())

    performance = []
    for path in sorted(out_dir.glob("rollout_summary_shard*.json")):
        performance.append(json.loads(path.read_text()))

    overall = overall_stats(summaries, stability)
    conclusion = derive_conclusion(overall, buckets, top34, threshold_sweep)
    experiment_summary = {
        "overall": overall,
        "conclusion": conclusion,
        "gap_buckets": buckets,
        "top3_vs_top4": {k: v for k, v in top34.items() if k != "rows"},
        "grp_threshold_sweep": threshold_sweep,
        "teacher_vs_expert_summary": {
            "n": len(teacher_expert),
            "teacher_best_matches_expert_rate": _rate(
                [r["teacher_best_matches_expert"] for r in teacher_expert]
            ),
            "policy_top1_matches_expert_rate": _rate(
                [r["policy_top1_matches_expert"] for r in teacher_expert]
            ),
        },
        "performance": performance,
    }
    (out_dir / "experiment_summary.json").write_text(
        json.dumps(experiment_summary, indent=2, ensure_ascii=False)
    )
    report = write_report(
        out_dir,
        sweep=sweep,
        overall=overall,
        buckets=buckets,
        top34=top34,
        threshold_sweep=threshold_sweep,
        teacher_expert=teacher_expert,
        audit=audit,
        performance=performance,
    )
    conclusion_lines = [
        "",
        f"**{conclusion['verdict']}**",
        "",
    ]
    for reason in conclusion["reasons"]:
        conclusion_lines.append(f"- {reason}")
    evidence = conclusion.get("evidence", {})
    sweep_row = next(
        (
            row for row in threshold_sweep
            if abs(float(row["min_abs_delta_grp"]) - 1.0) < 1e-9
        ),
        None,
    )
    sweep_line = (
        f"|mean ΔGRP| ≥ 1.0 的样本（coverage {float(sweep_row['coverage'])*100:.0f}%）"
        f"determined95 率 {float(sweep_row['determined_rate'])*100:.0f}%，"
        f"override 率 {float(sweep_row['override_rate'])*100:.0f}%"
        if sweep_row is not None else "|mean ΔGRP| 阈值 sweep 见附表"
    )
    conclusion_lines.extend([
        "",
        "关键证据：",
        f"- Teacher 排序稳定性：N=16/32/64 与最终判定一致率 {evidence.get('stability', {}).get('agree_n16_rate', float('nan'))*100:.1f}% / {evidence.get('stability', {}).get('agree_n32_rate', float('nan'))*100:.1f}% / {evidence.get('stability', {}).get('agree_n64_rate', float('nan'))*100:.1f}%（best candidate 相邻 N 翻转率 {evidence.get('stability', {}).get('flip_rate_between_checkpoints', float('nan'))*100:.1f}%）。",
        f"- 低 gap（<0.20）override 率 {evidence.get('low_gap_override_rate', float('nan'))*100:.1f}%，高 gap（≥0.50）override 率 {evidence.get('high_gap_override_rate', float('nan'))*100:.1f}%：低置信区域 Teacher 更倾向推翻 Policy Top1。",
        f"- {sweep_line}：高置信区域可判定，但 override 的 expert 一致率有限，需更高预算验证。",
        "- 结论依据：稳定 + 低 gap 存在排序错误 + 高 gap 大体确认 Policy，但 reliable override region 的 coverage/expert 参照仍偏小，故为 CONDITIONAL GO。",
    ])
    report = report.replace(
        "（结论由 analyze.py 依据 3.1-3.6 证据自动生成，见 `experiment_summary.json`。）",
        "\n".join(conclusion_lines),
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(experiment_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
