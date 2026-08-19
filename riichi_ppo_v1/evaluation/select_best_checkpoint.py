"""按 1V3 vs SFT 表现选择训练期间最佳 checkpoint。

遍历 ``audit/reports/<版本号>/eval`` 下的 ``vs_sft_uNNN.json`` 汇总,优先按
``point_diff_vs_mean_opponent_mean`` 取最大,缺失/相等时回退到
``mean_rank`` 取最小;输出 JSON 含 best checkpoint 路径与关键指标。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .mechanism import TOTAL_1V3_HANCHANS

_SUMMARY_PATTERN = re.compile(r"vs_sft_u(\d{2,5})\.json")


def iter_eval_summaries(eval_dir: str | Path):
    """按 update 顺序产出 (update, summary dict)。"""
    directory = Path(eval_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"eval dir does not exist: {directory}")
    for path in sorted(directory.glob("vs_sft_u*.json")):
        match = _SUMMARY_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        with path.open(encoding="utf-8") as file:
            summary = json.load(file)
        yield int(match.group(1)), summary


def select_best_checkpoint(
    eval_dir: str | Path,
    *,
    metric: str = "point_diff_vs_mean_opponent_mean",
    fallback_metric: str = "mean_rank",
    require_rate: float = 1.0,
) -> dict:
    """选择 1V3 vs SFT 表现最佳的 checkpoint。

    - 主指标 ``metric``(默认 point_diff)取最大;缺失或打平用
      ``fallback_metric``(默认 mean_rank)取最小。
    - ``require_rate``:只考虑已完成该比例半庄的评测(默认 1.0 = 全部完成)。
    """
    best: dict | None = None
    best_key: tuple[float, float, int] | None = None
    summaries: list[dict] = []
    for update, summary in iter_eval_summaries(eval_dir):
        hanchan_count = int(summary.get("hanchan_count", TOTAL_1V3_HANCHANS))
        summaries.append({"update": update, **summary})
        if hanchan_count < int(require_rate * TOTAL_1V3_HANCHANS):
            continue
        model_a = summary.get("model_a") or {}
        primary = model_a.get(metric)
        fallback = model_a.get(fallback_metric)
        if primary is None and fallback is None:
            continue
        key = (
            float(primary) if primary is not None else float("-inf"),
            -float(fallback) if fallback is not None else float("-inf"),
            -int(update),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "update": int(update),
                "checkpoint": (
                    str(Path(str(model_a.get("checkpoint", ""))).resolve())
                    if model_a.get("checkpoint")
                    else None
                ),
                "metric": metric,
                "metric_value": float(primary) if primary is not None else None,
                "fallback_metric": fallback_metric,
                "fallback_value": float(fallback) if fallback is not None else None,
                "first_place_rate": model_a.get("first_place_rate"),
                "top2_rate": model_a.get("top2_rate"),
                "fourth_place_rate": model_a.get("fourth_place_rate"),
                "mean_rank": fallback,
                "point_diff_vs_mean_opponent_mean": primary,
                "point_diff_vs_mean_opponent_bootstrap_ci95": model_a.get(
                    "point_diff_vs_mean_opponent_bootstrap_ci95"
                ),
            }
    if best is None:
        raise RuntimeError(
            f"no complete 1v3 summaries found under {eval_dir} with {metric}"
        )
    return {"best": best, "summaries": sorted(summaries, key=lambda item: item["update"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument(
        "--metric", default="point_diff_vs_mean_opponent_mean",
    )
    parser.add_argument("--fallback-metric", default="mean_rank")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_best_checkpoint(
        args.eval_dir,
        metric=args.metric,
        fallback_metric=args.fallback_metric,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result["best"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()