#!/usr/bin/env python
"""对比基准日志并计算 rollout/update/总时间加速比。

读取一个「优化后」console log(含 ``iteration=N ... rollout_wall_s=...``
行)与一个「基线」console log,提取第 2、3 轮(第 1 轮视为预热),汇总均值并
打印加速比。

用法:
  python audit/reports/v17/scripts/compare_benchmark.py \
      --new logs/v17/perf_512g4e_soa.log \
      --base logs/v17/perf_base_true_512g4e.log
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse(log_path: str) -> list[dict]:
    field_pattern = re.compile(r"(?:^|\s)iteration=(\d+)(?:\s|\Z)")
    value_pattern = re.compile(
        r"(?:^|\s)(rollout_wall_s|update_wall_s|algorithm_wall_s)=([\d.]+)(?:\s|\Z)"
    )
    rows = []
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        iteration_match = field_pattern.search(line)
        if not iteration_match:
            continue
        values = {key: float(value) for key, value in value_pattern.findall(line)}
        if "rollout_wall_s" in values and "update_wall_s" in values and "algorithm_wall_s" in values:
            rows.append({
                "iteration": int(iteration_match.group(1)),
                "rollout": values["rollout_wall_s"],
                "update": values["update_wall_s"],
                "total": values["algorithm_wall_s"],
            })
    return rows


def summarize(rows: list[dict]) -> dict[str, float]:
    """丢弃第一轮(预热),对第 2..N 轮取均值。"""
    measured = rows[1:] if len(rows) > 1 else rows
    if not measured:
        return {}
    return {
        key: float(sum(row[key] for row in measured) / len(measured))
        for key in ("rollout", "update", "total")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()

    new_rows = parse(args.new)
    base_rows = parse(args.base)
    new_stats = summarize(new_rows)
    base_stats = summarize(base_rows)
    if not new_stats or not base_stats:
        raise SystemExit("one of the logs has no measured iterations")

    print(f"{'metric':<12} {'baseline(s)':>12} {'new(s)':>12} {'speedup':>8}")
    for key in ("rollout", "update", "total"):
        print(f"{key:<12} {base_stats[key]:>12.3f} {new_stats[key]:>12.3f} "
              f"{base_stats[key] / new_stats[key]:>8.2f}x")
    print(f"\nnew measured rows: {len(new_rows) - 1 if len(new_rows) > 1 else 1}; "
          f"base measured rows: {len(base_rows) - 1 if len(base_rows) > 1 else 1}")


if __name__ == "__main__":
    main()
