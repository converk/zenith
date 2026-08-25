#!/usr/bin/env python3
"""汇总三轮 worker 线程/SoA 基准与真实规模验证为机器可读 JSON。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = (
    "transitions",
    "rollout/games",
    "rollout/kyokus",
    "rollout/grp_calls",
    "rollout/wall_s",
    "update/wall_s",
    "iteration/algorithm_wall_s",
    "rollout/worker/rollout_s/min",
    "rollout/worker/rollout_s/max",
    "rollout/worker/rollout_s/p50",
    "rollout/worker/rollout_s/p90",
    "rollout/worker/worker_soa_pack_s/p50",
    "rollout/worker/worker_soa_pack_s/p90",
    "rollout/result_get_s",
    "rollout/object_store_publish_gap_estimate_s",
    "rollout/transition_assembly_s",
    "rollout/return_array_count",
    "rollout/return_array_bytes",
    "ppo/update/configured_epochs",
    "ppo/update/epochs_completed",
    "ppo/update/planned_minibatches",
    "ppo/update/executed_minibatches",
    "ppo/update/executed_transition_samples",
    "gpu/utilization.gpu/mean",
    "gpu/utilization.gpu/max",
)


def _load(path: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summary(path: str, workers: int) -> dict[str, Any]:
    rows = _load(path)
    scored = rows[1:] if len(rows) >= 3 else rows
    metrics: dict[str, Any] = {}
    for name in METRICS:
        values = [float(row[name]) for row in scored if name in row]
        if values:
            metrics[name] = {
                "values": values,
                "mean": sum(values) / len(values),
            }
    # 旧 JSONL 中这三项按 worker 均值写入;同时给出可比较的全局总数。
    for name in (
        "rollout/grp_calls",
        "rollout/return_array_bytes",
    ):
        if name in metrics:
            metrics[name]["global_mean"] = metrics[name]["mean"] * workers
    return {
        "path": str(Path(path)),
        "rounds": len(rows),
        "scored_rounds": len(scored),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--thread-limit", required=True)
    parser.add_argument("--step2", required=True)
    parser.add_argument("--worker-soa", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "workers": args.workers,
        "runs": {
            "thread_limit": _summary(args.thread_limit, args.workers),
            "step2": _summary(args.step2, args.workers),
            "worker_soa": _summary(args.worker_soa, args.workers),
            "real": _summary(args.real, args.workers),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
