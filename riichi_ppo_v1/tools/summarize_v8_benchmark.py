"""Summarize V8 benchmark iterations 2–3 after excluding the warm-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "rollout/wall_s",
    "rollout/timing/rollout/reward_analysis/worker_mean_total_s",
    "rollout/inference_actor/inference/full_forward/total_s",
    "update/wall_s",
    "iteration/sps",
    "iteration/model_forward_sps",
    "system/learner_gpu_peak_allocated_mb",
    "rollout/reward_analysis/cache_hit_rate",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="checkpoints/train_riichi_v8_benchmark/performance.jsonl",
    )
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < 3:
        raise SystemExit("benchmark needs three completed iterations")
    steady = rows[-2:]
    print(f"warmup_iteration={rows[-3].get('iteration')}")
    print("steady_iterations=" + ",".join(str(row.get("iteration")) for row in steady))
    for name in METRICS:
        values = [float(row[name]) for row in steady if name in row]
        if values:
            print(
                f"{name}: iter2={values[0]:.6f} iter3={values[-1]:.6f} "
                f"mean={float(np.mean(values)):.6f}"
            )


if __name__ == "__main__":
    main()
