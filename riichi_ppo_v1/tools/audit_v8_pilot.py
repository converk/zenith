"""Fail fast when a completed V8 pilot misses its reward/health gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="checkpoints/train_riichi_v8_pilot/performance.jsonl",
    )
    parser.add_argument("--tail", type=int, default=50)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < 2:
        raise SystemExit("pilot log has too few completed updates")
    stable = rows[-max(1, int(args.tail)):]
    failures: list[str] = []
    for name in ("discard", "call"):
        actual_key = f"reward_scale/{name}_to_kyoku_ratio"
        target_key = f"reward_scale/{name}_target_ratio"
        checked = 0
        for row in stable:
            if actual_key not in row or target_key not in row:
                continue
            checked += 1
            actual, target = float(row[actual_key]), float(row[target_key])
            if target > 0 and abs(actual - target) / target > 0.15:
                failures.append(
                    f"iteration {row.get('iteration')}: {name} ratio {actual:.4f} "
                    f"outside target {target:.4f} ±15%"
                )
        if not checked:
            failures.append(f"missing {name} reward-scale metrics")
    call_opportunities = sum(float(row.get("train/action/call_opportunity_count", 0.0)) for row in stable)
    call_accepts = sum(
        float(row.get("train/action/call_opportunity_accept_rate", 0.0))
        * float(row.get("train/action/call_opportunity_count", 0.0))
        for row in stable
    )
    call_nonzero = sum(float(row.get("rollout/reward_scale/call_nonzero_count", 0.0)) for row in stable)
    if call_opportunities > 0 and call_accepts > 0 and call_nonzero <= 0:
        failures.append("call opportunities and accepted calls exist, but call regret stayed zero")
    for row in stable:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                failures.append(f"iteration {row.get('iteration')}: non-finite metric {key}")
                break
    if failures:
        raise SystemExit("V8 pilot audit failed:\n- " + "\n- ".join(failures[:20]))
    print(
        f"V8 pilot audit passed: updates={len(rows)} stable_window={len(stable)} "
        f"call_nonzero={call_nonzero:.0f}"
    )


if __name__ == "__main__":
    main()
