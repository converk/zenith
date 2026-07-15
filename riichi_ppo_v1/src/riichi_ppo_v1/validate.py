"""Run strict real-environment validation for the V3 event and V2 action protocols."""

from __future__ import annotations

import argparse
import json

from .validation import run_random_coverage, write_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--output", default="riichi_ppo_v1_coverage.json")
    args = parser.parse_args()
    summary = run_random_coverage(args.games, args.seed, args.max_steps)
    write_coverage(summary, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
