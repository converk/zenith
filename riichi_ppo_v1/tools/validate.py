"""在真实环境中严格验证语义状态与动作协议。"""

from __future__ import annotations

import argparse
import json

from ..model import KyokuTransformerActorCritic
from ..model.parameter_count import assert_v19_parameter_contract
from ..model.validation import run_random_coverage, write_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--output", default="riichi_ppo_v1_coverage.json")
    parser.add_argument("--parameter-contract", action="store_true")
    args = parser.parse_args()
    if args.parameter_contract:
        print(json.dumps(assert_v19_parameter_contract(KyokuTransformerActorCritic()), indent=2, sort_keys=True))
        return
    summary = run_random_coverage(args.games, args.seed, args.max_steps)
    write_coverage(summary, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
