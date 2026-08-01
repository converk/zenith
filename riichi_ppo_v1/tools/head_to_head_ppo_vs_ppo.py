"""2v2 between two PPO checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from riichi_ppo_v1.sft.head_to_head import evaluate_2v2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--hanchans", type=int, default=320)
    parser.add_argument("--parallel-hanchans", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=20260730)
    parser.add_argument("--game-mode", default="4p-red-half")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-a-device")
    parser.add_argument("--model-b-device")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = evaluate_2v2(
        args.model_a,
        args.model_b,
        device=args.device,
        model_a_device=args.model_a_device,
        model_b_device=args.model_b_device,
        hanchan_count=args.hanchans,
        parallel_hanchans=args.parallel_hanchans,
        seed_base=args.seed_base,
        game_mode=args.game_mode,
        max_steps=args.max_steps,
    )
    result["elapsed_s"] = time.perf_counter() - t0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)

    a = result["model_a"]
    b = result["model_b"]
    print("\n" + "=" * 80, flush=True)
    print(f"model_a (A): {Path(a['checkpoint']).name}", flush=True)
    print(f"  win={a['team_win_rate']:.4f} pt_diff={a['team_point_diff_mean']:+.2f} "
          f"first={a['first_place_rate']:.4f} mean_rank={a['individual_mean_rank']:.3f}", flush=True)
    print(f"model_b (B): {Path(b['checkpoint']).name}", flush=True)
    print(f"  win={b['team_win_rate']:.4f} pt_diff={b['team_point_diff_mean']:+.2f} "
          f"first={b['first_place_rate']:.4f} mean_rank={b['individual_mean_rank']:.3f}", flush=True)
    print(f"\nselected: {Path(result['selected_checkpoint']).name} (reason: {result['selection_reason']})", flush=True)
    print(f"elapsed: {result['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
