"""Batch 2v2 head-to-head evaluation of PPO checkpoints against an SFT baseline.

Usage (from the workspace root):
    CUDA_DEVICE=3 conda run -n Mahjong-AI python \
        riichi_ppo_v1/tools/benchmark_ppo_vs_sft.py \
        --baseline checkpoints/train_riichi_v11_sft_40pct_2v2_selection/best_heuristic.snapshot.pt \
        --output-dir checkpoints/train_riichi_v11_ppo_selected/ppo_vs_sft_benchmark \
        --hanchans 320 --parallel-hanchans 24

Each candidate PPO checkpoint is evaluated against the same baseline with the
same seed base, so results are directly comparable.  A summary table is written
to ``<output-dir>/summary.json`` and printed at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Project device convention, mirror head_to_head.py.
if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

# Make sure the project package is importable when invoked as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from riichi_ppo_v1.sft.head_to_head import evaluate_2v2  # noqa: E402


DEFAULT_CANDIDATES = [
    "checkpoints/train_riichi_v11_ppo_selected/checkpoint_00050.pt",
    "checkpoints/train_riichi_v11_ppo_selected/checkpoint_00100.pt",
    "checkpoints/train_riichi_v11_ppo_selected/checkpoint_00600.pt",
    "checkpoints/train_riichi_v11_ppo_selected/checkpoint_00900.pt",
]


def _resolve(path: str | Path) -> str:
    return str(Path(path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to the SFT baseline checkpoint.",
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=DEFAULT_CANDIDATES,
        help="PPO checkpoints to evaluate against the baseline.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/train_riichi_v11_ppo_selected/ppo_vs_sft_benchmark",
        help="Directory to write per-checkpoint JSON results and a summary.",
    )
    parser.add_argument("--hanchans", type=int, default=320)
    parser.add_argument("--parallel-hanchans", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=20260730)
    parser.add_argument("--game-mode", default="4p-red-half")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for both models when --model-a/b-device are not given.",
    )
    parser.add_argument(
        "--baseline-device",
        default=None,
        help="Optional separate device for the baseline model.",
    )
    parser.add_argument(
        "--ppo-device",
        default=None,
        help="Optional separate device for the PPO candidate model.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = _resolve(args.baseline)
    print(f"[benchmark] baseline: {baseline_path}", flush=True)
    print(f"[benchmark] candidates: {args.candidates}", flush=True)
    print(
        f"[benchmark] hanchans={args.hanchans} parallel={args.parallel_hanchans} "
        f"seed_base={args.seed_base} game_mode={args.game_mode}",
        flush=True,
    )

    per_checkpoint: list[dict] = []
    overall_start = time.perf_counter()

    for idx, candidate in enumerate(args.candidates, 1):
        candidate_path = _resolve(candidate)
        out_file = output_dir / (Path(candidate_path).stem + ".json")
        print(
            f"\n[benchmark] ({idx}/{len(args.candidates)}) "
            f"evaluating {candidate_path} vs {baseline_path}",
            flush=True,
        )
        if out_file.exists():
            print(f"[benchmark] cached result found at {out_file}; loading", flush=True)
            result = json.loads(out_file.read_text())
        else:
            t0 = time.perf_counter()
            result = evaluate_2v2(
                candidate_path,
                baseline_path,
                device=args.device,
                model_a_device=args.ppo_device,
                model_b_device=args.baseline_device,
                hanchan_count=args.hanchans,
                parallel_hanchans=args.parallel_hanchans,
                seed_base=args.seed_base,
                game_mode=args.game_mode,
                max_steps=args.max_steps,
            )
            result["elapsed_s"] = time.perf_counter() - t0
            tmp = out_file.with_suffix(out_file.suffix + ".tmp")
            tmp.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tmp.replace(out_file)
        # Print compact per-checkpoint stats.
        a = result["model_a"]
        b = result["model_b"]
        print(
            f"[benchmark] done {Path(candidate_path).name}: "
            f"PPO win={a['team_win_rate']:.4f} pt_diff={a['team_point_diff_mean']:+.2f} "
            f"first_place_rate={a['first_place_rate']:.4f} mean_rank={a['individual_mean_rank']:.3f} | "
            f"SFT win={b['team_win_rate']:.4f} pt_diff={b['team_point_diff_mean']:+.2f} "
            f"first_place_rate={b['first_place_rate']:.4f} mean_rank={b['individual_mean_rank']:.3f} | "
            f"reason={result['selection_reason']} elapsed={result['elapsed_s']:.1f}s",
            flush=True,
        )
        per_checkpoint.append({"candidate": candidate_path, "result": result})

    # ---------- summary ranking ----------
    summary = {
        "baseline": baseline_path,
        "hanchans": args.hanchans,
        "seed_base": args.seed_base,
        "game_mode": args.game_mode,
        "results": [
            {
                "checkpoint": r["candidate"],
                "team_win_rate": r["result"]["model_a"]["team_win_rate"],
                "team_point_diff_mean": r["result"]["model_a"]["team_point_diff_mean"],
                "first_place_rate": r["result"]["model_a"]["first_place_rate"],
                "individual_mean_rank": r["result"]["model_a"]["individual_mean_rank"],
                "elapsed_s": r["result"]["elapsed_s"],
                "selection_reason": r["result"]["selection_reason"],
                "selected_against_baseline": r["result"]["selected_checkpoint"]
                == r["candidate"],
            }
            for r in per_checkpoint
        ],
    }

    # Rank by team_win_rate (higher better), then team_point_diff_mean (higher better),
    # then first_place_rate, then lower individual_mean_rank.
    def _key(r: dict) -> tuple:
        return (
            r["team_win_rate"],
            r["team_point_diff_mean"],
            r["first_place_rate"],
            -r["individual_mean_rank"],
        )

    summary["ranking_by_team_win_rate"] = sorted(
        summary["results"], key=_key, reverse=True
    )

    summary_path = output_dir / "summary.json"
    tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(summary_path)

    overall_elapsed = time.perf_counter() - overall_start
    print("\n" + "=" * 80, flush=True)
    print("[benchmark] ALL DONE — ranking (by team_win_rate, then point_diff)", flush=True)
    print("=" * 80, flush=True)
    print(
        f"{'rank':>4}  {'checkpoint':<48}  {'win_rate':>8}  {'pt_diff':>8}  {'first%':>6}  {'mean_rank':>9}",
        flush=True,
    )
    for i, r in enumerate(summary["ranking_by_team_win_rate"], 1):
        print(
            f"{i:>4}  {Path(r['checkpoint']).name:<48}  "
            f"{r['team_win_rate']:>8.4f}  {r['team_point_diff_mean']:>+8.2f}  "
            f"{r['first_place_rate']:>6.4f}  {r['individual_mean_rank']:>9.3f}",
            flush=True,
        )
    print(f"\n[benchmark] summary written to {summary_path}", flush=True)
    print(f"[benchmark] total elapsed {overall_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
