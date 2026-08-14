"""Holdout 2000-hanchan 1v3 evaluation for the top V14 checkpoints.

Protocol is identical to the fixed 1600-hanchan curve (10 processes x 200
hanchans, candidate seat i%4, greedy, CUDA_DEVICE=0 for the first 5 shards and
CUDA_DEVICE=2 for the last 5), but uses seed base 20260822 so every one of the
2000 hanchans is disjoint from the original 20260812-based set.  Experiments
run two at a time (2 x 10 = 20 shard processes concurrently).
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import time

from riichi_ppo_v1.evaluation.head_to_head_1v3_shards import run_sharded_1v3


OUTPUT_DIR = Path("audit/reports/v14/eval/eval_holdout2k")
SEED_BASE = 20260822
MODEL_B = "checkpoints/train_riichi_v13/sft/best_heuristic.pt"
MODEL_A = {
    0: MODEL_B,  # four-seat SFT baseline
    120: "checkpoints/train_riichi_v14/checkpoint_00120.pt",
    510: "checkpoints/train_riichi_v14/checkpoint_00510.pt",
    660: "checkpoints/train_riichi_v14/checkpoint_00660.pt",
}
PAIRS = ((120, 660), (510, 0))


def run_one(update: int) -> dict:
    started = time.perf_counter()
    summary = run_sharded_1v3(
        MODEL_A[update],
        MODEL_B,
        update=update,
        processes=10,
        hanchans_per_process=200,
        parallel_hanchans=200,
        devices=("0", "2"),
        seed_base=SEED_BASE,
        output_dir=OUTPUT_DIR,
    )
    elapsed = time.perf_counter() - started
    model_a = summary["model_a"]
    print(
        f"EXPERIMENT_DONE u={update} first={model_a['first_place_rate']:.4f} "
        f"top2={model_a['top2_rate']:.4f} rank={model_a['mean_rank']:.3f} "
        f"pd={model_a['point_diff_mean']:+.1f} "
        f"ci={model_a['point_diff_bootstrap_ci95']} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: dict[int, dict] = {}
    for left, right in PAIRS:
        print(f"PAIR_START u{left}+u{right}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(run_one, left)
            right_future = executor.submit(run_one, right)
            results[left] = left_future.result()
            results[right] = right_future.result()
        print(f"PAIR_DONE u{left}+u{right}", flush=True)
    comparison = {
        "protocol": "1v3 holdout 2000 hanchans",
        "seed_base": SEED_BASE,
        "hanchans_per_experiment": 2000,
        "processes_per_experiment": 10,
        "devices": ["0", "2"],
        "model_b": str(MODEL_B),
        "elapsed_s": time.perf_counter() - started,
        "experiments": {
            str(update): {
                "checkpoint": str(MODEL_A[update]),
                "first_place_rate": summary["model_a"]["first_place_rate"],
                "top2_rate": summary["model_a"]["top2_rate"],
                "fourth_place_rate": summary["model_a"]["fourth_place_rate"],
                "mean_rank": summary["model_a"]["mean_rank"],
                "point_diff_mean": summary["model_a"]["point_diff_mean"],
                "point_diff_bootstrap_ci95": summary["model_a"][
                    "point_diff_bootstrap_ci95"
                ],
                "kyoku_metrics": summary["model_a"]["kyoku_metrics"],
            }
            for update, summary in results.items()
        },
    }
    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("ALL_DONE", flush=True)
    for update in (0, 120, 510, 660):
        experiment = comparison["experiments"][str(update)]
        print(
            f"u{update:>3} first={experiment['first_place_rate']:.4f} "
            f"top2={experiment['top2_rate']:.4f} "
            f"rank={experiment['mean_rank']:.3f} "
            f"pd={experiment['point_diff_mean']:+.1f} "
            f"ci={experiment['point_diff_bootstrap_ci95']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
