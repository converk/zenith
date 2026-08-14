"""2400-hanchan 1v3 comparison: u510 vs 3x u120, and 4x u120 all-play.

Both experiments use the identical 1v3 protocol (candidate seat i%4, greedy,
10 processes x 240 hanchans, CUDA_DEVICE=0 for the first 5 shards and
CUDA_DEVICE=2 for the last 5) with seed base 20260842, so the candidate-seat
data of u510 (1v3 vs 3x u120) and u120 (4x u120 all-play) are directly
comparable position-by-position.
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import time

from riichi_ppo_v1.sft.evaluation_cases import merge_evaluation_summaries
from riichi_ppo_v1.evaluation.head_to_head_1v3_shards import run_sharded_1v3


OUTPUT_DIR = Path("audit/reports/v14/eval/eval_510_vs_120_2400")
SEED_BASE = 20260842
MODEL_120 = "checkpoints/train_riichi_v14/checkpoint_00120.pt"
MODEL_510 = "checkpoints/train_riichi_v14/checkpoint_00510.pt"


def run_one(update: int, model_a: str, model_b: str) -> dict:
    started = time.perf_counter()
    summary = run_sharded_1v3(
        model_a,
        model_b,
        update=update,
        processes=10,
        hanchans_per_process=240,
        parallel_hanchans=240,
        devices=("0", "2"),
        seed_base=SEED_BASE,
        output_dir=OUTPUT_DIR,
    )
    model = summary["model_a"]
    print(
        f"EXPERIMENT_DONE u={update} model_a={model_a.split('/')[-1]} "
        f"model_b={model_b.split('/')[-1]} first={model['first_place_rate']:.4f} "
        f"top2={model['top2_rate']:.4f} rank={model['mean_rank']:.3f} "
        f"pd={model['point_diff_mean']:+.1f} "
        f"ci={model['point_diff_bootstrap_ci95']} "
        f"elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_510 = executor.submit(run_one, 510, MODEL_510, MODEL_120)
        future_120 = executor.submit(run_one, 120, MODEL_120, MODEL_120)
        summary_510 = future_510.result()
        summary_120 = future_120.result()
    shards_510 = sorted(
        OUTPUT_DIR.glob("shards/vs_sft_u510_shard*.json")
    )
    shards_120 = sorted(
        OUTPUT_DIR.glob("shards/vs_sft_u120_shard*.json")
    )

    def aggregate(summary: dict, shard_paths: list[Path]) -> dict:
        shards = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
        semantic = merge_evaluation_summaries(
            [shard["model_a"]["semantic_metrics"] for shard in shards]
        )
        return {
            "checkpoint": summary["model_a"]["checkpoint"],
            "first_place_rate": summary["model_a"]["first_place_rate"],
            "first_place_count": summary["model_a"]["first_place_count"],
            "top2_rate": summary["model_a"]["top2_rate"],
            "top2_count": summary["model_a"]["top2_count"],
            "fourth_place_rate": summary["model_a"]["fourth_place_rate"],
            "fourth_place_count": summary["model_a"]["fourth_place_count"],
            "mean_rank": summary["model_a"]["mean_rank"],
            "point_diff_mean": summary["model_a"]["point_diff_mean"],
            "point_diff_bootstrap_ci95": summary["model_a"][
                "point_diff_bootstrap_ci95"
            ],
            "point_diff_positive_rate": sum(
                1.0 for value in summary["model_a"]["point_diff_samples"] if value > 0
            )
            / len(summary["model_a"]["point_diff_samples"]),
            "kyoku_metrics": summary["model_a"]["kyoku_metrics"],
            "semantic_metrics": {
                key: value
                for key, value in semantic.items()
                if key.startswith("model_a/")
            },
        }

    comparison = {
        "protocol": "1v3, 2400 hanchans, candidate seat i%4, greedy",
        "seed_base": SEED_BASE,
        "hanchans_per_experiment": 2400,
        "processes_per_experiment": 10,
        "devices": ["0", "2"],
        "u510_vs_3x_u120": aggregate(summary_510, shards_510),
        "u120_vs_3x_u120": aggregate(summary_120, shards_120),
        "elapsed_s": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("ALL_DONE", flush=True)
    for name in ("u510_vs_3x_u120", "u120_vs_3x_u120"):
        experiment = comparison[name]
        print(
            f"{name}: first={experiment['first_place_rate']:.4f} "
            f"top2={experiment['top2_rate']:.4f} "
            f"fourth={experiment['fourth_place_rate']:.4f} "
            f"rank={experiment['mean_rank']:.3f} "
            f"pd={experiment['point_diff_mean']:+.1f} "
            f"ci={experiment['point_diff_bootstrap_ci95']} "
            f"pd_pos={experiment['point_diff_positive_rate']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
