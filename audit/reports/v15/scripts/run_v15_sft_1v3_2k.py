"""Run the three V15 SFT finalists against 3x V13 SFT on fixed 2k seeds."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import time

import numpy as np

from riichi_ppo_v1.evaluation.head_to_head_1v3_shards import run_sharded_1v3


ROOT = Path("audit/reports/v15/eval/v15_sft_1v3_2k")
BASELINE_PATH = Path(
    "audit/reports/v15/eval/v14_u510_reval_2k/"
    "sft_baseline/vs_sft_u000.json"
)
MODEL_B = "checkpoints/train_riichi_v13/sft/best_heuristic.pt"
SEED_BASE = 202608140000
HANCHANS_PER_PROCESS = 200
PROCESSES = 10

EXPERIMENTS = {
    "stage_b_best_heuristic": {
        "checkpoint": "checkpoints/train_riichi_v15/sft/stage_b/best_heuristic.pt",
        "step": 18000,
        "devices": ("0", "1"),
    },
    "stage_b_best_validation": {
        "checkpoint": "checkpoints/train_riichi_v15/sft/stage_b/best.pt",
        "step": 30000,
        "devices": ("2", "3"),
    },
    "stage_a_best_heuristic": {
        "checkpoint": "checkpoints/train_riichi_v15/sft/stage_a/best_heuristic.pt",
        "step": 6000,
        "devices": ("0", "1"),
    },
}


def run_one(name: str) -> dict:
    spec = EXPERIMENTS[name]
    started = time.perf_counter()
    print(
        f"START {name} step={spec['step']} devices={spec['devices']} "
        f"seed_base={SEED_BASE}",
        flush=True,
    )
    summary = run_sharded_1v3(
        spec["checkpoint"],
        MODEL_B,
        update=int(spec["step"]),
        processes=PROCESSES,
        hanchans_per_process=HANCHANS_PER_PROCESS,
        parallel_hanchans=HANCHANS_PER_PROCESS,
        devices=tuple(spec["devices"]),
        seed_base=SEED_BASE,
        output_dir=ROOT / name,
    )
    result = summary["model_a"]
    print(
        f"DONE {name} first={result['first_place_rate']:.4f} "
        f"top2={result['top2_rate']:.4f} rank={result['mean_rank']:.4f} "
        f"pd={result['point_diff_mean']:+.1f} "
        f"ci={result['point_diff_bootstrap_ci95']} "
        f"elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return summary


def paired_bootstrap(candidate: list[float], baseline: list[float]) -> dict:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if candidate_array.shape != baseline_array.shape:
        raise ValueError("candidate and baseline point-diff samples do not align")
    differences = candidate_array - baseline_array
    rng = np.random.default_rng(20260814)
    boot = np.empty(5000, dtype=np.float64)
    for index in range(5000):
        selected = rng.integers(0, differences.size, size=differences.size)
        boot[index] = float(differences[selected].mean())
    return {
        "mean": float(differences.mean()),
        "ci95": [float(value) for value in np.percentile(boot, [2.5, 97.5])],
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if int(baseline["seed_base"]) != SEED_BASE:
        raise RuntimeError("existing four-V13 baseline does not use the required seeds")
    if int(baseline["hanchan_count"]) != PROCESSES * HANCHANS_PER_PROCESS:
        raise RuntimeError("existing four-V13 baseline does not contain 2,000 hanchans")

    started = time.perf_counter()
    results: dict[str, dict] = {}

    # These two experiments use disjoint physical GPU groups.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            name: executor.submit(run_one, name)
            for name in ("stage_b_best_heuristic", "stage_b_best_validation")
        }
        for name, future in futures.items():
            results[name] = future.result()

    # Run the stage-A finalist after both stage-B evaluations release their GPUs.
    results["stage_a_best_heuristic"] = run_one("stage_a_best_heuristic")

    baseline_samples = baseline["model_a"]["point_diff_samples"]
    comparison = {
        "protocol": "V15 SFT vs 3x V13 SFT, 10 processes x 200 hanchans",
        "seed_base": SEED_BASE,
        "seed_ranges": [
            [SEED_BASE + shard * HANCHANS_PER_PROCESS,
             SEED_BASE + (shard + 1) * HANCHANS_PER_PROCESS - 1]
            for shard in range(PROCESSES)
        ],
        "hanchans_per_experiment": PROCESSES * HANCHANS_PER_PROCESS,
        "baseline": str(BASELINE_PATH),
        "elapsed_s": time.perf_counter() - started,
        "experiments": {},
    }
    for name, summary in results.items():
        model = summary["model_a"]
        comparison["experiments"][name] = {
            "checkpoint": EXPERIMENTS[name]["checkpoint"],
            "step": EXPERIMENTS[name]["step"],
            "first_place_rate": model["first_place_rate"],
            "top2_rate": model["top2_rate"],
            "fourth_place_rate": model["fourth_place_rate"],
            "mean_rank": model["mean_rank"],
            "point_diff_mean": model["point_diff_mean"],
            "point_diff_bootstrap_ci95": model["point_diff_bootstrap_ci95"],
            "paired_point_diff_vs_baseline": paired_bootstrap(
                model["point_diff_samples"], baseline_samples,
            ),
            "kyoku_metrics": model["kyoku_metrics"],
        }
    (ROOT / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"ALL_DONE elapsed_s={comparison['elapsed_s']:.1f}", flush=True)


if __name__ == "__main__":
    main()
