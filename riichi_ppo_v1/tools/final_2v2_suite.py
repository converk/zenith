"""Run the final 2v2 evaluation suite across goal checkpoints.

Two worker processes run concurrently on CUDA_DEVICE=0 and CUDA_DEVICE=2.
Every run is a deterministic seat-balanced 2v2 with paired walls; outputs are
written to the output directory and aggregated into ``summary.json``.

Usage (inside the Mahjong-AI conda environment):
    python -m riichi_ppo_v1.tools.final_2v2_suite \
        --out audit/reports/ppo_rl_next_goal_20260810/final_2v2_300

Re-running skips runs whose output JSON already exists.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SFT = "checkpoints/train_riichi_v13_sft/best_heuristic.pt"


RUNS = [
    # (name, model_a, model_b, hanchans)
    # --- 本 goal（current）候选 vs SFT，300 半庄 ---
    ("current_e5a_kl005", "checkpoints/train_riichi_ppo_next_e5a_kl005/checkpoint_00200.pt", SFT, 300),
    ("current_e5a_kl010", "checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00200.pt", SFT, 300),
    # --- 上一 goal（ppo_rl_goal_run_20260808）代表臂 vs SFT，300 半庄 ---
    ("prev_e3b_grp", "checkpoints/train_riichi_ppo_goal_e3_grp_reward/checkpoint_00200.pt", SFT, 300),
    ("prev_e2_opponent_mix", "checkpoints/train_riichi_ppo_goal_e2_opponent_mix/checkpoint_00200.pt", SFT, 300),
    ("prev_e4_dense", "checkpoints/train_riichi_ppo_goal_e4_dense_on_grp/checkpoint_00200.pt", SFT, 300),
    # --- 本 goal 其余 checkpoint 的 u50/u100 口径（240 半庄） ---
    ("current_e5a_kl010_u050", "checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00050.pt", SFT, 240),
    ("current_e5a_kl010_u100", "checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt", SFT, 240),
]


def _resolve(path: str) -> str:
    if path == SFT:
        path = SFT
    full = ROOT / path
    if not full.is_file():
        raise FileNotFoundError(f"checkpoint missing: {full}")
    return str(full)


def run_one(device: str, output_dir: Path, name: str, model_a: str, model_b: str, hanchans: int) -> dict:
    output = output_dir / f"{name}.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    env = dict(os.environ)
    env["CUDA_DEVICE"] = device
    cmd = [
        sys.executable, "-m", "riichi_ppo_v1.sft.head_to_head",
        "--model-a", model_a,
        "--model-b", model_b,
        "--hanchans", str(hanchans),
        "--parallel-hanchans", "24",
        "--output", str(output),
    ]
    print(f"[CUDA={device}] start {name} ({hanchans} hanchans)", flush=True)
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"run {name} failed (CUDA={device})\n"
            f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
        )
    result = json.loads(output.read_text(encoding="utf-8"))
    print(
        f"[CUDA={device}] done {name}: win={result['model_a']['team_win_rate']:.4f} "
        f"diff={result['model_a']['team_point_diff_mean']:.1f} "
        f"CI={[round(v, 1) for v in result['model_a']['team_point_diff_paired_bootstrap_ci95']]} "
        f"({elapsed:.0f}s)",
        flush=True,
    )
    return result


def worker_main(device: str, output_dir: Path) -> None:
    for index, (name, model_a, model_b, hanchans) in enumerate(RUNS):
        # Slot 0 (CUDA_DEVICE=0) takes even indices; slot 1 (CUDA_DEVICE=2)
        # takes odd indices.  CUDA_DEVICE values 0/2 cannot be used as parity.
        slot = 0 if device == "0" else 1
        if index % 2 != slot:
            continue
        run_one(device, output_dir, name, _resolve(model_a), _resolve(model_b), int(hanchans))


def write_summary(output_dir: Path) -> dict:
    rows = []
    for name, _a, _b, hanchans in RUNS:
        path = output_dir / f"{name}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        a = data["model_a"]
        rows.append({
            "name": name,
            "hanchans": data["hanchan_count"],
            "model_a": a["checkpoint"],
            "model_b": data["model_b"]["checkpoint"],
            "team_win_rate": a["team_win_rate"],
            "team_wins": a["team_wins"],
            "team_ties": a["team_ties"],
            "team_point_diff_mean": a["team_point_diff_mean"],
            "paired_bootstrap_ci95": a["team_point_diff_paired_bootstrap_ci95"],
            "first_place_rate": a["first_place_rate"],
            "individual_mean_rank": a["individual_mean_rank"],
            "elapsed_s": data["elapsed_s"],
            "selected_checkpoint": data["selected_checkpoint"],
            "selection_reason": data["selection_reason"],
        })
    rows.sort(key=lambda row: (-row["team_win_rate"], row["name"]))
    summary = {"runs": rows}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n=== summary (win rate vs model_b) ===", flush=True)
    for row in rows:
        ci = [round(v, 1) for v in row["paired_bootstrap_ci95"]]
        print(
            f"{row['name']:28s} {row['team_win_rate']:.4f} "
            f"diff={row['team_point_diff_mean']:9.1f} CI={ci} "
            f"first={row['first_place_rate']:.3f} rank={row['individual_mean_rank']:.3f}",
            flush=True,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="audit/reports/ppo_rl_next_goal_20260810/final_2v2_300")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker")
    args = parser.parse_args()
    output_dir = ROOT / args.out
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for name, model_a, model_b, hanchans in RUNS:
            print(name, hanchans, model_a, "vs", model_b)
        return
    if args.worker is not None:
        worker_main(args.worker, output_dir)
        return
    processes = []
    for device in ("0", "2"):
        processes.append(subprocess.Popen(
            [
                sys.executable, "-m", "riichi_ppo_v1.tools.final_2v2_suite",
                "--out", str(args.out), "--worker", device,
            ],
            cwd=ROOT,
            env={**os.environ, "CUDA_DEVICE": device},
        ))
    for proc in processes:
        proc.wait()
    write_summary(output_dir)
    print("all runs finished", flush=True)


if __name__ == "__main__":
    main()
