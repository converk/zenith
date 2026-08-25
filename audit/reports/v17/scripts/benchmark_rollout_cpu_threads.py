#!/usr/bin/env python3
"""并发模拟 rollout actor,对 step 线程与 CPU 线程限流做 A/B/C/D 微基准。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any


THREAD_ENV_NAMES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _thread_count() -> int:
    try:
        return sum(1 for _entry in Path("/proc/self/task").iterdir())
    except OSError:
        return 0


def _child(args: argparse.Namespace) -> int:
    # 子进程必须先设环境再 import NumPy/PyTorch,与 Ray runtime_env 的时序一致。
    if args.cpu_threads > 0:
        for name in THREAD_ENV_NAMES:
            os.environ[name] = str(args.cpu_threads)
    else:
        for name in THREAD_ENV_NAMES:
            os.environ.pop(name, None)

    import numpy as np
    import torch
    from riichienv import BatchedRiichiEnv
    from riichi_ppo_v1.model.grp import GRPModel

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(args.cpu_threads)

    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    envs = BatchedRiichiEnv(
        args.envs,
        seed=args.seed + args.worker_id * 1_000,
        step_threads=args.step_threads,
        game_mode=args.game_mode,
    )
    observations = list(envs.reset())
    step_started = time.perf_counter()
    table_steps = 0
    for _tick in range(args.ticks):
        actions_by_env: list[dict[int, Any]] = []
        for observations_by_seat in observations:
            row: dict[int, Any] = {}
            for seat, observation in observations_by_seat.items():
                legal = observation.legal_actions()
                if legal:
                    row[int(seat)] = legal[0]
            actions_by_env.append(row)
        observations = list(envs.step_batch(actions_by_env))
        done_indices = [
            index for index, done in enumerate(envs.done()) if bool(done)
        ]
        if done_indices:
            observations = list(envs.reset_indices(done_indices))
        table_steps += args.envs
    step_s = time.perf_counter() - step_started

    generator = torch.Generator().manual_seed(args.seed + args.worker_id)
    model = GRPModel().eval()
    features = torch.randn((1, 8, 7), generator=generator)
    lengths = torch.tensor([8], dtype=torch.long)
    with torch.inference_mode():
        for _warmup in range(16):
            model(features, lengths)
        grp_started = time.perf_counter()
        for _iteration in range(args.grp_iterations):
            model(features, lengths)
        grp_s = time.perf_counter() - grp_started

    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    print(json.dumps({
        "worker_id": args.worker_id,
        "step_s": step_s,
        "table_steps": table_steps,
        "table_steps_per_s": table_steps / max(step_s, 1e-9),
        "grp_s": grp_s,
        "grp_calls": args.grp_iterations,
        "grp_calls_per_s": args.grp_iterations / max(grp_s, 1e-9),
        "threads": _thread_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "voluntary_context_switches": usage_end.ru_nvcsw - usage_start.ru_nvcsw,
        "involuntary_context_switches": usage_end.ru_nivcsw - usage_start.ru_nivcsw,
        "array_probe": float(np.asarray(features).sum()),
    }, sort_keys=True), flush=True)
    return 0


def _run_combination(
    args: argparse.Namespace,
    *,
    name: str,
    step_threads: int,
    cpu_threads: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    processes: list[subprocess.Popen[str]] = []
    for worker_id in range(args.workers):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--worker-id", str(worker_id),
            "--workers", str(args.workers),
            "--envs", str(args.envs),
            "--ticks", str(args.ticks),
            "--grp-iterations", str(args.grp_iterations),
            "--step-threads", str(step_threads),
            "--cpu-threads", str(cpu_threads),
            "--seed", str(args.seed),
            "--game-mode", args.game_mode,
        ]
        processes.append(subprocess.Popen(
            command,
            cwd=args.repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ))
    for process in processes:
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"CPU benchmark child failed with {process.returncode}: {stderr}"
            )
        rows.append(json.loads(stdout.strip().splitlines()[-1]))
    wall_s = time.perf_counter() - started
    return {
        "name": name,
        "step_threads": step_threads,
        "cpu_threads": cpu_threads,
        "wall_s": wall_s,
        "step_max_s": max(float(row["step_s"]) for row in rows),
        "step_mean_s": sum(float(row["step_s"]) for row in rows) / len(rows),
        "table_steps_per_s": sum(int(row["table_steps"]) for row in rows)
        / max(max(float(row["step_s"]) for row in rows), 1e-9),
        "grp_max_s": max(float(row["grp_s"]) for row in rows),
        "grp_mean_s": sum(float(row["grp_s"]) for row in rows) / len(rows),
        "threads_max": max(int(row["threads"]) for row in rows),
        "voluntary_context_switches": sum(
            int(row["voluntary_context_switches"]) for row in rows
        ),
        "involuntary_context_switches": sum(
            int(row["involuntary_context_switches"]) for row in rows
        ),
        "workers": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--grp-iterations", type=int, default=1000)
    parser.add_argument("--step-threads", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--game-mode", default="4p-red-half")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[4]))
    parser.add_argument("--output")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.child:
        return _child(args)
    combinations = (
        ("A-current", 4, 0),
        ("B-step2", 2, 0),
        ("C-limit1", 4, 1),
        ("D-limit1-step2", 2, 1),
    )
    result = {
        "schema_version": 1,
        "command": " ".join(sys.argv),
        "rounds": [
            _run_combination(
                args, name=name, step_threads=step_threads, cpu_threads=cpu_threads,
            )
            for _repeat in range(args.repeats)
            for name, step_threads, cpu_threads in combinations
        ],
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
