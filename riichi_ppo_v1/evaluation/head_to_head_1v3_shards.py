"""PPO checkpoint 的同步分片 1v3 评测。

每个 update checkpoint 在并行子进程中以互不相交的连续种子区间对阵冻结的
V13 SFT 基线。训练循环在继续下一个 update 前阻塞等待本函数,因此单个分片失败
会中止训练,而不是静默产出残缺曲线。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .mechanism import (
    DEFAULT_1V3_HANCHANS_PER_PROCESS,
    DEFAULT_1V3_INTERVAL_UPDATES,
    REQUIRED_1V3_PROCESSES,
    progress_md_path,
)


def shard_summary_path(output_dir: str | Path, update: int) -> Path:
    return Path(output_dir) / f"vs_sft_u{int(update):03d}.json"


def shard_path(output_dir: str | Path, update: int, shard: int) -> Path:
    return Path(output_dir) / "shards" / (
        f"vs_sft_u{int(update):03d}_shard{int(shard):02d}.json"
    )


def validate_non_overlapping_seed_ranges(shards: list[dict[str, Any]]) -> None:
    """Require every shard's contiguous per-environment seed range to be disjoint."""
    ranges = sorted(
        (
            int(shard["seed_base"]),
            int(shard["seed_base"]) + int(shard["hanchan_count"]),
        )
        for shard in shards
    )
    for previous, current in zip(ranges, ranges[1:], strict=False):
        if current[0] < previous[1]:
            raise RuntimeError(
                "1v3 shard seed ranges overlap: "
                f"[{previous[0]}, {previous[1]}) and [{current[0]}, {current[1]})"
            )


def validate_1v3_shard_plan(
    shards: list[dict[str, Any]],
    *,
    seed_base: int,
    hanchans_per_process: int,
) -> None:
    """Enforce the project-wide 10-process, disjoint-seed 1v3 protocol."""
    if len(shards) != REQUIRED_1V3_PROCESSES:
        raise RuntimeError(
            f"1v3 evaluation requires exactly {REQUIRED_1V3_PROCESSES} shards; "
            f"got {len(shards)}"
        )
    expected_bases = [
        int(seed_base) + shard * int(hanchans_per_process)
        for shard in range(REQUIRED_1V3_PROCESSES)
    ]
    actual = sorted(
        (int(shard["seed_base"]), int(shard["hanchan_count"]))
        for shard in shards
    )
    expected = [(base, int(hanchans_per_process)) for base in expected_bases]
    if actual != expected:
        raise RuntimeError(
            "1v3 shard seed plan differs from the required disjoint allocation: "
            f"expected={expected} actual={actual}"
        )
    validate_non_overlapping_seed_ranges(shards)


def pooled_bootstrap_ci(
    samples: np.ndarray,
    seed_base: int,
    *,
    n_boot: int = 2000,
) -> list[float]:
    """Pooled-resample 95% CI over every per-hanchan point difference."""
    deltas = np.asarray(samples, dtype=np.float64).reshape(-1)
    if deltas.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(int(seed_base))
    means = np.asarray([
        float(np.mean(rng.choice(deltas, size=deltas.size, replace=True)))
        for _ in range(int(n_boot))
    ], dtype=np.float64)
    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def _weighted_kyoku_metrics(shards: list[dict[str, Any]]) -> dict[str, float]:
    """Merge model_a kyoku metrics weighted by each shard's kyoku count."""
    rows = [shard["model_a"]["kyoku_metrics"] for shard in shards]
    names = {name for row in rows for name in row}
    merged: dict[str, float] = {}
    for name in names:
        values = [
            (float(row[name]), float(row["kyoku_count"]))
            for row in rows
            if name in row
        ]
        if name == "kyoku_count" or name.endswith("_count"):
            merged[name] = float(sum(weight for _value, weight in values))
            continue
        total = sum(weight for _value, weight in values)
        merged[name] = (
            float(sum(value * weight for value, weight in values) / total)
            if total else float(np.mean([value for value, _weight in values]))
        )
    return merged


def merge_1v3_shards(
    shards: list[dict[str, Any]],
    *,
    seed_base: int,
    update: int | None = None,
) -> dict[str, Any]:
    """Merge 10x160 synthetic or real shards into one 1600-hanchan summary."""
    if not shards:
        raise ValueError("cannot merge an empty shard list")
    validate_non_overlapping_seed_ranges(shards)
    total = sum(int(shard["hanchan_count"]) for shard in shards)
    point_diffs = np.concatenate([
        np.asarray(shard["model_a"]["point_diff_samples"], dtype=np.float64)
        for shard in shards
    ])
    if point_diffs.size != total:
        raise RuntimeError(
            f"shard point-diff samples total {point_diffs.size} != hanchans {total}"
        )
    first_places = sum(int(shard["model_a"]["first_place_count"]) for shard in shards)
    top2 = sum(int(shard["model_a"]["top2_count"]) for shard in shards)
    fourths = sum(int(shard["model_a"]["fourth_place_count"]) for shard in shards)
    rank_sum = sum(
        float(shard["model_a"]["mean_rank"]) * int(shard["hanchan_count"])
        for shard in shards
    )
    elapsed = max(float(shard["elapsed_s"]) for shard in shards)
    first = shards[0]
    summary: dict[str, Any] = {
        "protocol_version": 1,
        "format": "1v3_sharded",
        "hanchan_count": total,
        "processes": len(shards),
        "seed_base": int(seed_base),
        "update": int(update) if update is not None else None,
        "model_a": {
            "checkpoint": first["model_a"]["checkpoint"],
            "first_place_count": first_places,
            "first_place_rate": first_places / total,
            "top2_count": top2,
            "top2_rate": top2 / total,
            "fourth_place_count": fourths,
            "fourth_place_rate": fourths / total,
            "mean_rank": rank_sum / total,
            "point_diff_mean": float(point_diffs.mean()),
            "point_diff_bootstrap_ci95": pooled_bootstrap_ci(
                point_diffs, seed_base,
            ),
            "point_diff_samples": [float(value) for value in point_diffs],
            "kyoku_metrics": _weighted_kyoku_metrics(shards),
        },
        "model_b": {
            "checkpoint": first["model_b"]["checkpoint"],
        },
        "shards": [
            {
                "index": index,
                "hanchan_count": int(shard["hanchan_count"]),
                "seed_base": int(shard["seed_base"]),
                "elapsed_s": float(shard["elapsed_s"]),
            }
            for index, shard in enumerate(shards)
        ],
        "elapsed_s": elapsed,
        "hanchan_per_s": total / max(elapsed, 1e-9),
    }
    return summary


def _record_progress_failure(
    output_dir: str | Path,
    update: int,
    failures: list[tuple[int, int, str]],
) -> None:
    progress = progress_md_path(output_dir).resolve()
    if not progress.is_file():
        return
    lines = [
        "",
        "## 评测失败记录",
        "",
        f"- update={update}：1v3 shard 子进程失败，训练已中止；"
        "已尝试路径与证据见下方失败详情。",
    ]
    for shard, returncode, detail in failures:
        lines.append(
            f"  - shard {shard}: returncode={returncode} {detail}"
        )
    with progress.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def run_sharded_1v3(
    checkpoint: str | Path,
    model_b: str | Path,
    *,
    update: int,
    processes: int = REQUIRED_1V3_PROCESSES,
    hanchans_per_process: int = DEFAULT_1V3_HANCHANS_PER_PROCESS,
    parallel_hanchans: int = DEFAULT_1V3_HANCHANS_PER_PROCESS,
    devices: tuple[str, ...] = ("0", "1"),
    seed_base: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run every shard as one subprocess, block until all finish, then merge."""
    output = Path(output_dir)
    shards_dir = output / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    summary_path = shard_summary_path(output, update)
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if int(processes) != REQUIRED_1V3_PROCESSES:
        raise ValueError(
            f"all 1v3 evaluations require exactly {REQUIRED_1V3_PROCESSES} processes"
        )
    if hanchans_per_process <= 0:
        raise ValueError("hanchans_per_process must be positive")
    if len(devices) < 2 or processes % len(devices) != 0:
        raise ValueError("processes must be divisible by the device count")
    commands: list[tuple[int, str, list[str], dict[str, str]]] = []
    for shard in range(int(processes)):
        shard_output = shard_path(output, update, shard)
        shard_seed = int(seed_base) + shard * int(hanchans_per_process)
        device = str(devices[shard // (int(processes) // len(devices))])
        environment = dict(os.environ)
        environment["CUDA_DEVICE"] = device
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        command = [
            sys.executable,
            "-m",
            "riichi_ppo_v1.evaluation.head_to_head_1v3",
            "--model-a", str(checkpoint),
            "--model-b", str(model_b),
            "--hanchans", str(int(hanchans_per_process)),
            "--parallel-hanchans", str(int(parallel_hanchans)),
            "--seed-base", str(shard_seed),
            "--device", "cuda",
            "--output", str(shard_output),
        ]
        commands.append((shard, str(shard_output), command, environment))

    processes_started: list[subprocess.Popen[str]] = []
    for _shard, _shard_output, command, environment in commands:
        processes_started.append(subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ))

    failures: list[tuple[int, int, str]] = []
    for (shard, shard_output, _command, _environment), process in zip(
        commands, processes_started, strict=True,
    ):
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            detail = stderr.strip().splitlines()
            failures.append((
                shard,
                int(process.returncode),
                ("; ".join(detail[-5:]) if detail else stdout.strip()[-500:]),
            ))
    if failures:
        _record_progress_failure(output, update, failures)
        raise RuntimeError(
            "1v3 sharded evaluation failed for update "
            f"{update}: {len(failures)}/{len(processes_started)} shards "
            f"failed ({failures[0][1]}); see PROGRESS.md"
        )

    shards = []
    for shard, shard_output, _command, _environment in commands:
        with open(shard_output, encoding="utf-8") as file:
            shards.append(json.load(file))
    validate_1v3_shard_plan(
        shards,
        seed_base=int(seed_base),
        hanchans_per_process=int(hanchans_per_process),
    )
    summary = merge_1v3_shards(shards, seed_base=seed_base, update=update)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return summary
