"""同一批互不相交的半庄上,对多个候选模型执行 1v3 对比评测。

所有候选共享同一个 seed_base(相同的牌山与候选座位轮转),对手固定为
``model_b``;每个候选独立跑 ``--processes`` 个分片,分片种子区间互不相交且
逐候选完全一致,保证候选之间除了策略本身外没有任何随机差异。进程数与每进程
半庄数由 CLI 指定,不触碰宪法固定的训练例行 1v3 机制常量(见
``mechanism.py`` 单一来源)。

用法示例:
    python -m riichi_ppo_v1.evaluation.run_1v3_compare \\
        --model-b <对手-checkpoint> \\
        --candidate <候选-a-checkpoint> --label candidate_a \\
        --candidate <候选-b-checkpoint> --label candidate_b \\
        --hanchans 4000 --processes 10 --seed-base <种子基数> \\
        --devices 0 1 --output-dir audit/reports/<版本号>/eval/compare_1v3_4000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .head_to_head_1v3_shards import (
    merge_1v3_shards,
    validate_non_overlapping_seed_ranges,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-b", required=True, help="固定对手(base)模型路径")
    parser.add_argument("--candidate", action="append", dest="candidates", required=True,
                        help="候选模型路径,可多次传入")
    parser.add_argument("--label", action="append", dest="labels", required=True,
                        help="与 --candidate 一一对应的输出标签,须唯一")
    parser.add_argument("--hanchans", type=int, default=4000,
                        help="每个候选评测的半庄总数(默认 4000,需能被 processes 整除)")
    parser.add_argument("--processes", type=int, default=10,
                        help="每个候选的分片进程数(默认 10,需能被设备数整除)")
    parser.add_argument("--parallel-hanchans", type=int, default=None,
                        help="单进程内并行推进的半庄数(默认等于每进程半庄数)")
    parser.add_argument("--seed-base", type=int, required=True,
                        help="所有候选共享的种子基数(同一牌山与座位轮转)")
    parser.add_argument("--devices", nargs="+", default=("2", "3"),
                        help="评测使用的 CUDA_DEVICE 列表(默认 CUDA 2,3 → 物理 GPU 3,4)")
    parser.add_argument("--device", default="cuda", help="传给评测子进程的 --device")
    parser.add_argument("--output-dir", required=True, help="汇总与分片输出目录")
    return parser


def _run_candidate(
    candidate: str,
    label: str,
    model_b: str,
    *,
    total: int,
    processes: int,
    parallel_hanchans: int,
    seed_base: int,
    devices: tuple[str, ...],
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    work_dir = output_dir / "shards" / label
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{label}.json"
    if summary_path.is_file():
        print(f"[{label}] 已存在,复用 {summary_path}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    per_process = total // processes
    per_card = processes // len(devices)
    commands: list[tuple[int, Path, list[str], dict[str, str]]] = []
    for shard in range(processes):
        shard_output = work_dir / f"shard{shard:02d}.json"
        shard_seed = seed_base + shard * per_process
        card_device = str(devices[shard // per_card])
        environment = dict(os.environ)
        environment["CUDA_DEVICE"] = card_device
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        command = [
            sys.executable,
            "-m",
            "riichi_ppo_v1.evaluation.head_to_head_1v3",
            "--model-a", str(Path(candidate).resolve()),
            "--model-b", str(Path(model_b).resolve()),
            "--hanchans", str(per_process),
            "--parallel-hanchans", str(parallel_hanchans),
            "--seed-base", str(shard_seed),
            "--device", device,
            "--output", str(shard_output),
        ]
        commands.append((shard, shard_output, command, environment))
    processes_started = [
        subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _shard, _shard_output, command, environment in commands
    ]
    failures: list[tuple[int, int, str]] = []
    for (shard, _shard_output, _command, _environment), process in zip(
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
        raise RuntimeError(
            f"[{label}] 1v3 对比评测失败 {len(failures)}/{len(processes_started)} "
            f"个分片;首个错误: {failures[0][2]}"
        )
    shards = [
        json.loads(shard_output.read_text(encoding="utf-8"))
        for _shard, shard_output, _command, _environment in commands
    ]
    validate_non_overlapping_seed_ranges(shards)
    summary = merge_1v3_shards(shards, seed_base=seed_base)
    summary["candidate"] = label
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return summary


def main() -> None:
    args = _parser().parse_args()
    if len(args.candidates) != len(args.labels):
        raise ValueError("--candidate 与 --label 必须一一对应")
    if len(set(args.labels)) != len(args.labels):
        raise ValueError("--label 必须唯一")
    total = int(args.hanchans)
    processes = int(args.processes)
    if total <= 0 or total % processes != 0:
        raise ValueError("hanchans 必须为正且能被 processes 整除")
    devices = tuple(str(device) for device in args.devices)
    if len(devices) < 1 or processes % len(devices) != 0:
        raise ValueError("processes 必须能被设备数整除")
    parallel = int(
        args.parallel_hanchans
        if args.parallel_hanchans is not None
        else total // processes
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate, label in zip(args.candidates, args.labels, strict=True):
        summary = _run_candidate(
            candidate,
            label,
            args.model_b,
            total=total,
            processes=processes,
            parallel_hanchans=parallel,
            seed_base=int(args.seed_base),
            devices=devices,
            device=args.device,
            output_dir=output_dir,
        )
        model_a = summary["model_a"]
        print(
            f"1v3_compare {label}: first={model_a['first_place_rate']:.4f} "
            f"top2={model_a['top2_rate']:.4f} "
            f"rank={model_a['mean_rank']:.3f} "
            f"diff={model_a['point_diff_mean']:.2f} "
            f"ci95={model_a['point_diff_bootstrap_ci95']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
