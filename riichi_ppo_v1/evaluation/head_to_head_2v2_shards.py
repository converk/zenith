"""跨代 2v2 SFT 评测的分片驱动(5 对 host/partner 进程,10 进程 × 600 = 6000 半庄)。

与 1v3 分片机制同构:互不相交的连续种子区间、分片子进程独立写盘、全部成功
后合并汇总并落盘。差异在于每分片是**一对锁步进程**(见 ``head_to_head_2v2``):

- V19 host 进程与 V18 partner 进程经父进程创建的 socketpair 通讯,各自加载
  本代代码与模型;5 对进程双卡各 3 对以下均分(默认 shards 0-4 → 设备 0,
  shards 5-9 → 设备 1,共 10 个评测工作进程);
- 机制常量(2v2 机制,2026-09-08 起用):10 进程 × 每进程 600 半庄;种子基、
  设备与输出目录由 CLI 提供,禁止硬编码具体版本路径。

本模块是一次性跨代评测的入口,不属于 PPO 训练期的 1v3 机制;机制常量的任何
调整必须同步更新 AGENTS.md 的评测机制描述与 PROGRESS.md 记录。
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .head_to_head_1v3_shards import checkpoint_sha256, pooled_bootstrap_ci
from .mechanism import progress_md_path

# 2v2 机制常量(单一来源):10 进程 × 每进程 600 半庄 = 6000。
REQUIRED_2V2_PROCESSES = 10
DEFAULT_2V2_HANCHANS_PER_PROCESS = 600
TOTAL_2V2_HANCHANS = REQUIRED_2V2_PROCESSES * DEFAULT_2V2_HANCHANS_PER_PROCESS
DEFAULT_2V2_DEVICES = ("0", "1")
_SAMPLES_PER_HANCHAN = 2  # 每队每半庄 2 个座位样本
_RANK_NAMES = {
    1: "first_place",
    2: "second_place",
    3: "third_place",
    4: "fourth_place",
}


def summary_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "vs_v18_sft_2v2.json"


def shard_path(output_dir: str | Path, shard: int) -> Path:
    return Path(output_dir) / "shards" / f"vs_v18_sft_2v2_shard{int(shard):02d}.json"


def validate_2v2_shard_plan(
    shards: list[dict[str, Any]],
    *,
    seed_base: int,
    hanchans_per_process: int,
) -> None:
    """校验分片互不相交的连续种子区间与既定进程数。"""
    if len(shards) != REQUIRED_2V2_PROCESSES:
        raise RuntimeError(
            f"2v2 evaluation requires exactly {REQUIRED_2V2_PROCESSES} shards; "
            f"got {len(shards)}"
        )
    expected_bases = [
        int(seed_base) + shard * int(hanchans_per_process)
        for shard in range(REQUIRED_2V2_PROCESSES)
    ]
    actual = sorted(
        (int(shard["seed_base"]), int(shard["hanchan_count"])) for shard in shards
    )
    expected = [(base, int(hanchans_per_process)) for base in expected_bases]
    if actual != expected:
        raise RuntimeError(
            "2v2 shard seed plan differs from the required disjoint allocation: "
            f"expected={expected} actual={actual}"
        )
    # 精确分配校验通过即保证区间互不相交且连续,无需二次重叠检查。


def _weighted_metrics(rows: list[dict[str, float]], count_key: str) -> dict[str, float]:
    """按 count_key 权重合并一组平铺指标字典;计数类字段直接求和。"""
    names = {name for row in rows for name in row}
    merged: dict[str, float] = {}
    for name in names:
        values = [
            (float(row[name]), float(row.get(count_key, 0.0)))
            for row in rows
            if name in row
        ]
        if name == count_key or name.endswith("_count"):
            merged[name] = float(sum(value for value, _weight in values))
            continue
        total = sum(weight for _value, weight in values)
        merged[name] = (
            float(sum(value * weight for value, weight in values) / total)
            if total
            else float(np.mean([value for value, _weight in values]))
        )
    return merged


def _merged_seat_counts(shards: list[dict[str, Any]], key: str) -> dict[str, int]:
    merged: Counter = Counter()
    for shard in shards:
        merged.update(shard[key])
    return {str(seat): int(count) for seat, count in sorted(merged.items())}


def merge_2v2_shards(
    shards: list[dict[str, Any]],
    *,
    seed_base: int,
    hanchans_per_process: int = DEFAULT_2V2_HANCHANS_PER_PROCESS,
) -> dict[str, Any]:
    """把各分片(默认 10 × 600 = 6000 半庄)合并为单一汇总 summary。"""
    if not shards:
        raise ValueError("cannot merge an empty shard list")
    validate_2v2_shard_plan(
        shards, seed_base=seed_base, hanchans_per_process=hanchans_per_process,
    )
    total = sum(int(shard["hanchan_count"]) for shard in shards)
    sample_total = total * _SAMPLES_PER_HANCHAN

    def rank_block(policy: str) -> dict[str, Any]:
        counts = {rank: 0 for rank in range(1, 5)}
        rank_sum = 0.0
        for shard in shards:
            block = shard[f"model_{policy}"]
            for rank in range(1, 5):
                counts[rank] += int(block[f"{_RANK_NAMES[rank]}_count"])
            rank_sum += float(block["mean_rank"]) * int(block["sample_count"])
        point_diffs = np.concatenate([
            np.asarray(shard[f"model_{policy}"]["point_diff_samples"], dtype=np.float64)
            for shard in shards
        ])
        if point_diffs.size != sample_total:
            raise RuntimeError(
                f"model_{policy} point-diff samples total {point_diffs.size} "
                f"!= seat samples {sample_total}"
            )
        semantic_metrics = _weighted_metrics(
            [shard[f"model_{policy}"]["semantic_metrics"] for shard in shards],
            f"model_{policy}/match/count",
        )
        final_score_mean = semantic_metrics.get(f"model_{policy}/match/final_score_mean")
        flying_count = sum(
            float(shard[f"model_{policy}"]["flying_rate"]) * int(shard["hanchan_count"])
            for shard in shards
        )
        block: dict[str, Any] = {
            "sample_count": sample_total,
            "first_place_count": counts[1],
            "first_place_rate": counts[1] / sample_total,
            "second_place_count": counts[2],
            "second_place_rate": counts[2] / sample_total,
            "third_place_count": counts[3],
            "third_place_rate": counts[3] / sample_total,
            "fourth_place_count": counts[4],
            "fourth_place_rate": counts[4] / sample_total,
            "last_place_rate": counts[4] / sample_total,
            "top2_count": counts[1] + counts[2],
            "top2_rate": (counts[1] + counts[2]) / sample_total,
            "mean_rank": rank_sum / sample_total,
            "final_score_mean": (
                float(final_score_mean) if final_score_mean is not None else 0.0
            ),
            "flying_count": flying_count,
            "flying_rate": flying_count / total,
            "point_diff_vs_mean_others_mean": float(point_diffs.mean()),
            "point_diff_vs_mean_others_bootstrap_ci95": pooled_bootstrap_ci(
                point_diffs, seed_base,
            ),
            "point_diff_samples": [float(value) for value in point_diffs],
            "kyoku_metrics": _weighted_metrics(
                [shard[f"model_{policy}"]["kyoku_metrics"] for shard in shards],
                "kyoku_count",
            ),
            "semantic_metrics": semantic_metrics,
        }
        if policy == "a":
            block["belief_metrics"] = _weighted_metrics(
                [shard["model_a"]["belief_metrics"] for shard in shards],
                "decision_count",
            )
        # 动作分组率按半庄数加权合并;metadata 各分片一致,取首分片。
        rate_rows = [
            {**shard[f"model_{policy}"]["action_type_rates"],
             "hanchans": float(shard["hanchan_count"])}
            for shard in shards
        ]
        block["action_type_rates"] = {
            name: value
            for name, value in _weighted_metrics(rate_rows, "hanchans").items()
            if name != "hanchans"
        }
        block["metadata"] = first[f"model_{policy}"]["metadata"]
        return block

    team_deltas = np.concatenate([
        np.asarray(shard["model_a"]["team_point_diff_samples"], dtype=np.float64)
        for shard in shards
    ])
    if team_deltas.size != total:
        raise RuntimeError(
            f"team point-diff samples total {team_deltas.size} != hanchans {total}"
        )
    first = shards[0]
    elapsed = max(float(shard["elapsed_s"]) for shard in shards)
    summary: dict[str, Any] = {
        "protocol_version": 1,
        "format": "2v2_sharded",
        "hanchan_count": total,
        "processes": len(shards),
        "samples_per_hanchan": _SAMPLES_PER_HANCHAN,
        "seed_base": int(seed_base),
        "team_a_seat_rotation": first["team_a_seat_rotation"],
        "model_a_seat_counts": _merged_seat_counts(shards, "model_a_seat_counts"),
        "model_b_seat_counts": _merged_seat_counts(shards, "model_b_seat_counts"),
        "model_a": {
            "checkpoint": first["model_a"]["checkpoint"],
            **rank_block("a"),
            "team_point_diff_mean": float(team_deltas.mean()),
            "team_point_diff_bootstrap_ci95": pooled_bootstrap_ci(team_deltas, seed_base),
            "team_point_diff_samples": [float(value) for value in team_deltas],
        },
        "model_b": {
            "checkpoint": first["model_b"]["checkpoint"],
            **rank_block("b"),
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
    failures: list[tuple[int, int, str]],
) -> None:
    progress = progress_md_path(output_dir).resolve()
    if not progress.is_file():
        return
    lines = [
        "",
        "## 评测失败记录",
        "",
        "- 2v2 SFT 跨代评测:shard 子进程失败;已尝试路径与证据见下方失败详情。",
    ]
    for shard, returncode, detail in failures:
        lines.append(f"  - shard {shard}: returncode={returncode} {detail}")
    with progress.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def run_sharded_2v2(
    model_a: str | Path,
    model_b: str | Path,
    *,
    v18_root: str | Path,
    v19_runtime_dir: str | Path,
    seed_base: int,
    output_dir: str | Path,
    processes: int = REQUIRED_2V2_PROCESSES,
    hanchans_per_process: int = DEFAULT_2V2_HANCHANS_PER_PROCESS,
    parallel_hanchans: int = DEFAULT_2V2_HANCHANS_PER_PROCESS,
    devices: tuple[str, ...] = DEFAULT_2V2_DEVICES,
    shards: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """按固定分片计划启动 host/partner 进程对,阻塞至全部完成后合并。

    ``shards`` 指定本波启动的分片子集(分波占用显存时使用);未给定时启动
    全部分片。子集全部完成后,若机制要求的全部分片文件均已就绪则自动合并。
    """
    output = Path(output_dir)
    shards_dir = output / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    # partner 进程 cwd 在 V18 副本,checkpoint 路径必须先按本进程 cwd 定死为绝对路径。
    model_a = Path(model_a).resolve()
    model_b = Path(model_b).resolve()
    v18_root = Path(v18_root).resolve()
    v19_runtime_dir = Path(v19_runtime_dir).resolve()
    partner_script = v18_root / "partner_2v2.py"
    if not partner_script.is_file():
        raise RuntimeError(f"V18 partner 脚本缺失: {partner_script}")
    eval_params = {
        "seed_base": int(seed_base),
        "hanchans_per_process": int(hanchans_per_process),
        "parallel_hanchans": int(parallel_hanchans),
        "v18_root": str(v18_root),
        "v19_runtime_dir": str(v19_runtime_dir),
    }
    cache_path = summary_path(output)
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        fingerprint_ok = (
            cached.get("checkpoint_sha256_a") == checkpoint_sha256(model_a)
            and cached.get("checkpoint_sha256_b") == checkpoint_sha256(model_b)
            and cached.get("eval_params") == eval_params
        )
        if fingerprint_ok:
            return cached
    if int(processes) != REQUIRED_2V2_PROCESSES:
        raise ValueError(
            f"all 2v2 evaluations require exactly {REQUIRED_2V2_PROCESSES} processes"
        )
    if hanchans_per_process <= 0:
        raise ValueError("hanchans_per_process must be positive")
    if len(devices) < 2 or processes % len(devices) != 0:
        raise ValueError("processes must be divisible by the device count")
    processes_per_device = int(processes) // len(devices)
    selected = list(range(int(processes))) if shards is None else [int(v) for v in shards]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"invalid shard subset: {selected}")
    if any(value < 0 or value >= int(processes) for value in selected):
        raise ValueError(f"shard subset out of range: {selected}")

    host_commands: list[tuple[int, str, list[str], dict[str, str]]] = []
    partner_commands: list[tuple[int, str, list[str], dict[str, str]]] = []
    for shard in selected:
        shard_seed = int(seed_base) + shard * int(hanchans_per_process)
        shard_output = str(shard_path(output, shard))
        device = str(devices[shard // processes_per_device])
        environment = dict(os.environ)
        environment["CUDA_DEVICE"] = device
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        host_command = [
            sys.executable,
            "-m",
            "riichi_ppo_v1.evaluation._2v2_host_entry",
            "--model-a", str(model_a),
            "--hanchans", str(int(hanchans_per_process)),
            "--parallel-hanchans", str(int(parallel_hanchans)),
            "--seed-base", str(shard_seed),
            "--device", "cuda",
            "--v19-runtime-dir", str(v19_runtime_dir),
            "--output", shard_output,
        ]
        host_commands.append((shard, shard_output, host_command, dict(environment)))
        partner_command = [
            sys.executable,
            str(partner_script),
            "--model-b", str(model_b),
            "--hanchans", str(int(hanchans_per_process)),
            "--parallel-hanchans", str(int(parallel_hanchans)),
            "--seed-base", str(shard_seed),
            "--device", "cuda",
        ]
        partner_commands.append(
            (shard, shard_output, partner_command, dict(environment)),
        )

    running: list[tuple[int, subprocess.Popen[str], subprocess.Popen[str], Any]] = []
    try:
        for (
            shard, _out, host_command, host_env
        ), (_shard2, _out2, partner_command, partner_env) in zip(
            host_commands, partner_commands, strict=True,
        ):
            host_sock, partner_sock = socket.socketpair()
            # pass_fds 保持原 fd 号,分别经 CLI 注入两个子进程。
            host_command = [*host_command, "--partner-fd", str(host_sock.fileno())]
            partner_command = [
                *partner_command, "--partner-fd", str(partner_sock.fileno()),
            ]
            host_process = subprocess.Popen(
                host_command,
                env=host_env,
                cwd=Path(__file__).resolve().parent.parent.parent,  # 仓库根
                pass_fds=(host_sock.fileno(),),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            partner_process = subprocess.Popen(
                partner_command,
                env=partner_env,
                cwd=str(v18_root),
                pass_fds=(partner_sock.fileno(),),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            running.append((shard, host_process, partner_process, host_sock))
            host_sock.close()   # 子进程持有一端,父进程即刻释放自己的副本
            partner_sock.close()
    except Exception:
        for _shard, host_process, partner_process, _sock in running:
            host_process.kill()
            partner_process.kill()
        raise

    failures: list[tuple[int, int, str]] = []
    for shard, host_process, partner_process, _sock in running:
        host_stdout, host_stderr = host_process.communicate()
        partner_stdout, partner_stderr = partner_process.communicate()
        for role, process, stdout, stderr in (
            ("host", host_process, host_stdout, host_stderr),
            ("partner", partner_process, partner_stdout, partner_stderr),
        ):
            if process.returncode != 0:
                detail = stderr.strip().splitlines()
                failures.append((
                    shard,
                    int(process.returncode),
                    f"{role}: "
                    + ("; ".join(detail[-5:]) if detail else stdout.strip()[-500:]),
                ))
    if failures:
        for shard, returncode, detail in failures:
            print(f"2v2 shard {shard} failure (rc={returncode}): {detail}", flush=True)
        _record_progress_failure(output, failures)
        failed_shards = sorted({shard for shard, _rc, _detail in failures})
        raise RuntimeError(
            "2v2 sharded evaluation failed: "
            f"{len(failed_shards)}/{len(running)} shard pairs failed "
            f"({failures[0][1]}: {failures[0][2][-400:]}); see PROGRESS.md"
        )

    shard_rows = []
    for shard, _output, _command, _environment in host_commands:
        with open(shard_path(output, shard), encoding="utf-8") as file:
            shard_rows.append(json.load(file))
    all_shard_files = [
        shard_path(output, shard) for shard in range(int(processes))
    ]
    if not all(path.is_file() for path in all_shard_files):
        print(
            f"2v2 wave done for shards {selected}; "
            "remaining shards pending, merge deferred",
            flush=True,
        )
        return None
    shards = []
    for path in all_shard_files:
        with open(path, encoding="utf-8") as file:
            shards.append(json.load(file))
    validate_2v2_shard_plan(
        shards,
        seed_base=int(seed_base),
        hanchans_per_process=int(hanchans_per_process),
    )
    summary = merge_2v2_shards(
        shards, seed_base=int(seed_base),
        hanchans_per_process=int(hanchans_per_process),
    )
    summary["checkpoint_sha256_a"] = checkpoint_sha256(model_a)
    summary["checkpoint_sha256_b"] = checkpoint_sha256(model_b)
    summary["eval_params"] = eval_params
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards", default=None,
        help="本波启动的分片子集(逗号分隔);缺省为全部分片",
    )
    parser.add_argument("--model-a", required=True, help="V19 SFT checkpoint")
    parser.add_argument("--model-b", required=True, help="V18 SFT checkpoint")
    parser.add_argument("--v18-root", required=True, help="V18 独立工作副本根目录")
    parser.add_argument(
        "--v19-runtime-dir", required=True,
        help="V19 扩展新鲜构建目录(host 侧锁步一致性运行时)",
    )
    parser.add_argument(
        "--seed-base", type=int, default=None,
        help="分片种子基;缺省时随机生成并在汇总中记录",
    )
    parser.add_argument("--processes", type=int, default=REQUIRED_2V2_PROCESSES)
    parser.add_argument(
        "--hanchans-per-process", type=int, default=DEFAULT_2V2_HANCHANS_PER_PROCESS,
    )
    parser.add_argument(
        "--parallel-hanchans", type=int, default=DEFAULT_2V2_HANCHANS_PER_PROCESS,
    )
    parser.add_argument("--devices", default=",".join(DEFAULT_2V2_DEVICES))
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    seed_base = (
        args.seed_base
        if args.seed_base is not None
        else 100_000_000 + secrets.randbelow(900_000_000)
    )
    print(f"2v2 evaluation seed_base={seed_base}", flush=True)
    shard_subset = (
        tuple(int(v) for v in args.shards.split(","))
        if args.shards is not None
        else None
    )
    summary = run_sharded_2v2(
        args.model_a,
        args.model_b,
        v18_root=args.v18_root,
        v19_runtime_dir=args.v19_runtime_dir,
        seed_base=seed_base,
        output_dir=args.output_dir,
        processes=args.processes,
        hanchans_per_process=args.hanchans_per_process,
        parallel_hanchans=args.parallel_hanchans,
        devices=tuple(args.devices.split(",")),
        shards=shard_subset,
    )
    if summary is None:
        return
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
