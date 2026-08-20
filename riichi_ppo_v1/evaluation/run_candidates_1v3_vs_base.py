"""对多个 V16 候选 checkpoint 逐一执行 2400 半庄分片并行 1v3 评测,对手为
冻结的 V14 base(checkpoint_00510),所有候选用**同一种子区间与同一候选座位
轮转**,保证六模型对比公平;并额外执行 V14 base 3v1 baseline(自己打自己)。

可对比性:每个候选以同一个 seed_base 分片(12 进程 × 200),同一片内的种子区间
逐候选一致,因此同一片/同 env 序号的牌山与候选座位完全一致,六模型之间除了
策略差异外没有任何随机因素不同。

用法:
    python -m riichi_ppo_v1.evaluation.run_candidates_1v3_vs_base \\
        --model-b checkpoints/train_riichi_v14/checkpoint_00510.pt \\
        --candidate checkpoints/.../ppo/checkpoint_00090.pt \\
        --candidate checkpoints/.../ppo/checkpoint_00120.pt \\
        --seed-base 2026082000 \\
        --output audit/reports/v16/eval/vs_v14_base_same_seeds.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .head_to_head_1v3_shards import REQUIRED_1V3_PROCESSES, run_sharded_1v3
from .mechanism import DEFAULT_1V3_HANCHANS_PER_PROCESS

# 单进程内并行推进的半庄数(与训练评测一致的节奏)。
PARALLEL_HANCHANS = DEFAULT_1V3_HANCHANS_PER_PROCESS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-b", required=True, help="冻结的对手(base)模型")
    parser.add_argument("--candidate", action="append", dest="candidates", required=True,
                        help="候选模型路径,可多次传入")
    parser.add_argument("--hanchans", type=int, default=2400,
                        help="每个候选的半庄总数,需能被 12 整除")
    parser.add_argument("--seed-base", type=int, required=True,
                        help="所有候选共享的种子基数(同一牌山与座位轮转)")
    parser.add_argument("--devices", nargs="+", default=("0", "3"),
                        help="评测使用的 CUDA_DEVICE 列表(约定映射到物理卡)")
    parser.add_argument("--work-root", required=False,
                        help="分片工作目录(每候选一个子目录);默认输出目录下 shards_work")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-base-3v1", action="store_true",
                        help="额外运行 V14 base 自己打自己的 3v1 baseline")
    return parser


def _run_one(
    model_a: str,
    model_b: str,
    *,
    total: int,
    seed_base: int,
    devices: tuple[str, ...],
    work_dir: Path,
    key: str,
) -> dict:
    """运行单个 1v3 评测:12 分片 × total/12,种子区间由 seed_base 决定。"""
    per_process = total // REQUIRED_1V3_PROCESSES
    work_dir.mkdir(parents=True, exist_ok=True)
    # 每个候选/实验都使用相同的 seed_base,保证可对比性;update 仅用于区分
    # 分片输出文件名,不改变种子。
    summary = run_sharded_1v3(
        checkpoint=model_a,
        model_b=model_b,
        update=hash(key) % 100000,
        processes=REQUIRED_1V3_PROCESSES,
        hanchans_per_process=per_process,
        parallel_hanchans=min(PARALLEL_HANCHANS, per_process),
        devices=devices,
        seed_base=seed_base,
        output_dir=work_dir,
    )
    return summary


def main() -> None:
    args = _parser().parse_args()
    total = int(args.hanchans)
    if total % REQUIRED_1V3_PROCESSES != 0:
        raise SystemExit(f"--hanchans 必须能被 {REQUIRED_1V3_PROCESSES} 整除")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root) if args.work_root else output.parent / "shards_work"
    work_root.mkdir(parents=True, exist_ok=True)

    experiments: dict = {}
    for index, model_a in enumerate(args.candidates):
        key = f"experiment_{index + 1:02d}"
        print(f"\n===== 实验 {key}: {model_a} vs {args.model_b} "
              f"(同一种子区间 [{args.seed_base}, {args.seed_base} + {total})) =====",
              flush=True)
        summary = _run_one(
            model_a, args.model_b, total=total, seed_base=args.seed_base,
            devices=tuple(args.devices), work_dir=work_root / key, key=key,
        )
        experiments[key] = {
            "model_a": model_a,
            "model_b": args.model_b,
            "summary": summary,
        }
        print(f"===== 完成 {model_a}: first_rate={summary['model_a']['first_place_rate']:.3f} "
              f"point_diff={summary['model_a']['point_diff_mean']:+.1f} "
              f"elapsed={summary['elapsed_s']:.0f}s =====", flush=True)

    if args.include_base_3v1:
        key = "experiment_base_3v1"
        print(f"\n===== baseline: {args.model_b} 自己打自己(3v1) =====", flush=True)
        summary = _run_one(
            args.model_b, args.model_b, total=total, seed_base=args.seed_base,
            devices=tuple(args.devices), work_dir=work_root / key, key=key,
        )
        experiments[key] = {
            "model_a": args.model_b,
            "model_b": args.model_b,
            "summary": summary,
        }
        print(f"===== 完成 baseline: first_rate={summary['model_a']['first_place_rate']:.3f} "
              f"point_diff={summary['model_a']['point_diff_mean']:+.1f} "
              f"elapsed={summary['elapsed_s']:.0f}s =====", flush=True)

    results: dict = {
        "format": "1v3_vs_v14_base_same_seeds_sharded",
        "model_b": args.model_b,
        "hanchan_count": total,
        "processes": REQUIRED_1V3_PROCESSES,
        "seed_base": args.seed_base,
        "identical_seed_and_seats_across_experiments": True,
        "candidates": list(args.candidates),
        "experiments": experiments,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"\n全部完成,汇总写入 {output}", flush=True)


if __name__ == "__main__":
    main()
