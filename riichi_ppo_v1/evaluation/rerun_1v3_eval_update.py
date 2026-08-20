"""独立重跑某个 update 的固定 6000 半庄 1v3 评测。

用途:训练循环中偶发分片失败(如 CUDA OOM)导致某个 update 的评测缺失时,
不改动训练循环即可补齐该 update 的评测。参数全部取自版本配置,与
``train.py`` 的 ``run_1v3_evaluation`` 使用同一套机制常量与种子/设备约定,
不硬编码任何版本、checkpoint 或数据集。

用法:
    python -m riichi_ppo_v1.evaluation.rerun_1v3_eval_update \\
        --config riichi_ppo_v1/configs/v17_ppo_resume.yaml \\
        --update 10

若 ``--update`` 的汇总文件已存在,直接复用;否则重新运行全部 12 个分片
(已有分片文件会被同种子确定性覆盖)。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .head_to_head_1v3_shards import REQUIRED_1V3_PROCESSES, run_sharded_1v3
from .mechanism import DEFAULT_1V3_HANCHANS_PER_PROCESS


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("评测配置必须是一个 mapping")
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="自包含版本配置 YAML")
    parser.add_argument("--update", type=int, required=True, help="要补齐评测的 update 编号")
    parser.add_argument("--devices", nargs="+", default=None,
                        help="覆盖评测分片使用的 CUDA_DEVICE 列表(默认取配置)")
    parser.add_argument("--output-dir", default=None,
                        help="覆盖评测输出目录(默认取配置)")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args.config)
    update = int(args.update)
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_path = checkpoint_dir / f"checkpoint_{update:05d}.pt"
    if not checkpoint_path.is_file():
        raise RuntimeError(
            "评测所需的 checkpoint 不存在: " f"{checkpoint_path}"
        )
    output_dir = Path(args.output_dir or config["eval1v3_output_dir"])
    model_b = config.get("eval1v3_model_b")
    if not model_b:
        raise RuntimeError("配置缺少 eval1v3_model_b")
    processes = int(config.get("eval1v3_processes", REQUIRED_1V3_PROCESSES))
    if processes != REQUIRED_1V3_PROCESSES:
        raise RuntimeError(
            f"所有 1v3 评测必须恰好使用 {REQUIRED_1V3_PROCESSES} 个进程"
        )
    hanchans_per_process = int(
        config.get("eval1v3_hanchans_per_process", DEFAULT_1V3_HANCHANS_PER_PROCESS)
    )
    seed_base = int(config["eval1v3_seed_base"])
    parallel_hanchans = int(
        config.get("eval1v3_parallel_hanchans", hanchans_per_process)
    )
    devices = tuple(args.devices) if args.devices else tuple(
        str(device) for device in config.get("eval1v3_devices", ("0", "1"))
    )
    summary = run_sharded_1v3(
        checkpoint_path,
        model_b,
        update=update,
        processes=processes,
        hanchans_per_process=hanchans_per_process,
        parallel_hanchans=parallel_hanchans,
        devices=devices,
        seed_base=seed_base,
        output_dir=output_dir,
    )
    model_a = summary["model_a"]
    print(
        f"1v3_eval update={update} "
        f"first_place_rate={model_a['first_place_rate']:.4f} "
        f"top2_rate={model_a['top2_rate']:.4f} "
        f"mean_rank={model_a['mean_rank']:.3f} "
        f"point_diff_mean={model_a['point_diff_mean']:.2f} "
        f"ci95={model_a['point_diff_bootstrap_ci95']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
