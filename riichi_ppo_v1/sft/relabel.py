"""V19 SFT 数据集「只重标信念标签」工具（2026-09-07 模糊化，不重新编码）。

旧数据集已经包含完整 V19 Actor 序列与 Rust 上帝视角**精确**信念标签
（hand [B,102]、wait [B,105]）。本工具只把这两个数组映射为模糊语义：

- hand：精确逐牌种计数 → 花色×段位 16 组 × {0,1,≥2}，[B,48]；
- wait：精确待牌位集 → 听牌+宽度桶 5 类，[B,15]；
- shanten/danger/loss 与 actor 序列、query、合法掩码等**原样复制**。

因此无需重跑 precompute / 重放牌局，成本只是读改写 61 GB 级 npz。
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..model.belief_labels import (
    HAND_GROUPS,
    HAND_LEN,
    WAIT_CLASSES,
    WAIT_LEN,
    _fuzzy_hand,
    _fuzzy_wait,
)

# 新 manifest 里显式声明的标签方案。
BELIEF_LABEL_SCHEME = "fuzzy-v1"
BELIEF_SHAPE = {
    "hand": [HAND_LEN],
    "shanten": [3],
    "wait": [WAIT_LEN],
    "danger": [102],
    "loss": [102],
}


def relabel_file(src: Path, dst: Path) -> int:
    """读取单个 npz，替换 fuzzy 标签后写回目标路径，返回样本行数。"""
    with np.load(src, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
        rows = int(arrays["actions"].shape[0])
        exact_hand = np.asarray(arrays["belief_hand"], dtype=np.uint8).reshape(-1, 3, 34)
        exact_wait = np.asarray(arrays["belief_wait"], dtype=np.uint8).reshape(-1, 3, 35)
        hand = _fuzzy_hand(exact_hand).reshape(-1, HAND_LEN)
        wait = _fuzzy_wait(exact_wait).reshape(-1, WAIT_LEN)
        if hand.shape[0] != rows or wait.shape[0] != rows:
            raise RuntimeError(f"fuzzy label row mismatch in {src}")
        arrays["belief_hand"] = hand
        arrays["belief_wait"] = wait
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, dst)
    return rows


def _relabel_worker(args: tuple[str, str]) -> tuple[str, int]:
    src, dst = args
    return src, relabel_file(Path(src), Path(dst))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="旧精确标签数据集目录")
    parser.add_argument("--output", type=Path, required=True, help="新模糊标签数据集目录")
    parser.add_argument("--splits", type=str, default="train,validation")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not (source / "manifest.json").is_file():
        raise FileNotFoundError(f"source manifest missing: {source}")
    if output.exists() and (output / "manifest.json").is_file():
        raise RuntimeError(f"output dataset already exists, use a new dir or delete first: {output}")

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["belief_shape"] = BELIEF_SHAPE
    manifest["belief_label_scheme"] = BELIEF_LABEL_SCHEME
    manifest["relabeled_from"] = str(source)
    manifest["source_manifest_sha256"] = manifest.get("source_manifest_sha256", "")

    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    tasks: list[tuple[str, str]] = []
    for split in splits:
        src_dir = source / split
        dst_dir = output / split
        if not src_dir.is_dir():
            raise FileNotFoundError(f"source split missing: {src_dir}")
        src_files = sorted(src_dir.glob(f"{split}-*.npz"))
        if not src_files:
            raise RuntimeError(f"no {split} npz files in {src_dir}")
        for src_file in src_files:
            tasks.append((str(src_file), str(dst_dir / src_file.name)))

    total_files = len(tasks)
    total_rows = 0
    print(f"[relabel] {total_files} files across {splits}", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for i, (src, rows) in enumerate(executor.map(_relabel_worker, tasks), start=1):
            total_rows += rows
            if i % args.progress_every == 0 or i == total_files:
                print(f"[relabel] {i}/{total_files} files, {total_rows} rows", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(f"[relabel] done -> {output}/manifest.json rows={total_rows}", flush=True)


if __name__ == "__main__":
    main()
