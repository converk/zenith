"""只读统计既定编码选择在 V18 固定 Snapshot 下的序列长度。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..model.encoding_protocol import SNAPSHOT_FIELD_COUNT


def _offset_lengths(offsets: np.ndarray) -> np.ndarray:
    values = np.asarray(offsets, dtype=np.int64)
    if values.ndim != 1 or len(values) < 2 or values[0] != 0:
        raise RuntimeError("encoded shard contains malformed offsets")
    lengths = np.diff(values)
    if np.any(lengths < 0):
        raise RuntimeError("encoded shard offsets are not monotonic")
    return lengths


def calculate(dataset: Path, split: str) -> dict[str, Any]:
    """读取现有选择的 offset,不重放、不写数据集。"""
    paths = sorted((dataset / split).glob(f"{split}-*.npz"))
    if not paths:
        raise FileNotFoundError(f"no encoded shards found for {split}: {dataset}")
    count = 0
    history_sum = query_sum = 0
    minimum: int | None = None
    maximum = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            history = _offset_lengths(data["history_offsets"])
            query = _offset_lengths(data["query_offsets"])
        if history.shape != query.shape:
            raise RuntimeError(f"history/query row count differs in {path}")
        totals = history + SNAPSHOT_FIELD_COUNT + query
        count += len(totals)
        history_sum += int(history.sum())
        query_sum += int(query.sum())
        if len(totals):
            local_min = int(totals.min())
            minimum = local_min if minimum is None else min(minimum, local_min)
            maximum = max(maximum, int(totals.max()))
    if count == 0:
        raise RuntimeError(f"encoded split contains no decisions: {dataset / split}")
    snapshot_sum = count * SNAPSHOT_FIELD_COUNT
    return {
        "split": split,
        "shards": len(paths),
        "decisions": count,
        "history_mean": history_sum / count,
        "snapshot_mean": snapshot_sum / count,
        "query_mean": query_sum / count,
        "total_mean": (history_sum + snapshot_sum + query_sum) / count,
        "total_min": minimum,
        "total_max": maximum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    args = parser.parse_args()
    print(json.dumps(calculate(args.dataset, args.split), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
