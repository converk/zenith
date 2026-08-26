"""只读统计 V18 当前局面编码选择下的序列长度与 segment 贡献。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..model.encoding_protocol import (
    CATEGORY_SCHEMAS,
    SEGMENT_ACTIONS,
    SEGMENT_ANALYSIS,
    SEGMENT_CRITIC_FUTURE,
    SEGMENT_CRITIC_PRIVATE,
    SEGMENT_SHARED,
    is_separator_kind,
)

_SEGMENT_NAMES = {
    SEGMENT_SHARED: "shared",
    SEGMENT_ANALYSIS: "analysis",
    SEGMENT_ACTIONS: "actions",
    SEGMENT_CRITIC_PRIVATE: "critic_private",
    SEGMENT_CRITIC_FUTURE: "critic_future",
}


def _offset_lengths(offsets: np.ndarray) -> np.ndarray:
    values = np.asarray(offsets, dtype=np.int64)
    if values.ndim != 1 or len(values) < 2 or values[0] != 0:
        raise RuntimeError("encoded shard contains malformed offsets")
    lengths = np.diff(values)
    if np.any(lengths < 0):
        raise RuntimeError("encoded shard offsets are not monotonic")
    return lengths


def calculate(dataset: Path, split: str) -> dict[str, Any]:
    """读取现有选择的 offset 与行内容,不重放、不写数据集。"""
    paths = sorted((dataset / split).glob(f"{split}-*.npz"))
    if not paths:
        raise FileNotFoundError(f"no encoded shards found for {split}: {dataset}")
    count = 0
    actor_sum = 0
    minimum: int | None = None
    maximum = 0
    segment_sums = {name: 0 for name in _SEGMENT_NAMES.values()}
    lengths_all: list[int] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            actor = _offset_lengths(data["actor_offsets"])
            factors = data["actor_factors"]
        if factors.shape[0] not in (len(actor), int(np.asarray(actor).sum())):
            raise RuntimeError(f"actor_offsets disagree with factors rows in {path}")
        for row in range(len(actor)):
            length = int(actor[row])
            rows = factors[int(np.asarray(actor[:row]).sum()):int(np.asarray(actor[:row]).sum()) + length]
            count += 1
            actor_sum += length
            lengths_all.append(integer := length)
            if minimum is None or integer < minimum:
                minimum = integer
            if integer > maximum:
                maximum = integer
            for segment, name in _SEGMENT_NAMES.items():
                segment_sums[name] += int(np.count_nonzero(rows[:, 0].astype(int) == segment))
    if count == 0:
        raise RuntimeError(f"encoded split contains no decisions: {dataset / split}")
    lengths_all = np.asarray(lengths_all, dtype=np.int64)
    return {
        "split": split,
        "shards": len(paths),
        "decisions": count,
        "actor_mean": actor_sum / count,
        "actor_p50": float(np.percentile(lengths_all, 50)),
        "actor_p95": float(np.percentile(lengths_all, 95)),
        "actor_p99": float(np.percentile(lengths_all, 99)),
        "actor_min": minimum,
        "actor_max": maximum,
        "segment_mean": {
            name: segment_sums[name] / count for name in _SEGMENT_NAMES.values()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    args = parser.parse_args()
    print(json.dumps(calculate(args.dataset, args.split), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
