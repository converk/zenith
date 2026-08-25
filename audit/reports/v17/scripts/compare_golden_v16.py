#!/usr/bin/env python
"""逐元素比较两个 V16 golden NPZ 及其同名 JSON 边界日志。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle")
    parser.add_argument("candidate")
    args = parser.parse_args()

    oracle_path = Path(args.oracle)
    candidate_path = Path(args.candidate)
    oracle = np.load(oracle_path, allow_pickle=True)
    candidate = np.load(candidate_path, allow_pickle=True)
    if set(oracle.files) != set(candidate.files):
        raise AssertionError((oracle.files, candidate.files))
    checked = 0
    for key in oracle.files:
        left = oracle[key]
        right = candidate[key]
        if left.ndim == 0 or right.ndim == 0:
            if not np.array_equal(left, right):
                raise AssertionError((key, left, right))
            checked += int(left.size)
            continue
        if len(left) != len(right):
            raise AssertionError((key, len(left), len(right)))
        for batch, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
            if not np.array_equal(left_row, right_row):
                raise AssertionError((key, batch, np.shape(left_row), np.shape(right_row)))
            checked += int(np.asarray(left_row).size)

    with oracle_path.with_suffix(".json").open(encoding="utf-8") as file:
        oracle_json = json.load(file)
    with candidate_path.with_suffix(".json").open(encoding="utf-8") as file:
        candidate_json = json.load(file)
    if oracle_json != candidate_json:
        raise AssertionError("state-machine boundary/decode JSON differs")
    print(
        f"golden_equal arrays={len(oracle.files)} elements={checked} "
        "integer_exact=1 float_exact=1 json_equal=1"
    )


if __name__ == "__main__":
    main()
