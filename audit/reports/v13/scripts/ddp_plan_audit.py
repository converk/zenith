import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np


dataset = Path("datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16")
report = {}
for split in ("train", "validation"):
    started = time.perf_counter()
    paths = sorted((dataset / split).glob(f"{split}-*.npz"))
    rows = []
    for path in paths:
        with zipfile.ZipFile(path) as archive, archive.open("actions.npy") as member:
            version = np.lib.format.read_magic(member)
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, _fortran, _dtype = reader(member)
            rows.append(int(shape[0]))
    total = sum(rows)
    base, extra = divmod(total, 2)
    intervals = [(rank * base + min(rank, extra), rank * base + min(rank, extra) + base + int(rank < extra)) for rank in range(2)]
    boundaries = {intervals[0][1]}
    cumulative = 0
    shared = []
    for path, count in zip(paths, rows, strict=True):
        if cumulative < intervals[0][1] < cumulative + count:
            shared.append(path.name)
        cumulative += count
    rank_rows = [end - start for start, end in intervals]
    report[split] = {
        "files": len(paths), "rows": total,
        "header_scan_seconds": time.perf_counter() - started,
        "rank_intervals": intervals, "rank_rows": rank_rows,
        "shared_boundary_files": shared,
        "steps_local_batch_256": [math.ceil(value / 256) for value in rank_rows],
        "final_local_batch": [value % 256 or 256 for value in rank_rows],
    }
output = Path(__file__).resolve().parents[1] / "ddp_plan.json"
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
