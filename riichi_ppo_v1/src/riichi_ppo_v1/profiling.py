"""Low-overhead stage timing and GPU telemetry for PPO bottleneck analysis."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Iterator


@dataclass
class _Timing:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, seconds: float) -> None:
        self.count += 1
        self.total_s += seconds
        self.max_s = max(self.max_s, seconds)


class StageProfiler:
    """Accumulates count/total/mean/max duration without retaining samples."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._values: dict[str, _Timing] = defaultdict(_Timing)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started)

    def add(self, name: str, seconds: float) -> None:
        if self.enabled:
            self._values[name].add(max(0.0, float(seconds)))

    def checkpoint(self) -> dict[str, _Timing]:
        return {name: _Timing(value.count, value.total_s, value.max_s) for name, value in self._values.items()}

    def reset(self) -> None:
        self._values.clear()

    def delta(self, before: dict[str, _Timing], prefix: str = "timing") -> dict[str, float]:
        result: dict[str, float] = {}
        for name, current in self._values.items():
            old = before.get(name, _Timing())
            count = current.count - old.count
            total = current.total_s - old.total_s
            maximum = current.max_s if count and current.max_s >= old.max_s else 0.0
            base = f"{prefix}/{name}"
            result[f"{base}/count"] = float(count)
            result[f"{base}/total_s"] = total
            result[f"{base}/mean_ms"] = total * 1_000.0 / max(count, 1)
            result[f"{base}/max_ms"] = maximum * 1_000.0
        return result


class GpuSampler:
    """Daemon `nvidia-smi` sampler; unavailable telemetry never stops training."""

    FIELDS = ("utilization.gpu", "utilization.memory", "memory.used", "memory.total", "power.draw", "temperature.gpu", "clocks.sm")

    def __init__(self, enabled: bool, interval_s: float = 0.25) -> None:
        self.enabled = bool(enabled)
        self.interval_s = max(0.05, float(interval_s))
        self.samples: list[dict[str, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        visible = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_DEVICE") or "0"
        self.device = visible.split(",", 1)[0].strip()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="riichi-ppo-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 3 + 1.0)

    def checkpoint(self) -> int:
        with self._lock:
            return len(self.samples)

    def summary(self, start: int) -> dict[str, float]:
        with self._lock:
            rows = self.samples[start:]
        if not rows:
            return {"gpu/sample_count": 0.0}
        result: dict[str, float] = {"gpu/sample_count": float(len(rows))}
        for key in self.FIELDS:
            values = [row[key] for row in rows if key in row]
            if values:
                result[f"gpu/{key}/mean"] = sum(values) / len(values)
                result[f"gpu/{key}/max"] = max(values)
                result[f"gpu/{key}/min"] = min(values)
        return result

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            sample = self._sample_once()
            if sample:
                with self._lock:
                    self.samples.append(sample)
                    if len(self.samples) > 20_000:
                        del self.samples[:10_000]
            self._stop.wait(max(0.0, self.interval_s - (time.monotonic() - started)))

    def _sample_once(self) -> dict[str, float] | None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", self.device, f"--query-gpu={','.join(self.FIELDS)}", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=max(1.0, self.interval_s * 3),
            )
            values = [value.strip() for value in result.stdout.strip().splitlines()[0].split(",")]
            if len(values) != len(self.FIELDS):
                return None
            return {field: float(value) for field, value in zip(self.FIELDS, values) if value not in {"N/A", "[Not Supported]"}}
        except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
            return None


def append_jsonl(path: str | Path, row: dict[str, float | int]) -> None:
    import json

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")
