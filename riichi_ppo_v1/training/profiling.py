"""Low-overhead stage timing and GPU telemetry for PPO bottleneck analysis."""

from __future__ import annotations

import ctypes
import os
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


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


class _NvmlUtilization(ctypes.Structure):
    """nvmlUtilization_t:GPU 与显存利用率百分比。"""

    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


class _NvmlMemory(ctypes.Structure):
    """nvmlMemory_t:显存总量、空闲量与已用量(字节)。"""

    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


# NVML 枚举常量:温度传感器与时钟域类型。
_NVML_TEMPERATURE_GPU = 0
_NVML_CLOCK_SM = 1


class GpuSampler:
    """Daemon NVML 采样器;遥测不可用时不会中断训练。

    使用 NVML 而非每轮 fork ``nvidia-smi``,避免后台子进程与 OpenBLAS/OpenMP
    并行计算竞争导致死锁。
    """

    FIELDS = (
        "utilization.gpu", "utilization.memory", "memory.used", "memory.total",
        "power.draw", "temperature.gpu", "clocks.sm",
    )

    def __init__(self, enabled: bool, interval_s: float = 0.25) -> None:
        self.enabled = bool(enabled)
        self.interval_s = max(0.05, float(interval_s))
        self.samples: list[dict[str, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml_lib: ctypes.CDLL | None = None
        self._nvml_device: ctypes.c_void_p | None = None
        self._nvml_unavailable = False
        visible = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_DEVICE") or "0"
        self.device = visible.split(",", 1)[0].strip()
        try:
            self._device_index = int(self.device)
        except ValueError:
            # 非数字设备标识(如 UUID)无法直接映射到 NVML 索引,按不可用处理。
            self._device_index = -1

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="riichi-ppo-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 3 + 1.0)
        if (
            self._thread is not None
            and self._thread.is_alive()
        ):
            return
        if self._nvml_device is not None and self._nvml_lib is not None:
            self._nvml_lib.nvmlShutdown()
            self._nvml_device = None

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

    def _ensure_nvml(self) -> bool:
        """惰性初始化 NVML 并缓存设备句柄;失败后不再反复尝试。"""
        if self._nvml_device is not None:
            return True
        if self._nvml_unavailable or self._device_index < 0:
            return False
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            lib.nvmlInit_v2.argtypes = []
            lib.nvmlInit_v2.restype = ctypes.c_int
            if lib.nvmlInit_v2() != 0:
                self._nvml_unavailable = True
                return False
            lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            lib.nvmlShutdown.argtypes = []
            lib.nvmlShutdown.restype = ctypes.c_int
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(
                self._device_index, ctypes.byref(handle)
            ) != 0:
                lib.nvmlShutdown()
                self._nvml_unavailable = True
                return False

            lib.nvmlDeviceGetUtilizationRates.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_NvmlUtilization),
            ]
            lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
            lib.nvmlDeviceGetMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_NvmlMemory),
            ]
            lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
            lib.nvmlDeviceGetPowerUsage.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
            ]
            lib.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
            lib.nvmlDeviceGetTemperature.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_uint),
            ]
            lib.nvmlDeviceGetTemperature.restype = ctypes.c_int
            lib.nvmlDeviceGetClockInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_uint),
            ]
            lib.nvmlDeviceGetClockInfo.restype = ctypes.c_int

            self._nvml_lib = lib
            self._nvml_device = handle
            return True
        except (AttributeError, OSError, ctypes.ArgumentError):
            self._nvml_unavailable = True
            return False

    def _sample_once(self) -> dict[str, float] | None:
        if not self._ensure_nvml():
            return None
        lib = self._nvml_lib
        device = self._nvml_device
        if lib is None or device is None:
            return None
        result: dict[str, float] = {}
        try:
            utilization = _NvmlUtilization()
            if lib.nvmlDeviceGetUtilizationRates(device, ctypes.byref(utilization)) == 0:
                result["utilization.gpu"] = float(utilization.gpu)
                result["utilization.memory"] = float(utilization.memory)
            memory = _NvmlMemory()
            if lib.nvmlDeviceGetMemoryInfo(device, ctypes.byref(memory)) == 0:
                # NVML 返回字节,与原有 nvidia-smi 的 MiB 口径保持一致。
                result["memory.used"] = float(memory.used) / (1024.0 * 1024.0)
                result["memory.total"] = float(memory.total) / (1024.0 * 1024.0)
            power = ctypes.c_uint()
            if lib.nvmlDeviceGetPowerUsage(device, ctypes.byref(power)) == 0:
                # NVML 返回毫瓦,统一为瓦特。
                result["power.draw"] = float(power.value) / 1000.0
            temperature = ctypes.c_uint()
            if lib.nvmlDeviceGetTemperature(
                device, _NVML_TEMPERATURE_GPU, ctypes.byref(temperature)
            ) == 0:
                result["temperature.gpu"] = float(temperature.value)
            clock = ctypes.c_uint()
            if lib.nvmlDeviceGetClockInfo(
                device, _NVML_CLOCK_SM, ctypes.byref(clock)
            ) == 0:
                result["clocks.sm"] = float(clock.value)
        except (OSError, ctypes.ArgumentError):
            return None
        return result or None


def append_jsonl(path: str | Path, row: dict[str, float | int]) -> None:
    import json

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")
