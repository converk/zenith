"""Structured, secret-safe runtime metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import statistics
import time
from typing import Any


@dataclass
class SessionMetrics:
    requests: int = 0
    responses: int = 0
    model_actions: int = 0
    fallback_actions: int = 0
    withheld_actions: int = 0
    accepted: int = 0
    rejected: int = 0
    unparseable: int = 0
    stale: int = 0
    defaulted: int = 0
    bank_consumed_ms: int = 0
    inference_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        result = {
            key: value
            for key, value in asdict(self).items()
            if key != "inference_ms"
        }
        values = sorted(self.inference_ms)
        if values:
            result.update(
                {
                    "inference_count": len(values),
                    "inference_mean_ms": statistics.fmean(values),
                    "inference_p50_ms": _percentile(values, 50),
                    "inference_p95_ms": _percentile(values, 95),
                    "inference_max_ms": values[-1],
                }
            )
        else:
            result["inference_count"] = 0
        return result


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


class EventRecorder:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path).expanduser() if path else None

    def emit(self, event: str, **fields: Any) -> None:
        payload = {
            "timestamp": time.time(),
            "event": event,
            **fields,
        }
        logging.getLogger("riichi_lab_bot").info(
            "%s", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

