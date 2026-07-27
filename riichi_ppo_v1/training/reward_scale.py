"""Driver-owned V7-style GAE-trace reward scale calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping


@dataclass
class RewardScaleController:
    beta: float = 0.95
    minimum: float = 0.02
    maximum: float = 2.0
    discard_weight: float = 0.25
    call_weight: float = 0.10
    stage1_end: int = 1500
    stage2_end: int = 3500
    kyoku_rms: float | None = None
    discard_rms: float | None = None
    call_rms: float | None = None

    def targets(self, update: int) -> tuple[float, float, int]:
        if int(update) <= self.stage1_end:
            return 0.35, 0.15, 1
        if int(update) <= self.stage2_end:
            return 0.20, 0.10, 2
        return 0.12, 0.05, 3

    def context(self, update: int) -> dict[str, float]:
        discard, call, stage = self.targets(update)
        return {
            "discard_weight": self.discard_weight,
            "call_weight": self.call_weight,
            "discard_target": discard,
            "call_target": call,
            "reward_stage": float(stage),
        }

    def update(self, update: int, rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
        squares = {name: 0.0 for name in ("kyoku", "discard", "call")}
        count = 0.0
        for row in rows:
            count += float(row.get("reward_scale/trace_count", 0.0))
            for name in squares:
                squares[name] += float(row.get(f"reward_scale/{name}_trace_sum_squares", 0.0))
        if count > 0:
            for name, total in squares.items():
                value = math.sqrt(total / count)
                previous = getattr(self, f"{name}_rms")
                setattr(self, f"{name}_rms", value if previous is None else self.beta * previous + (1 - self.beta) * value)
            discard_target, call_target, _stage = self.targets(update)
            if self.kyoku_rms and self.discard_rms and self.discard_rms > 1e-6:
                self.discard_weight = min(
                    self.maximum, max(self.minimum, discard_target * self.kyoku_rms / self.discard_rms),
                )
            if self.kyoku_rms and self.call_rms and self.call_rms > 1e-6:
                self.call_weight = min(
                    self.maximum, max(self.minimum, call_target * self.kyoku_rms / self.call_rms),
                )
        return self.metrics(update)

    def metrics(self, update: int) -> dict[str, float]:
        discard_target, call_target, stage = self.targets(update)
        kyoku = self.kyoku_rms or 0.0
        discard = self.discard_rms or 0.0
        call = self.call_rms or 0.0
        return {
            "reward_scale/stage": float(stage),
            "reward_scale/discard_weight": self.discard_weight,
            "reward_scale/call_weight": self.call_weight,
            "reward_scale/discard_target_ratio": discard_target,
            "reward_scale/call_target_ratio": call_target,
            "reward_scale/kyoku_trace_rms": kyoku,
            "reward_scale/discard_trace_rms_raw": discard,
            "reward_scale/call_trace_rms_raw": call,
            "reward_scale/discard_to_kyoku_ratio": self.discard_weight * discard / max(kyoku, 1e-6),
            "reward_scale/call_to_kyoku_ratio": self.call_weight * call / max(kyoku, 1e-6),
        }

    def state_dict(self) -> dict[str, float | int | None]:
        return asdict(self)

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        for name in self.state_dict():
            if name in state:
                setattr(self, name, state[name])
