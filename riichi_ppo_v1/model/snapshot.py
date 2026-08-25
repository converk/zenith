"""固定 29 行原子 Snapshot 的原生桥接。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import riichienv

from .encoding_protocol import (
    SNAPSHOT_FACTOR_WIDTH,
    SNAPSHOT_FIELD_COUNT,
    SNAPSHOT_FIELDS,
    SNAPSHOT_NUMERIC_WIDTH,
)


@dataclass(frozen=True)
class AtomicSnapshot:
    factors: np.ndarray
    numeric: np.ndarray

    @property
    def length(self) -> int:
        return SNAPSHOT_FIELD_COUNT


def _validate(factors: np.ndarray, numeric: np.ndarray) -> None:
    if factors.shape != (SNAPSHOT_FIELD_COUNT, SNAPSHOT_FACTOR_WIDTH):
        raise RuntimeError(
            f"snapshot_factors must be [{SNAPSHOT_FIELD_COUNT},{SNAPSHOT_FACTOR_WIDTH}]"
        )
    if numeric.shape != (SNAPSHOT_FIELD_COUNT, SNAPSHOT_NUMERIC_WIDTH):
        raise RuntimeError(
            f"snapshot_numeric must be [{SNAPSHOT_FIELD_COUNT},{SNAPSHOT_NUMERIC_WIDTH}]"
        )
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("snapshot_numeric contains non-finite values")
    for row, field in enumerate(SNAPSHOT_FIELDS):
        field_id, relative, categorical, tile = (int(value) for value in factors[row])
        if field_id != field.field_id or relative != field.relative_seat:
            raise RuntimeError(f"snapshot row {row} violates canonical field order")
        if not 0 <= categorical <= field.categorical_max:
            raise RuntimeError(f"snapshot row {row} categorical value is outside its domain")
        if not 0 <= tile <= field.tile_max:
            raise RuntimeError(f"snapshot row {row} tile value is outside its domain")
        value = float(numeric[row, 0])
        if field.numeric:
            if not -1.0 <= value <= 1.0:
                raise RuntimeError(f"snapshot row {row} numeric value is outside [-1,1]")
        elif value != 0.0:
            raise RuntimeError(f"snapshot row {row} must not carry a numeric value")


def encode_snapshot_batch(observations: list[object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """通过 Rust 从原生 Observation 批量派生 Snapshot。"""
    if not observations:
        raise ValueError("cannot encode an empty Snapshot batch")
    native = [getattr(observation, "native_observation", observation) for observation in observations]
    encoded = riichienv.prepare_atomic_snapshots(native)
    factors = np.asarray(encoded.factors, dtype=np.uint8)
    numeric = np.asarray(encoded.numeric, dtype=np.float32)
    lengths = np.asarray(encoded.lengths, dtype=np.int64)
    expected_factors = (len(observations), SNAPSHOT_FIELD_COUNT, SNAPSHOT_FACTOR_WIDTH)
    expected_numeric = (len(observations), SNAPSHOT_FIELD_COUNT, SNAPSHOT_NUMERIC_WIDTH)
    if factors.shape != expected_factors or numeric.shape != expected_numeric:
        raise RuntimeError("native Snapshot batch returned malformed shapes")
    if lengths.shape != (len(observations),) or np.any(lengths != SNAPSHOT_FIELD_COUNT):
        raise RuntimeError("native Snapshot lengths must all equal 29")
    for row in range(len(observations)):
        _validate(factors[row], numeric[row])
    return factors, numeric, lengths


def encode_snapshot_rows(observation: object) -> tuple[np.ndarray, np.ndarray]:
    """编码一个观察者的固定 Snapshot 行。"""
    factors, numeric, _lengths = encode_snapshot_batch([observation])
    return factors[0], numeric[0]


def build_atomic_snapshot(observation: object) -> AtomicSnapshot:
    factors, numeric = encode_snapshot_rows(observation)
    return AtomicSnapshot(factors=factors, numeric=numeric)
