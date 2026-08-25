"""V18 Atomic Snapshot 的形状、N/A、信息边界与非法域测试。"""

import base64
import json

import numpy as np
import pytest
from riichienv import BatchedRiichiEnv, Observation

from riichi_ppo_v1.model.encoding_protocol import (
    SNAPSHOT_FACTOR_WIDTH,
    SNAPSHOT_FIELD_COUNT,
    SNAPSHOT_FIELDS,
    SNAPSHOT_NUMERIC_WIDTH,
)
from riichi_ppo_v1.model.snapshot import _validate, encode_snapshot_batch


def test_live_snapshot_is_exactly_54_rows() -> None:
    observations = list(BatchedRiichiEnv(1, seed=3, step_threads=1).reset())[0]
    factors, numeric, lengths = encode_snapshot_batch(list(observations.values()))
    assert SNAPSHOT_FIELD_COUNT == 54
    assert factors.shape == (4, SNAPSHOT_FIELD_COUNT, SNAPSHOT_FACTOR_WIDTH)
    assert numeric.shape == (4, SNAPSHOT_FIELD_COUNT, SNAPSHOT_NUMERIC_WIDTH)
    assert np.array_equal(lengths, np.full(4, SNAPSHOT_FIELD_COUNT))
    assert np.array_equal(factors[0, :, 0], np.arange(1, SNAPSHOT_FIELD_COUNT + 1))


def test_snapshot_rejects_order_domain_and_nonfinite_values() -> None:
    factors = np.zeros((SNAPSHOT_FIELD_COUNT, SNAPSHOT_FACTOR_WIDTH), dtype=np.uint8)
    for index, field in enumerate(SNAPSHOT_FIELDS):
        factors[index] = (field.field_id, field.relative_seat, 0, 0)
    numeric = np.zeros((SNAPSHOT_FIELD_COUNT, SNAPSHOT_NUMERIC_WIDTH), dtype=np.float32)
    _validate(factors, numeric)
    malformed = factors.copy(); malformed[[0, 1]] = malformed[[1, 0]]
    with pytest.raises(RuntimeError, match="order"):
        _validate(malformed, numeric)
    invalid = factors.copy(); invalid[0, 2] = SNAPSHOT_FIELDS[0].categorical_max + 1
    with pytest.raises(RuntimeError, match="domain"):
        _validate(invalid, numeric)
    bad_numeric = numeric.copy(); bad_numeric[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        _validate(factors, bad_numeric)


def test_snapshot_ignores_opponent_concealed_hands() -> None:
    observation = list(BatchedRiichiEnv(1, seed=3, step_threads=1).reset())[0][0]
    factors, numeric = encode_snapshot_batch([observation])[:2]
    payload = json.loads(base64.b64decode(observation.serialize_to_base64()))
    payload["hands"][1] = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
    changed = Observation.deserialize_from_base64(
        base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    )
    changed_factors, changed_numeric = encode_snapshot_batch([changed])[:2]
    np.testing.assert_array_equal(factors, changed_factors)
    np.testing.assert_array_equal(numeric, changed_numeric)
