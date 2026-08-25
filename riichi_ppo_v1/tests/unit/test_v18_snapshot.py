"""V18 Atomic Snapshot 的形状、N/A 与非法域测试。"""

import numpy as np
import pytest
from riichienv import BatchedRiichiEnv

from riichi_ppo_v1.model.encoding_protocol import SNAPSHOT_FIELDS
from riichi_ppo_v1.model.snapshot import _validate, encode_snapshot_batch


def test_live_snapshot_is_exactly_29_rows() -> None:
    observations = list(BatchedRiichiEnv(1, seed=3, step_threads=1).reset())[0]
    factors, numeric, lengths = encode_snapshot_batch(list(observations.values()))
    assert factors.shape == (4, 29, 4)
    assert numeric.shape == (4, 29, 1)
    assert np.array_equal(lengths, np.full(4, 29))
    assert np.array_equal(factors[0, :, 0], np.arange(1, 30))


def test_snapshot_rejects_order_domain_and_nonfinite_values() -> None:
    factors = np.zeros((29, 4), dtype=np.uint8)
    for index, field in enumerate(SNAPSHOT_FIELDS):
        factors[index] = (field.field_id, field.relative_seat, 0, 0)
    numeric = np.zeros((29, 1), dtype=np.float32)
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
