"""V18 Query metadata、动作 ID 集合、supplier 域与 O0-O9/D0-D9 域测试。"""

from __future__ import annotations

import numpy as np

from riichi_ppo_v1.model.encoding_protocol import (
    QUERY_ROW_ACTION_ID,
    QUERY_ROW_ACTION_TYPE,
    QUERY_ROW_ANSWER_START,
    QUERY_ROW_PRIMARY_TILE,
    QUERY_ROW_QUERY_TYPE,
    QUERY_ROW_SOURCE_SEAT,
    SLOT_CARDINALITIES,
    SUPPLIER_REQUIRED_ACTION_TYPES,
)
from riichi_ppo_v1.sft.data import encode_kyoku
from riichi_ppo_v1.tests.v18_fixtures import first_kyoku_record


def test_query_rows_pair_and_order() -> None:
    record, _game_id = first_kyoku_record()
    samples = encode_kyoku(record, year=2024, game_id="test-1", kyoku_index=0)
    for sample in samples[:8]:
        rows = sample.query_rows
        ids = sample.action_ids
        assert rows.shape[0] == 2 * len(ids)
        assert rows[0::2, QUERY_ROW_QUERY_TYPE].tolist() == [1] * len(ids)
        assert rows[1::2, QUERY_ROW_QUERY_TYPE].tolist() == [2] * len(ids)
        assert rows[0::2, QUERY_ROW_ACTION_ID].tolist() == ids.tolist()
        assert rows[1::2, QUERY_ROW_ACTION_ID].tolist() == ids.tolist()
        assert np.all(np.diff(ids) > 0)  # 升序唯一


def test_supplier_and_answer_domains() -> None:
    record, _game_id = first_kyoku_record()
    samples = encode_kyoku(record, year=2024, game_id="test-1", kyoku_index=0)
    for sample in samples[:12]:
        rows = sample.query_rows
        action_types = rows[:, QUERY_ROW_ACTION_TYPE]
        source_seats = rows[:, QUERY_ROW_SOURCE_SEAT]
        supplier = np.isin(action_types, tuple(SUPPLIER_REQUIRED_ACTION_TYPES))
        assert np.all(source_seats[supplier] >= 1)
        assert np.all(source_seats[~supplier] == 0)
        for slot_index, slot in enumerate(("O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9")):
            codes = rows[0::2, QUERY_ROW_ANSWER_START + slot_index]
            assert np.all(codes < SLOT_CARDINALITIES[slot])
        for slot_index, slot in enumerate(("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9")):
            codes = rows[1::2, QUERY_ROW_ANSWER_START + slot_index]
            assert np.all(codes < SLOT_CARDINALITIES[slot])


def test_primary_tile_and_source_domains() -> None:
    record, _game_id = first_kyoku_record()
    samples = encode_kyoku(record, year=2024, game_id="test-1", kyoku_index=0)
    for sample in samples[:12]:
        rows = sample.query_rows
        assert np.all(rows[:, QUERY_ROW_PRIMARY_TILE] <= 34)
        assert np.all(rows[:, QUERY_ROW_SOURCE_SEAT] <= 3)
