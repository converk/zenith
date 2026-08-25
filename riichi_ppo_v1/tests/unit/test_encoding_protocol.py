"""信息编码协议 V18 单一来源的契约测试。"""

from __future__ import annotations

from riichi_ppo_v1.model import encoding_protocol as protocol
from riichi_ppo_v1.model.schema import TOKEN_SCHEMA_VERSION


def test_single_protocol_version() -> None:
    """协议版本唯一且数据集格式标识由同一常量派生。"""
    assert protocol.ENCODING_PROTOCOL_VERSION == 18
    assert protocol.ENCODED_FORMAT == "riichi-sft-encoded-v18"
    assert TOKEN_SCHEMA_VERSION == protocol.ENCODING_PROTOCOL_VERSION


def test_slot_cardinalities_match_contract() -> None:
    """20 个 slot 的基数与 V18 输入契约逐项一致。"""
    expected = {
        "O0": 7, "O1": 11, "O2": 7, "O3": 14, "O4": 4, "O5": 6,
        "O6": 4, "O7": 2, "O8": 3, "O9": 6,
        "D0": 3, "D1": 3, "D2": 3, "D3": 3, "D4": 3, "D5": 3,
        "D6": 5, "D7": 5, "D8": 5, "D9": 6,
    }
    assert protocol.SLOT_CARDINALITIES == expected
    assert len(protocol.SLOT_CARDINALITIES) == 20


def test_na_is_code_zero_where_present() -> None:
    """带 N/A 的 slot 必须把 N/A 放在编码 0。"""
    for slot in ("O3", "O4", "O5", "O6"):
        assert protocol.OFFENSE_SLOT_LABELS[slot][0] == "N/A"
    for slot in ("D0", "D1", "D2", "D3", "D4", "D5"):
        assert protocol.DEFENSE_SLOT_LABELS[slot][-1] == "N/A"
    assert protocol.OFFENSE_SLOT_LABELS["O8"][-1] == "N/A"
    assert protocol.DEFENSE_SLOT_LABELS["D9"][-1] == "N/A"


def test_bucket_boundaries() -> None:
    """bucket 边界:小值精确、大值截断,覆盖契约例示值。"""
    assert [protocol.bucket_o1(v) for v in (0, 3, 7, 9, 10, 12)] == [0, 3, 7, 9, 10, 10]
    assert [protocol.bucket_o2(v) for v in (0, 1, 4, 5, 9, 12, 13, 17, 21, 52)] == [
        0, 1, 1, 2, 3, 3, 4, 5, 6, 6,
    ]
    assert [protocol.bucket_o3(v) for v in (None, 1, 2, 13, 14)] == [0, 1, 2, 13, 13]
    assert [protocol.bucket_o5(v) for v in (None, 1, 4, 5, 13)] == [0, 1, 4, 5, 5]
    assert [protocol.bucket_o9(v) for v in (0, 4, 5, 6)] == [0, 4, 5, 5]
    assert [protocol.bucket_d6(v) for v in (0, 3, 4, 7)] == [0, 3, 4, 4]
    assert [protocol.bucket_d9(v) for v in (None, 0, 4, 5)] == [5, 0, 4, 4]
