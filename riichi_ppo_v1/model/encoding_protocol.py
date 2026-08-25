"""现行 V18 信息编码协议的 Python 消费边界。

29 个 Snapshot 字段的顺序与域只在 Rust ``atomic_snapshot`` 定义一次,本模块
直接读取其机器可读 schema。Query 行与 answer slot 的存储契约仍在本模块单点定义。
"""

from __future__ import annotations

from dataclasses import dataclass

import riichi

# 唯一协议版本与派生的格式标识。
ENCODING_PROTOCOL_VERSION = int(riichi.ENCODING_PROTOCOL_VERSION)
if ENCODING_PROTOCOL_VERSION != 18:
    raise RuntimeError("installed riichi extension does not provide encoding protocol V18")
ENCODED_FORMAT = f"riichi-sft-encoded-v{ENCODING_PROTOCOL_VERSION}"

# 每个合法动作固定一对 query;每个 query 固定 10 个 answer slot。
QUERY_OFFENSE = 1
QUERY_DEFENSE = 2
QUERY_SLOT_COUNT = 10

# 每个 query 的存储行宽:[query_type, action_id, action_type_code,
# primary_tile_code, source_seat_code, answer_0..answer_9]。
QUERY_ROW_WIDTH = 15
# query 存储行的字段下标(单一来源,供编码/嵌入/审计共用)。
QUERY_ROW_QUERY_TYPE = 0
QUERY_ROW_ACTION_ID = 1
QUERY_ROW_ACTION_TYPE = 2
QUERY_ROW_PRIMARY_TILE = 3
QUERY_ROW_SOURCE_SEAT = 4
QUERY_ROW_ANSWER_START = 5

# 动作类型 → 编码(0 保留给 padding;自摸与荣和必须区分 supplier 语义)。
ACTION_TYPE_CODES: dict[str, int] = {
    "none": 1,
    "pass": 1,
    "dahai": 2,
    "reach": 3,
    "chi": 4,
    "pon": 5,
    "daiminkan": 6,
    "ankan": 7,
    "kakan": 8,
    "tsumo": 9,
    "ron": 10,
    "hora": 10,
    "ryukyoku": 11,
}
ACTION_TYPE_CARDINALITY = 12
SUPPLIER_REQUIRED_ACTION_TYPES = frozenset(
    ACTION_TYPE_CODES[name] for name in ("chi", "pon", "daiminkan", "ron")
)

@dataclass(frozen=True)
class SnapshotField:
    field_id: int
    name: str
    relative_seat: int
    categorical_max: int
    tile_max: int
    numeric: bool


SNAPSHOT_FIELDS: tuple[SnapshotField, ...] = tuple(
    SnapshotField(*row) for row in riichi.atomic_snapshot_schema()
)
SNAPSHOT_FIELD_COUNT = int(riichi.SNAPSHOT_FIELD_COUNT)
SNAPSHOT_FACTOR_WIDTH = 4
SNAPSHOT_NUMERIC_WIDTH = 1
if len(SNAPSHOT_FIELDS) != SNAPSHOT_FIELD_COUNT:
    raise RuntimeError("Rust Snapshot schema length does not match exported field count")
if tuple(field.field_id for field in SNAPSHOT_FIELDS) != tuple(
    range(1, SNAPSHOT_FIELD_COUNT + 1)
):
    raise RuntimeError("Rust Snapshot field ids are not contiguous")
SNAPSHOT_FACTOR_CARDINALITIES = (
    max(field.field_id for field in SNAPSHOT_FIELDS) + 1,
    max(field.relative_seat for field in SNAPSHOT_FIELDS) + 1,
    max(field.categorical_max for field in SNAPSHOT_FIELDS) + 1,
    max(field.tile_max for field in SNAPSHOT_FIELDS) + 1,
)
SNAPSHOT_FIELD_BY_NAME = {field.name: field for field in SNAPSHOT_FIELDS}

# slot 的规范顺序(嵌入与审计共用,禁止依赖 dict 的隐式迭代顺序)。
OFFENSE_SLOT_ORDER: tuple[str, ...] = (
    "O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9",
)
DEFENSE_SLOT_ORDER: tuple[str, ...] = (
    "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
)

# 每个 slot 的规范取值列表:列表下标即 categorical 编码值。N/A 的位置严格沿用
# 设计文档钦定的枚举顺序:O3/O4/O5/O6 的 N/A 在首位,O8 与 D0–D5、D9 的 N/A 在
# 末位。
OFFENSE_SLOT_LABELS: dict[str, tuple[str, ...]] = {
    "O0": ("AGARI", "0", "1", "2", "3", "4", "5+"),
    "O1": ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"),
    "O2": ("0", "1-4", "5-8", "9-12", "13-16", "17-20", "21+"),
    "O3": ("N/A", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"),
    "O4": ("N/A", "NO_YAKU", "PARTIAL_YAKU", "ALL_YAKU"),
    "O5": ("N/A", "1", "2", "3", "4", "5+"),
    "O6": ("N/A", "NO_FURITEN", "PERMANENT_FURITEN", "TEMPORARY_FURITEN"),
    "O7": ("YES", "NO"),
    "O8": ("YES", "NO", "N/A"),
    "O9": ("0", "1", "2", "3", "4", "5+"),
}

_GENBUTSU = ("GENBUTSU", "NOT_GENBUTSU", "N/A")
_SUJI = ("SUJI", "NOT_SUJI", "N/A")
_STOCK = ("0", "1", "2", "3", "4+")

DEFENSE_SLOT_LABELS: dict[str, tuple[str, ...]] = {
    "D0": _GENBUTSU,
    "D1": _GENBUTSU,
    "D2": _GENBUTSU,
    "D3": _SUJI,
    "D4": _SUJI,
    "D5": _SUJI,
    "D6": _STOCK,
    "D7": _STOCK,
    "D8": _STOCK,
    "D9": ("0", "1", "2", "3", "4", "N/A"),
}

SLOT_CARDINALITIES: dict[str, int] = {
    **{slot: len(labels) for slot, labels in OFFENSE_SLOT_LABELS.items()},
    **{slot: len(labels) for slot, labels in DEFENSE_SLOT_LABELS.items()},
}


def bucket_o1(kinds: int) -> int:
    """O1 有效牌种类数 → 编码:0..9 精确,10+ 截断到 10。"""
    return max(0, min(int(kinds), 10))


def bucket_o2(remaining: int) -> int:
    """O2 有效牌剩余枚数 → 编码:0 精确,1-4/5-8/9-12/13-16/17-20/21+。"""
    value = max(0, int(remaining))
    if value == 0:
        return 0
    return min((value - 1) // 4 + 1, 6)


def bucket_o3(waits: int | None) -> int:
    """O3 合法等待种类数 → 编码:None(非听牌)N/A=0,1..13 精确。"""
    if waits is None:
        return 0
    return max(1, min(int(waits), 13))


def bucket_o5(han: int | None) -> int:
    """O5 基础番数 → 编码:None=N/A=0,1..4 精确,5+ 截断到 5。"""
    if han is None:
        return 0
    return max(1, min(int(han), 5))


def bucket_o9(dora_aka: int) -> int:
    """O9 保留宝牌/赤牌数 → 编码:0..4 精确,5+ 截断到 5。"""
    return max(0, min(int(dora_aka), 5))


def bucket_d6(stock: int) -> int:
    """D6-D8 安全牌库存 → 编码:0..3 精确,4+ 截断到 4。"""
    return max(0, min(int(stock), 4))


def bucket_d9(visible: int | None) -> int:
    """D9 候选牌公开出现数 → 编码:0..4 精确,N/A=5。"""
    if visible is None:
        return 5
    return max(0, min(int(visible), 4))
