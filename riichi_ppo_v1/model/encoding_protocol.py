"""V16 信息编码协议单一来源。

2026-08-16 澄清:输入编码收敛为单一协议版本 v16,废弃 token schema / feature
schema / rust-analysis / decision-analysis 多版本拆分与 `_v<A>_v<B>` 组合命名。
本模块是协议版本、数据集格式标识、全部 query slot 语义/基数/bucket 边界的唯一
权威来源;其余模块与测试一律引用本模块常量,禁止散落魔法数字。
"""

from __future__ import annotations

# 唯一协议版本与派生的格式标识。
ENCODING_PROTOCOL_VERSION = 16
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

# 动作类型 → 编码(0 保留给 padding;和牌自摸/荣和归一为同一类型)。
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
    "ron": 9,
    "hora": 9,
    "ryukyoku": 10,
}
ACTION_TYPE_CARDINALITY = 11

# Compact Snapshot 行类型(单一来源)。
SNAPSHOT_KIND_BASE = 0
SNAPSHOT_KIND_DORA = 1
SNAPSHOT_KIND_SCORE = 2
SNAPSHOT_KIND_SUMMARY = 3
SNAPSHOT_KIND_COUNT = 4
SNAPSHOT_CAT_WIDTH = 4
SNAPSHOT_NUM_WIDTH = 7

# Snapshot 连续字段的固定归一化刻度(编码期一次性应用,训练期不再动态缩放)。
SNAPSHOT_SCORE_SCALE = 25_000.0
SNAPSHOT_PRESSURE_SCALE = 25_000.0
SNAPSHOT_HONBA_SCALE = 10.0
SNAPSHOT_STICKS_SCALE = 10.0
SNAPSHOT_TILES_LEFT_SCALE = 136.0
SNAPSHOT_TURN_SCALE = 20.0
SNAPSHOT_MELD_SCALE = 4.0
SNAPSHOT_RIVER_SCALE = 20.0

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
