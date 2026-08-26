"""现行 V18 当前局面输入协议的 Python 单一来源。

本模块定义：
- 全局协议常数（行宽、数值宽、context 上限、segment/kind/separator 编号）；
- 每个 token 类别的字段 schema（离散字段基数 + 数值字段归一化），顺序即行偏移；
- Action Query 行与 O0–O9 / D0–D9 槽位定义（沿用旧协议语义）。

Rust 编码器 ``riichienv-python/src/current_state_encoding.rs`` 镜像同一常量；
两处必须同步修改，并由 ``semantic_validation.py`` 与集成测试做交叉验证。
"""

from __future__ import annotations

from dataclasses import dataclass

import riichi

# 唯一协议版本与派生的格式标识（版本号保持 V18，契约 hash 随 schema 更新而变化）。
ENCODING_PROTOCOL_VERSION = int(riichi.ENCODING_PROTOCOL_VERSION)
if ENCODING_PROTOCOL_VERSION != 18:
    raise RuntimeError("installed riichi extension does not provide encoding protocol V18")
ENCODED_FORMAT = f"riichi-sft-encoded-v{ENCODING_PROTOCOL_VERSION}"
STATE_PROTOCOL_VERSION = "riichi-current-state-v18-1"

# 动作空间维度（领域常量，schema.py 从此单源导入）。
NUM_ACTIONS = 241

# 行布局：row[:, 0]=segment, row[:, 1]=token_kind, row[:, 2:]=类别字段。
TOKEN_ROW_WIDTH = 32
TOKEN_NUMERIC_WIDTH = 8
# 生产 context 上限（严格上界见 specs/010 的 research §2.7）。
CONTEXT_TOKENS = 256

# ---- segment 编号 ----
SEGMENT_SHARED = 1
SEGMENT_ANALYSIS = 2
SEGMENT_ACTIONS = 3
SEGMENT_CRITIC_PRIVATE = 4
SEGMENT_CRITIC_FUTURE = 5

# ---- token kind 编号 ----
KIND_BOS = 1
KIND_TABLE = 2
KIND_SELF_HAND = 3
KIND_SELF_STATE_ANALYSIS = 4
KIND_PLAYER = 5
KIND_RIVER_SUMMARY = 6
KIND_RIVER_DISCARD = 7
KIND_MELD = 8
KIND_TILE_STATE = 9
KIND_OPPONENT_ANALYSIS = 10
KIND_ACTION_OFFENSE_QUERY = 11
KIND_ACTION_DEFENSE_QUERY = 12
KIND_CRITIC_HAND = 13
KIND_CRITIC_FUTURE = 14

# ---- 分隔符（kind = 100 + separator_id；单点定义） ----
SEPARATOR_IDS: dict[str, int] = {
    "SEP_SELF_HAND": 1,
    "SEP_PLAYERS": 2,
    "SEP_RIVERS": 3,
    "SEP_SHIMOCHA_RIVER": 4,
    "SEP_TOIMEN_RIVER": 5,
    "SEP_KAMICHA_RIVER": 6,
    "SEP_MELDS": 7,
    "SEP_TILE_STATE": 8,
    "SEP_OPPONENT_ANALYSIS": 9,
    "SEP_ACTIONS": 10,
    "SEP_CRITIC": 11,
}
SEPARATOR_KINDS: dict[str, int] = {
    name: 100 + separator_id for name, separator_id in SEPARATOR_IDS.items()
}
NUM_SEPARATORS = len(SEPARATOR_IDS)
# 常用分隔符 kind 的单点别名（供架构/校验直接引用）。
KIND_SEP_SELF_HAND = SEPARATOR_KINDS["SEP_SELF_HAND"]
KIND_SEP_PLAYERS = SEPARATOR_KINDS["SEP_PLAYERS"]
KIND_SEP_RIVERS = SEPARATOR_KINDS["SEP_RIVERS"]
KIND_SEP_SHIMOCHA_RIVER = SEPARATOR_KINDS["SEP_SHIMOCHA_RIVER"]
KIND_SEP_TOIMEN_RIVER = SEPARATOR_KINDS["SEP_TOIMEN_RIVER"]
KIND_SEP_KAMICHA_RIVER = SEPARATOR_KINDS["SEP_KAMICHA_RIVER"]
KIND_SEP_MELDS = SEPARATOR_KINDS["SEP_MELDS"]
KIND_SEP_TILE_STATE = SEPARATOR_KINDS["SEP_TILE_STATE"]
KIND_SEP_OPPONENT_ANALYSIS = SEPARATOR_KINDS["SEP_OPPONENT_ANALYSIS"]
KIND_SEP_ACTIONS = SEPARATOR_KINDS["SEP_ACTIONS"]
KIND_SEP_CRITIC = SEPARATOR_KINDS["SEP_CRITIC"]
# 分隔符所在 segment（单点定义，供结构校验使用）。
SEPARATOR_SEGMENTS: dict[int, int] = {
    KIND_SEP_SELF_HAND: SEGMENT_SHARED,
    KIND_SEP_PLAYERS: SEGMENT_SHARED,
    KIND_SEP_RIVERS: SEGMENT_SHARED,
    KIND_SEP_SHIMOCHA_RIVER: SEGMENT_SHARED,
    KIND_SEP_TOIMEN_RIVER: SEGMENT_SHARED,
    KIND_SEP_KAMICHA_RIVER: SEGMENT_SHARED,
    KIND_SEP_MELDS: SEGMENT_SHARED,
    KIND_SEP_TILE_STATE: SEGMENT_SHARED,
    KIND_SEP_OPPONENT_ANALYSIS: SEGMENT_ANALYSIS,
    KIND_SEP_ACTIONS: SEGMENT_ACTIONS,
    KIND_SEP_CRITIC: SEGMENT_CRITIC_PRIVATE,
}


@dataclass(frozen=True)
class DiscreteField:
    """一个离散槽位：名称 + 编码基数（取值 0..cardinality-1）。"""

    name: str
    cardinality: int


@dataclass(frozen=True)
class NumericField:
    """一个数值槽位：名称 + 归一化除数（clip 到 [-1,1]）。"""

    name: str
    scale: float = 100_000.0


@dataclass(frozen=True)
class CategorySchema:
    """一个 token 类别的完整 schema（顺序即行偏移）。"""

    kind: int
    name: str
    segment: int
    cls: str  # "SIMPLE" / "DENSE" / "SEPARATOR"
    discrete: tuple[DiscreteField, ...]
    numeric: tuple[NumericField, ...] = ()
    slot_count: int = 0  # 仅 RIVER_SUMMARY 为 6


_D = DiscreteField
_N = NumericField

# ---- 类别 schema 表（顺序必须与 contracts/v18-current-state-contract.md §3 一致） ----
CATEGORY_SCHEMAS: dict[int, CategorySchema] = {
    KIND_BOS: CategorySchema(KIND_BOS, "BOS", SEGMENT_SHARED, "SIMPLE", ()),
    KIND_TABLE: CategorySchema(
        KIND_TABLE, "TABLE", SEGMENT_SHARED, "DENSE",
        (
            _D("round_wind", 4),
            _D("kyoku_index", 4),
            _D("honba_bucket", 21),
            _D("riichi_sticks_bucket", 5),
            _D("oya_seat", 4),
            _D("self_seat", 4),
            _D("decision_mode", 3),
            _D("drawn_tile_type", 35),
            _D("drawn_tile_red", 2),
            _D("drawn_is_current", 2),
            _D("self_riichi_status", 3),
            _D("dora_indicator_type_slot_1", 35),
            _D("dora_indicator_type_slot_2", 35),
            _D("dora_indicator_type_slot_3", 35),
            _D("dora_indicator_type_slot_4", 35),
            _D("dora_indicator_type_slot_5", 35),
            _D("dora_indicator_red_slot_1", 2),
            _D("dora_indicator_red_slot_2", 2),
            _D("dora_indicator_red_slot_3", 2),
            _D("dora_indicator_red_slot_4", 2),
            _D("dora_indicator_red_slot_5", 2),
            _D("own_rank", 5),
        ),
        (
            _N("score_0"),
            _N("score_1"),
            _N("score_2"),
            _N("score_3"),
            _N("diff_shimo"),
            _N("diff_toimen"),
            _N("diff_kamicha"),
        ),
    ),
    KIND_SELF_HAND: CategorySchema(
        KIND_SELF_HAND, "SELF_HAND", SEGMENT_SHARED, "SIMPLE",
        (_D("tile_type", 35), _D("count", 5), _D("has_red", 2), _D("is_drawn", 2), _D("locked_under_riichi", 2)),
    ),
    KIND_SELF_STATE_ANALYSIS: CategorySchema(
        KIND_SELF_STATE_ANALYSIS, "SELF_STATE_ANALYSIS", SEGMENT_SHARED, "DENSE",
        (
            _D("menzen", 2),
            _D("concealed_count", 15),
            _D("meld_count", 5),
            _D("overall_shanten", 10),
            _D("standard_shanten", 10),
            _D("chiitoitsu_shanten", 10),
            _D("kokushi_shanten", 10),
            _D("advance_kind_count", 35),
            _D("advance_remaining", 101),
            _D("wait_kind_count", 35),
            _D("wait_remaining", 101),
            _D("permanent_furiten", 2),
            _D("doujun_furiten", 2),
            _D("riichi_furiten", 2),
            _D("own_dora_count", 6),
            _D("own_aka_count", 6),
            _D("own_yakuhai_han", 7),
            _D("base_han_total", 11),
        ),
    ),
    KIND_PLAYER: CategorySchema(
        KIND_PLAYER, "PLAYER", SEGMENT_SHARED, "DENSE",
        (
            _D("relative_seat", 4),
            _D("absolute_seat", 4),
            _D("seat_wind", 4),
            _D("is_oya", 2),
            _D("rank", 5),
            _D("concealed_count", 15),
            _D("meld_count", 5),
            _D("kan_count", 5),
            _D("menzen", 2),
            _D("river_length", 25),
            _D("riichi_status", 3),
            _D("riichi_turn", 27),
            _D("riichi_decl_tile_type", 35),
            _D("riichi_decl_red", 2),
            _D("post_riichi_discard_count", 17),
            _D("open_meld_yakuhai_han", 7),
            _D("visible_meld_dora_aka_han", 9),
        ),
        (_N("points"), _N("diff")),
    ),
    KIND_RIVER_SUMMARY: CategorySchema(
        KIND_RIVER_SUMMARY, "RIVER_SUMMARY", SEGMENT_SHARED, "DENSE",
        (
            _D("valid_length", 7),
            *(
                field
                for i in range(1, 7)
                for field in (
                    _D(f"slot_{i}_tile_type", 35), _D(f"slot_{i}_red", 2),
                    _D(f"slot_{i}_cut", 3), _D(f"slot_{i}_riichi_stage", 4),
                )
            ),
        ),
        slot_count=6,
    ),
    KIND_RIVER_DISCARD: CategorySchema(
        KIND_RIVER_DISCARD, "RIVER_DISCARD", SEGMENT_SHARED, "SIMPLE",
        (
            _D("relative_seat", 4), _D("river_index", 25), _D("tile_type", 35),
            _D("red", 2), _D("cut", 2), _D("riichi_stage", 3), _D("supplied", 2),
            _D("age_bucket", 4),
        ),
    ),
    KIND_MELD: CategorySchema(
        KIND_MELD, "MELD", SEGMENT_SHARED, "DENSE",
        (
            _D("owner_relative", 4), _D("meld_type_code", 6),
            _D("tile0_type", 35), _D("tile0_red", 2),
            _D("tile1_type", 35), _D("tile1_red", 2),
            _D("tile2_type", 35), _D("tile2_red", 2),
            _D("tile3_type", 35), _D("tile3_red", 2),
            _D("called_tile_type", 35), _D("called_tile_red", 2),
            _D("supplier_relative", 4), _D("open", 2), _D("meld_index", 5),
            _D("yakuhai_han", 7), _D("visible_dora_aka_han", 9),
        ),
    ),
    KIND_TILE_STATE: CategorySchema(
        KIND_TILE_STATE, "TILE_STATE", SEGMENT_SHARED, "SIMPLE",
        (
            _D("tile_type", 35), _D("self_concealed_count", 5), _D("self_discard_count", 5),
            _D("self_ever_discarded", 2), _D("public_count", 5), _D("known_count", 5),
            _D("unknown_count", 5), _D("all_seen", 2), _D("dora_multiplicity", 6),
            _D("is_dora", 2), _D("round_wind_match", 2), _D("seat_wind_match", 2),
            _D("red_five_kind", 2), _D("is_advance", 2), _D("is_win", 2),
            _D("genbutsu_shimo", 2), _D("genbutsu_toimen", 2), _D("genbutsu_kamicha", 2),
            _D("suji_shimo", 4), _D("suji_toimen", 4), _D("suji_kamicha", 4),
            _D("wall_class", 3), _D("dora_neighbor", 2),
        ),
    ),
    KIND_OPPONENT_ANALYSIS: CategorySchema(
        KIND_OPPONENT_ANALYSIS, "OPPONENT_ANALYSIS", SEGMENT_ANALYSIS, "DENSE",
        (
            _D("relative_seat", 4), _D("riichi_status", 3), _D("riichi_turn", 27),
            _D("riichi_decl_tile_type", 35), _D("riichi_decl_red", 2), _D("menzen", 2),
            _D("concealed_count", 15), _D("meld_count", 5), _D("kan_count", 5),
            _D("open_meld_yakuhai_han", 7), _D("visible_meld_dora_aka_han", 9),
            _D("post_riichi_tedashi", 17), _D("post_riichi_tsumogiri", 17),
            _D("recent6_tedashi", 7), _D("recent6_tsumogiri", 7),
            _D("own_genbutsu_kind_count", 35), _D("own_genbutsu_entity_count", 101),
            _D("river_length", 25),
        ),
    ),
    KIND_CRITIC_HAND: CategorySchema(
        KIND_CRITIC_HAND, "CRITIC_HAND", SEGMENT_CRITIC_PRIVATE, "SIMPLE",
        (_D("relative_seat", 4), _D("tile_type", 35), _D("red", 2), _D("count", 5)),
    ),
    KIND_CRITIC_FUTURE: CategorySchema(
        KIND_CRITIC_FUTURE, "CRITIC_FUTURE", SEGMENT_CRITIC_FUTURE, "SIMPLE",
        (_D("position", 6), _D("tile_type", 35), _D("red", 2)),
    ),
}


# ---- Action Query 行（沿用旧协议，行宽 15；槽位定义先于 ACTION schema 构造） ----
QUERY_OFFENSE = 1
QUERY_DEFENSE = 2
QUERY_SLOT_COUNT = 10
QUERY_ROW_WIDTH = 15
QUERY_ROW_QUERY_TYPE = 0
QUERY_ROW_ACTION_ID = 1
QUERY_ROW_ACTION_TYPE = 2
QUERY_ROW_PRIMARY_TILE = 3
QUERY_ROW_SOURCE_SEAT = 4
QUERY_ROW_ANSWER_START = 5

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

OFFENSE_SLOT_ORDER: tuple[str, ...] = (
    "O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9",
)
DEFENSE_SLOT_ORDER: tuple[str, ...] = (
    "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
)
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


def _action_fields(slot_prefix: str) -> tuple[DiscreteField, ...]:
    """Action Query 的嵌入特征：metadata + action_id + 10 个 answer 槽（Offense 用 O、Defense 用 D）。

    ``action_id``（0..240）是完整 consume/赤牌身份的规范编码，进入 token embedding；
    query_rows 仍保留 15 宽原始元数据供一致性校验。
    """
    return (
        DiscreteField("action_type_code", 12),
        DiscreteField("primary_tile_code", 35),
        DiscreteField("source_seat_code", 4),
        DiscreteField("tsumogiri_mode", 2),
        DiscreteField("action_id", NUM_ACTIONS),
        *(DiscreteField(f"answer_{index}", SLOT_CARDINALITIES[slot_prefix + str(index)]) for index in range(10)),
    )


CATEGORY_SCHEMAS[KIND_ACTION_OFFENSE_QUERY] = CategorySchema(
    KIND_ACTION_OFFENSE_QUERY, "ACTION_OFFENSE_QUERY", SEGMENT_ACTIONS, "DENSE",
    _action_fields("O"),
)
CATEGORY_SCHEMAS[KIND_ACTION_DEFENSE_QUERY] = CategorySchema(
    KIND_ACTION_DEFENSE_QUERY, "ACTION_DEFENSE_QUERY", SEGMENT_ACTIONS, "DENSE",
    _action_fields("D"),
)

# ---- 段 → kind 集合（在全部 schema 就绪后计算） ----
SEGMENT_KINDS: dict[int, frozenset[int]] = {
    segment: frozenset(schema.kind for schema in CATEGORY_SCHEMAS.values() if schema.segment == segment)
    for segment in range(1, 6)
}


def is_separator_kind(kind: int) -> bool:
    return kind in SEPARATOR_KINDS.values()


def separator_id_of_kind(kind: int) -> int:
    for name, value in SEPARATOR_KINDS.items():
        if value == kind:
            return SEPARATOR_IDS[name]
    raise ValueError(f"kind {kind} is not a separator")


def separator_kind_of_id(separator_id: int) -> int:
    for name, value in SEPARATOR_IDS.items():
        if value == separator_id:
            return SEPARATOR_KINDS[name]
    raise ValueError(f"separator id {separator_id} is unknown")


def bucket_o1(kinds: int) -> int:
    """O1 有效牌种类数 → 编码：0..9 精确,10+ 截断到 10。"""
    return max(0, min(int(kinds), 10))


def bucket_o2(remaining: int) -> int:
    """O2 有效牌剩余枚数 → 编码：0 精确,1-4/5-8/9-12/13-16/17-20/21+。"""
    value = max(0, int(remaining))
    if value == 0:
        return 0
    return min((value - 1) // 4 + 1, 6)


def bucket_o3(waits: int | None) -> int:
    """O3 合法等待种类数 → 编码：None(非听牌)N/A=0,1..13 精确。"""
    if waits is None:
        return 0
    return max(1, min(int(waits), 13))


def bucket_o5(han: int | None) -> int:
    """O5 基础番数 → 编码：None=N/A=0,1..4 精确,5+ 截断到 5。"""
    if han is None:
        return 0
    return max(1, min(int(han), 5))


def bucket_o9(dora_aka: int) -> int:
    """O9 保留宝牌/赤牌数 → 编码：0..4 精确,5+ 截断到 5。"""
    return max(0, min(int(dora_aka), 5))


def bucket_d6(stock: int) -> int:
    """D6-D8 安全牌库存 → 编码：0..3 精确,4+ 截断到 4。"""
    return max(0, min(int(stock), 4))


def bucket_d9(visible: int | None) -> int:
    """D9 候选牌公开出现数 → 编码：0..4 精确,N/A=5。"""
    if visible is None:
        return 5
    return max(0, min(int(visible), 4))
