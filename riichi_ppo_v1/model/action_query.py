"""V16 Action Query 的 Rust 语义兼容接口。

牌形、动作分派、向听、有效牌、防守与役种均由 Rust 单一实现计算。本模块只保留
历史 Python 调用方需要的 ``ActionQuery`` 返回类型与固定宽度行转换。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoding_protocol import (
    ACTION_TYPE_CODES,
    QUERY_ROW_ACTION_ID,
    QUERY_ROW_ACTION_TYPE,
    QUERY_ROW_ANSWER_START,
    QUERY_ROW_PRIMARY_TILE,
    QUERY_ROW_QUERY_TYPE,
    QUERY_ROW_SOURCE_SEAT,
    QUERY_ROW_WIDTH,
)
from .v16_rust_encoding import encode_action_queries_batch_native


@dataclass(frozen=True)
class ActionQuery:
    query_type: int
    action_id: int
    action_type: str
    primary_tile: int | None
    source_seat: int | None
    answers: tuple[int, ...]


def _action_kind_label(action: object) -> str:
    """把原生 ActionType 名称规范为 V16 兼容标签,不参与语义计算。"""
    name = str(getattr(action, "action_type", "")).lower().rsplit(".", 1)[-1]
    return {
        "discard": "dahai",
        "riichi": "reach",
        "pass": "none",
        "kyushukyuhai": "ryukyoku",
    }.get(name, name)


def _decode_query_row(row: np.ndarray, action_type: str) -> ActionQuery:
    """把 Rust 连续行恢复为旧调用方使用的不可变对象。"""
    primary_code = int(row[QUERY_ROW_PRIMARY_TILE])
    source_code = int(row[QUERY_ROW_SOURCE_SEAT])
    return ActionQuery(
        query_type=int(row[QUERY_ROW_QUERY_TYPE]),
        action_id=int(row[QUERY_ROW_ACTION_ID]),
        action_type=action_type,
        primary_tile=None if primary_code == 0 else primary_code - 1,
        source_seat=None if source_code == 0 else source_code - 1,
        answers=tuple(
            int(value)
            for value in row[QUERY_ROW_ANSWER_START:QUERY_ROW_WIDTH]
        ),
    )


def analyze_action_queries(
    observation: object,
    action: object,
    action_id: int,
) -> tuple[ActionQuery, ActionQuery]:
    """通过 Rust 融合编码器生成一个动作的 Offense/Defense Query。"""
    encoded = encode_action_queries_batch_native([(observation, action, int(action_id))])
    rows = encoded.query_rows[0]
    action_type = _action_kind_label(action)
    return (
        _decode_query_row(rows[0], action_type),
        _decode_query_row(rows[1], action_type),
    )


def encode_query_row(query: ActionQuery) -> np.ndarray:
    """把兼容对象编码为固定宽度存储行。"""
    if len(query.answers) != 10:
        raise ValueError(f"action query must have 10 answers, got {len(query.answers)}")
    row = np.zeros(QUERY_ROW_WIDTH, dtype=np.int32)
    row[QUERY_ROW_QUERY_TYPE] = int(query.query_type)
    row[QUERY_ROW_ACTION_ID] = int(query.action_id)
    row[QUERY_ROW_ACTION_TYPE] = int(ACTION_TYPE_CODES.get(query.action_type, 0))
    row[QUERY_ROW_PRIMARY_TILE] = 0 if query.primary_tile is None else int(query.primary_tile) + 1
    row[QUERY_ROW_SOURCE_SEAT] = 0 if query.source_seat is None else int(query.source_seat) + 1
    row[QUERY_ROW_ANSWER_START:QUERY_ROW_WIDTH] = np.asarray(query.answers, dtype=np.int32)
    return row
