"""V18 当前局面批编码的 Python 装配层。

Rust/PyO3 编码器（``riichienv.prepare_current_state_batch``）直接以原生
Observation 当前字段构造 Shared 公共前缀 + 三个 Opponent Analysis 的扁平行；
本模块负责：
1. 按决策切分 Rust 行，补写 ``SEP_ACTIONS`` 分隔符行；
2. 调用既有 ``encode_action_queries_batch_native`` 生成 O/D Query 行（15 宽），
   并把每个动作对的 15 个嵌入特征按 schema 顺序写入 action token 行；
3. 按 action ID 升序规范排序（调用方提供的 triples 本身即升序，这里再次校验）；
4. 输出模型/训练入口统一的 ``EncodedStateBatch``。

同一局面的 Observation/replay、PyO3、precompute、shard、collator 均走本模块保证一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import riichienv

from .encoding_protocol import (
    KIND_ACTION_DEFENSE_QUERY,
    KIND_ACTION_OFFENSE_QUERY,
    KIND_SEP_ACTIONS,
    QUERY_ROW_ACTION_ID,
    QUERY_ROW_ACTION_TYPE,
    QUERY_ROW_ANSWER_START,
    QUERY_ROW_PRIMARY_TILE,
    QUERY_ROW_QUERY_TYPE,
    QUERY_ROW_SOURCE_SEAT,
    QUERY_ROW_WIDTH,
    SEGMENT_ACTIONS,
    TOKEN_NUMERIC_WIDTH,
    TOKEN_ROW_WIDTH,
)
from .native_encoding import encode_action_queries_batch_native
from .schema import NUM_ACTIONS


@dataclass(frozen=True)
class EncodedStateBatch:
    """一批决策的完整 Actor 输入（行已按规范排序并含分隔符）。"""

    actor_factors: np.ndarray  # [B, T_max, 32] int32
    actor_numeric: np.ndarray  # [B, T_max, 8] float32
    actor_lengths: np.ndarray  # [B] int64
    query_rows: np.ndarray  # [B, 2*Q_max, 15] int32
    action_ids: np.ndarray  # [B, Q_max] int32
    query_pair_counts: np.ndarray  # [B] int64
    legal_mask: np.ndarray  # [B, 241] bool


def _action_row(query_type: int, row: np.ndarray, tsumogiri_mode: int) -> np.ndarray:
    """把 15 宽 query 行转换为 32 宽 action token 行（15 个嵌入特征，含 action_id）。"""
    action_row = np.zeros(TOKEN_ROW_WIDTH, dtype=np.int32)
    action_row[0] = SEGMENT_ACTIONS
    action_row[1] = KIND_ACTION_OFFENSE_QUERY if query_type == 1 else KIND_ACTION_DEFENSE_QUERY
    features = (
        int(row[QUERY_ROW_ACTION_TYPE]),
        int(row[QUERY_ROW_PRIMARY_TILE]),
        int(row[QUERY_ROW_SOURCE_SEAT]),
        int(tsumogiri_mode),
        int(row[QUERY_ROW_ACTION_ID]),
        *[int(value) for value in row[QUERY_ROW_ANSWER_START:QUERY_ROW_WIDTH]],
    )
    action_row[2:2 + len(features)] = np.asarray(features, dtype=np.int32)
    return action_row


def _tsumogiri_mode(action_id: int) -> int:
    """舍弃行动作 id 低 1 位恢复 tsumogiri 模式（仅 discard id∈[1,75) 有效）。"""
    if 1 <= int(action_id) < 75:
        return (int(action_id) - 1) % 2
    return 0


def encode_batch(
    decisions: list[tuple[object, list[tuple[object, int]]]],
) -> EncodedStateBatch:
    """一次编码一批决策。

    ``decisions`` 每项为 ``(observation, [(action, action_id), ...])``，
    合法动作列表必须已按 action_id 升序、每个 action_id 唯一。
    """
    if not decisions:
        raise ValueError("cannot encode an empty current-state batch")
    observations = [obs for obs, _actions in decisions]
    ordered_rows: list[list[tuple[object, int]]] = []
    triples: list[tuple[object, object, int]] = []
    row_action_counts: list[int] = []
    for _obs, actions in decisions:
        # 规范排序：按 action_id 升序，环境返回顺序不影响编码结果。
        ordered = sorted(actions, key=lambda item: int(item[1]))
        previous = -1
        for action, action_id in ordered:
            action_id = int(action_id)
            if action_id <= previous:
                raise ValueError("legal actions must be unique action_ids")
            previous = action_id
            triples.append((_obs, action, action_id))
        ordered_rows.append(ordered)
        row_action_counts.append(len(ordered))
    # Rust 以原生 Observation 批编码 Shared + Analysis 行。
    native = [getattr(obs, "native_observation", obs) for obs in observations]
    encoded = riichienv.prepare_current_state_batch(native)
    rows = np.asarray(encoded.rows, dtype=np.int32)
    numerics = np.asarray(encoded.numeric, dtype=np.float32)
    offsets = np.asarray(encoded.offsets, dtype=np.int64)
    if offsets.shape != (len(decisions) + 1,) or offsets[0] != 0:
        raise RuntimeError("native current-state offsets are malformed")
    lengths = np.diff(offsets)
    if np.any(lengths <= 0):
        raise RuntimeError("native current-state produced an empty decision row")

    encoded_queries = encode_action_queries_batch_native(triples)
    query_rows_all = np.asarray(encoded_queries.query_rows, dtype=np.int32).reshape(-1, QUERY_ROW_WIDTH)
    action_capacity = max(row_action_counts)
    batch = len(decisions)
    query_array = np.zeros((batch, 2 * action_capacity, QUERY_ROW_WIDTH), dtype=np.int32)
    action_ids_array = np.zeros((batch, action_capacity), dtype=np.int32)
    pair_counts = np.zeros(batch, dtype=np.int64)
    actor_factors = np.zeros((batch, max(int(lengths.max()) + 1 + 2 * action_capacity, 1), TOKEN_ROW_WIDTH), dtype=np.int32)
    actor_numeric = np.zeros((batch, actor_factors.shape[1], TOKEN_NUMERIC_WIDTH), dtype=np.float32)
    actor_lengths = np.zeros(batch, dtype=np.int64)

    query_cursor = 0
    for row, (_obs, actions) in enumerate(decisions):
        ordered = ordered_rows[row]
        start, end = int(offsets[row]), int(offsets[row + 1])
        native_rows = rows[start:end]
        native_numeric = numerics[start:end]
        count = len(ordered)
        if end - start + 1 + 2 * count > actor_factors.shape[1]:
            raise RuntimeError("assembled actor sequence exceeds padded capacity")
        # SEP_ACTIONS 行。
        sep = np.zeros(TOKEN_ROW_WIDTH, dtype=np.int32)
        sep[0] = SEGMENT_ACTIONS
        sep[1] = KIND_SEP_ACTIONS
        assembled = np.vstack([native_rows, sep])
        assembled_numeric = np.vstack([native_numeric, np.zeros((1, TOKEN_NUMERIC_WIDTH), dtype=np.float32)])
        per_row_queries = query_rows_all[query_cursor:query_cursor + 2 * count]
        for offset in range(count):
            offense = _action_row(1, per_row_queries[2 * offset], _tsumogiri_mode(int(ordered[offset][1])))
            defense = _action_row(2, per_row_queries[2 * offset + 1], _tsumogiri_mode(int(ordered[offset][1])))
            assembled = np.vstack([assembled, offense[None, :], defense[None, :]])
            assembled_numeric = np.vstack(
                [assembled_numeric, np.zeros((2, TOKEN_NUMERIC_WIDTH), dtype=np.float32)]
            )
        actor_factors[row, : assembled.shape[0]] = assembled
        actor_numeric[row, : assembled.shape[0]] = assembled_numeric
        actor_lengths[row] = assembled.shape[0]
        query_array[row, : 2 * count] = per_row_queries
        action_ids_array[row, :count] = [ordered[index][1] for index in range(count)]
        pair_counts[row] = count
        query_cursor += 2 * count

    legal_mask = np.zeros((batch, NUM_ACTIONS), dtype=np.bool_)
    for row, (_obs, actions) in enumerate(decisions):
        for _action, action_id in ordered_rows[row]:
            legal_mask[row, int(action_id)] = True
    return EncodedStateBatch(
        actor_factors=actor_factors,
        actor_numeric=actor_numeric,
        actor_lengths=actor_lengths,
        query_rows=query_array,
        action_ids=action_ids_array,
        query_pair_counts=pair_counts,
        legal_mask=legal_mask,
    )
