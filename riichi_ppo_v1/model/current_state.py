"""V18 当前局面批编码的 Python 装配层。

Rust/PyO3 编码器（``riichienv.prepare_current_state_batch``）直接以原生
Observation 当前字段构造 Shared 公共前缀 + 三个 Opponent Analysis 的扁平行；
``riichienv.assemble_current_state_batch`` 进一步完成 SEP/O-D 行穿插、query
元数据与合法掩码的整批物化。本模块只负责：

1. 按 action ID 升序规范排序（调用方提供的 triples 本身即升序，这里再次校验）；
2. 调用既有 ``encode_action_queries_batch_native`` 生成 O/D Query 行（15 宽）；
3. 输出模型/训练入口统一的 ``EncodedStateBatch``。

同一局面的 Observation/replay、PyO3、precompute、shard、collator 均走本模块保证一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import riichienv

from .encoding_protocol import QUERY_ROW_WIDTH
from .native_encoding import encode_action_queries_batch_native


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


def encode_batch(
    decisions: list[tuple[object, list[tuple[object, int]]]],
) -> EncodedStateBatch:
    """一次编码一批决策。

    ``decisions`` 每项为 ``(observation, [(action, action_id), ...])``，
    合法动作列表必须已按 action_id 升序、每个 action_id 唯一。

    Shared/Analysis 行与 Action Query 行均由 Rust 编码；本轮把逐决策的行间
    装配（SEP、O/D 行穿插、query 元数据与合法掩码）也下沉到 Rust
    ``assemble_current_state_batch``，消除 Python 侧逐决策 numpy 分配与拷贝。
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
        for action, raw_action_id in ordered:
            action_id = int(raw_action_id)
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
    query_rows_all = np.asarray(encoded_queries.query_rows, dtype=np.int32).reshape(
        -1, QUERY_ROW_WIDTH
    )
    action_ids_flat = np.asarray(
        [int(action_id) for _obs, _action, action_id in triples], dtype=np.int32
    )
    pair_counts = np.asarray(row_action_counts, dtype=np.int64)
    assembled = riichienv.assemble_current_state_batch(
        rows,
        numerics,
        offsets,
        query_rows_all,
        action_ids_flat,
        pair_counts,
        None,
    )
    return EncodedStateBatch(
        actor_factors=np.asarray(assembled.actor_factors, dtype=np.int32),
        actor_numeric=np.asarray(assembled.actor_numeric, dtype=np.float32),
        actor_lengths=np.asarray(assembled.actor_lengths, dtype=np.int64),
        query_rows=np.asarray(assembled.query_rows, dtype=np.int32),
        action_ids=np.asarray(assembled.action_ids, dtype=np.int32),
        query_pair_counts=np.asarray(assembled.query_pair_counts, dtype=np.int64),
        legal_mask=np.asarray(assembled.legal_mask, dtype=np.bool_),
    )
