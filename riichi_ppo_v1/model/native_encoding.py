"""现行动作 Query 的唯一 Rust 融合编码边界。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import riichi
import riichienv

from .encoding_protocol import QUERY_ROW_ANSWER_START


@dataclass(frozen=True)
class NativeQueryBatch:
    """逐动作连续 query 行及融合内核的去重统计。"""

    query_rows: np.ndarray
    unique_offense_rows: int
    unique_shanten_rows: int


@dataclass(frozen=True)
class _NativeBatchInputs:
    observations: list[object]
    observation_indices: np.ndarray
    actions: list[object]
    action_ids: np.ndarray
    missed_doujun: np.ndarray
    missed_riichi: np.ndarray
    riichi_declared: np.ndarray
    drawn_tiles: np.ndarray


def _native_batch_inputs(
    rows: list[tuple[object, object, int]],
) -> _NativeBatchInputs:
    """只规范化 PyO3 边界;牌形与动作语义不在 Python 计算。"""
    observations: list[object] = []
    observation_lookup: dict[int, int] = {}
    observation_indices = np.empty(len(rows), dtype=np.uint32)
    actions: list[object] = []
    action_ids = np.empty(len(rows), dtype=np.uint16)
    missed_doujun: list[bool] = []
    missed_riichi: list[bool] = []
    riichi_declared: list[bool] = []
    drawn_tiles: list[int] = []
    for row, (observation, action, action_id) in enumerate(rows):
        key = id(observation)
        index = observation_lookup.get(key)
        if index is None:
            index = len(observations)
            observation_lookup[key] = index
            native = getattr(observation, "native_observation", observation)
            observations.append(native)
            seat = int(observation.player_id)
            missed_doujun.append(bool(observation.missed_agari_doujun))
            missed_riichi.append(bool(observation.missed_agari_riichi))
            riichi_declared.append(bool(observation.riichi_declared[seat]))
            drawn = getattr(observation, "drawn_tile", None)
            drawn_tiles.append(-1 if drawn is None else int(drawn))
        observation_indices[row] = index
        actions.append(action)
        action_ids[row] = int(action_id)
    return _NativeBatchInputs(
        observations=observations,
        observation_indices=observation_indices,
        actions=actions,
        action_ids=action_ids,
        missed_doujun=np.asarray(missed_doujun, dtype=np.bool_),
        missed_riichi=np.asarray(missed_riichi, dtype=np.bool_),
        riichi_declared=np.asarray(riichi_declared, dtype=np.bool_),
        drawn_tiles=np.asarray(drawn_tiles, dtype=np.int16),
    )


def encode_action_queries_batch_native(
    rows: list[tuple[object, object, int]],
) -> NativeQueryBatch:
    """Observation/Action 只各跨一次 PyO3,其余语义全部由 Rust 计算。"""
    if not rows:
        return NativeQueryBatch(np.zeros((0, 2, 15), dtype=np.int32), 0, 0)
    inputs = _native_batch_inputs(rows)
    facts = riichienv.prepare_encoding_facts(
        inputs.observations,
        inputs.observation_indices,
        inputs.actions,
        inputs.action_ids,
        inputs.missed_doujun,
        inputs.missed_riichi,
        inputs.riichi_declared,
        inputs.drawn_tiles,
    )
    encoded = riichi.encode_query_batch(
        facts.action_ids,
        facts.action_types,
        facts.primary_types,
        facts.source_seats,
        facts.modes,
        facts.shape_counts,
        facts.open_melds,
        facts.remaining,
        facts.own_rivers,
        facts.opponent_rivers,
        facts.defense_counts,
        facts.discard_types,
        facts.defense_visible,
        facts.missed_doujun,
        facts.missed_riichi,
        facts.riichi_declared,
        facts.scores,
        facts.o7_values,
        facts.o8_values,
        facts.o9_values,
    )
    query_rows = np.asarray(encoded.query_rows, dtype=np.int32)
    wait_masks = np.asarray(encoded.wait_masks, dtype=np.uint64)
    if np.any(wait_masks):
        yaku = riichienv.analyze_encoding_yaku_batch(
            inputs.observations,
            inputs.observation_indices,
            inputs.actions,
            wait_masks,
            inputs.drawn_tiles,
        )
        wait_rows = np.flatnonzero(wait_masks)
        query_rows[wait_rows, 0, QUERY_ROW_ANSWER_START + 4] = np.asarray(
            yaku.yaku_class, dtype=np.uint8,
        )[wait_rows]
        query_rows[wait_rows, 0, QUERY_ROW_ANSWER_START + 5] = np.asarray(
            yaku.base_han, dtype=np.uint8,
        )[wait_rows]
    return NativeQueryBatch(
        query_rows=query_rows,
        unique_offense_rows=int(encoded.unique_offense_rows),
        unique_shanten_rows=int(encoded.unique_shanten_rows),
    )
