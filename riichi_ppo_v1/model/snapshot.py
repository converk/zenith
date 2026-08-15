"""V16 Compact Snapshot 事实层:基础场况 + Score Pressure + 3×7 对手摘要。

本模块只产出可从当前观察者可见的公开 MJAI 状态确定的事实;对手 7 项摘要的归一化
由 state-machine(`riichi.public_opponent_summary`)完成,模型输入转换侧只做读取
与组装。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import riichi

from .encoding_protocol import (
    SNAPSHOT_HONBA_SCALE,
    SNAPSHOT_MELD_SCALE,
    SNAPSHOT_PRESSURE_SCALE,
    SNAPSHOT_RIVER_SCALE,
    SNAPSHOT_SCORE_SCALE,
    SNAPSHOT_STICKS_SCALE,
    SNAPSHOT_TILES_LEFT_SCALE,
    SNAPSHOT_TURN_SCALE,
    SNAPSHOT_KIND_BASE,
    SNAPSHOT_KIND_DORA,
    SNAPSHOT_KIND_SCORE,
    SNAPSHOT_KIND_SUMMARY,
    SNAPSHOT_CAT_WIDTH,
    SNAPSHOT_NUM_WIDTH,
)


@dataclass(frozen=True)
class OpponentSummary:
    declared: int
    reach_turn: int  # 立直巡目,-1 表示 N/A(输出编码 255)
    meld_count: int
    menzen: int  # 1=门清,0=副露
    river_count: int
    tedashi_count: int
    tsumogiri_count: int


@dataclass(frozen=True)
class SnapshotFacts:
    round_wind: int
    kyoku_index: int
    oya_relative: int
    honba: int
    riichi_sticks: int
    tiles_left: int
    dora_indicator_types: tuple[int, ...]
    scores: tuple[int, ...]
    self_rank: int
    score_pressure: tuple[int, int, int]
    opponent_summary: tuple[OpponentSummary, OpponentSummary, OpponentSummary]


def _required(observation: object, name: str) -> object:
    value = getattr(observation, name, None)
    if value is None:
        raise RuntimeError(f"V16 snapshot requires observation.{name}")
    return value


def build_snapshot_facts(observation: object) -> SnapshotFacts:
    """从单个观察者 Observation 组装 Snapshot 事实。"""
    seat = int(observation.player_id)
    scores = tuple(int(value) for value in _required(observation, "scores"))
    order = sorted(range(4), key=lambda player: (-scores[player], player))
    self_rank = order.index(seat) + 1

    opponents = [(seat + offset) % 4 for offset in (1, 2, 3)]
    declared = [bool(value) for value in _required(observation, "riichi_declared")]
    declaration_indices = _required(observation, "riichi_declaration_indices")
    melds = _required(observation, "melds")
    discards = _required(observation, "discards")
    tsumogiri_flags = _required(observation, "tsumogiri_flags")

    reach_turns: list[int] = []
    meld_counts: list[int] = []
    river_counts: list[int] = []
    tedashi_counts: list[int] = []
    tsumogiri_counts: list[int] = []
    for opponent in opponents:
        index = declaration_indices[opponent]
        reach_turns.append(-1 if index is None else int(index))
        meld_counts.append(len(melds[opponent]))
        river_counts.append(len(discards[opponent]))
        flags = list(tsumogiri_flags[opponent])
        tsumogiri_counts.append(sum(bool(flag) for flag in flags))
        tedashi_counts.append(sum(not bool(flag) for flag in flags))

    summary_rows = riichi.public_opponent_summary(
        np.asarray([int(declared[opponent]) for opponent in opponents], dtype=np.uint8),
        np.asarray(reach_turns, dtype=np.int16),
        np.asarray(meld_counts, dtype=np.uint8),
        np.asarray(river_counts, dtype=np.uint8),
        np.asarray(tedashi_counts, dtype=np.uint8),
        np.asarray(tsumogiri_counts, dtype=np.uint8),
    )
    summaries = tuple(
        OpponentSummary(
            declared=int(row[0]),
            reach_turn=-1 if int(row[1]) == 255 else int(row[1]),
            meld_count=int(row[2]),
            menzen=int(row[3]),
            river_count=int(row[4]),
            tedashi_count=int(row[5]),
            tsumogiri_count=int(row[6]),
        )
        for row in summary_rows
    )

    return SnapshotFacts(
        round_wind=int(_required(observation, "round_wind")),
        kyoku_index=int(_required(observation, "kyoku_index")),
        oya_relative=(int(_required(observation, "oya")) - seat) % 4,
        honba=int(_required(observation, "honba")),
        riichi_sticks=int(_required(observation, "riichi_sticks")),
        tiles_left=int(_required(observation, "tiles_left")),
        dora_indicator_types=tuple(
            int(tile) // 4 for tile in _required(observation, "dora_indicators")
        ),
        scores=scores,
        self_rank=self_rank,
        score_pressure=tuple(scores[seat] - scores[opponent] for opponent in opponents),
        opponent_summary=summaries,
    )


def _clip_unit(value: float, scale: float) -> float:
    """非负计数的 [0,1] 固定归一化。"""
    return float(np.clip(float(value) / scale, 0.0, 1.0))


def _clip_signed(value: float, scale: float) -> float:
    """点数/分差的 [-5,5] 固定归一化。"""
    return float(np.clip(float(value) / scale, -5.0, 5.0))


def encode_snapshot_rows(
    facts: SnapshotFacts,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 SnapshotFacts 编码为统一行格式:(kinds, categorical, numeric)。

    返回 ``kinds`` [S]、``categorical`` [S,4] uint8、``numeric`` [S,7] float32;
    行顺序固定为 base(1)+ dora(k)+ score(1)+ summary(21),连续字段在此一次性
    应用固定归一化刻度,训练/推理不再动态缩放。
    """
    kinds: list[int] = [SNAPSHOT_KIND_BASE]
    categorical: list[list[int]] = [[
        min(max(int(facts.round_wind), 0), 1),
        min(max(int(facts.kyoku_index), 0), 7),
        int(facts.oya_relative),
        max(0, int(facts.self_rank) - 1),
    ]]
    numeric: list[list[float]] = [[
        _clip_unit(facts.honba, SNAPSHOT_HONBA_SCALE),
        _clip_unit(facts.riichi_sticks, SNAPSHOT_STICKS_SCALE),
        _clip_unit(facts.tiles_left, SNAPSHOT_TILES_LEFT_SCALE),
        0.0, 0.0, 0.0, 0.0,
    ]]
    for tile_type in facts.dora_indicator_types:
        kinds.append(SNAPSHOT_KIND_DORA)
        categorical.append([min(max(int(tile_type), 0), 33), 0, 0, 0])
        numeric.append([0.0] * SNAPSHOT_NUM_WIDTH)
    kinds.append(SNAPSHOT_KIND_SCORE)
    categorical.append([0, 0, 0, 0])
    numeric.append([
        *(_clip_signed(score, SNAPSHOT_SCORE_SCALE) for score in facts.scores),
        *(_clip_signed(pressure, SNAPSHOT_PRESSURE_SCALE) for pressure in facts.score_pressure),
    ])
    for summary in facts.opponent_summary:
        kinds.append(SNAPSHOT_KIND_SUMMARY)
        categorical.append([int(summary.declared), int(summary.menzen), 0, 0])
        reach_turn = 0.0 if summary.reach_turn < 0 else float(summary.reach_turn)
        numeric.append([
            _clip_unit(reach_turn, SNAPSHOT_TURN_SCALE),
            _clip_unit(summary.meld_count, SNAPSHOT_MELD_SCALE),
            _clip_unit(summary.river_count, SNAPSHOT_RIVER_SCALE),
            _clip_unit(summary.tedashi_count, SNAPSHOT_RIVER_SCALE),
            _clip_unit(summary.tsumogiri_count, SNAPSHOT_RIVER_SCALE),
            0.0, 0.0,
        ])
    kinds_array = np.asarray(kinds, dtype=np.uint8)
    categorical_array = np.zeros((len(kinds), SNAPSHOT_CAT_WIDTH), dtype=np.uint8)
    for row, values in enumerate(categorical):
        categorical_array[row, :len(values)] = values
    numeric_array = np.zeros((len(kinds), SNAPSHOT_NUM_WIDTH), dtype=np.float32)
    for row, values in enumerate(numeric):
        numeric_array[row, :len(values)] = values
    return kinds_array, categorical_array, numeric_array
