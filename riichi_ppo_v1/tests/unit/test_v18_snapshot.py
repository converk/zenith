"""V18 当前局面 Rust 编码器的结构与守恒检查。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import riichienv
from riichienv import MjaiReplay

from riichi_ppo_v1.model.encoding_protocol import (
    KIND_OPPONENT_ANALYSIS,
    KIND_RIVER_SUMMARY,
    KIND_SEP_OPPONENT_ANALYSIS,
    KIND_TILE_STATE,
    SEGMENT_ANALYSIS,
    SEGMENT_SHARED,
    TOKEN_ROW_WIDTH,
)


def _first_seat_observations(limit: int = 8) -> list[object]:
    path = Path("RiichiEnv/tests/data/126_204_0_mjai.jsonl")
    replay = MjaiReplay.from_jsonl(str(path))
    kyoku = list(replay.take_kyokus())[0]
    results = []
    for observation, _action in list(kyoku.steps(seat=0, skip_single_action=False)):
        results.append(observation)
        if len(results) >= limit:
            break
    return results


def test_current_state_rows_structure() -> None:
    observations = _first_seat_observations()
    batch = riichienv.prepare_current_state_batch(
        [getattr(obs, "native_observation", obs) for obs in observations]
    )
    rows = np.asarray(batch.rows, dtype=np.int32)
    offsets = np.asarray(batch.offsets, dtype=np.int64)
    assert rows.shape[1] == TOKEN_ROW_WIDTH
    assert offsets.shape == (len(observations) + 1,)
    assert offsets[0] == 0 and offsets[-1] == rows.shape[0]
    for index in range(len(observations)):
        start, end = int(offsets[index]), int(offsets[index + 1])
        chunk = rows[start:end]
        # BOS 开头；shared 段+analysis 段；无 actions/critic。
        assert chunk[0, 1] == 1
        segments = set(chunk[:, 0].tolist())
        assert segments <= {SEGMENT_SHARED, SEGMENT_ANALYSIS}
        # 恰好 34 个 TILE_STATE（升序 1..34）。
        tile_rows = chunk[chunk[:, 1] == KIND_TILE_STATE]
        assert tile_rows.shape[0] == 34
        assert tile_rows[:, 2].tolist() == list(range(1, 35))
        # 每家 river 恰好两个摘要。
        summary_count = int(np.count_nonzero(chunk[:, 1] == KIND_RIVER_SUMMARY))
        assert summary_count == 6
        # 三个 Opponent Analysis + 分隔符。
        analysis_rows = chunk[chunk[:, 1] == KIND_OPPONENT_ANALYSIS]
        assert analysis_rows.shape[0] == 3
        assert analysis_rows[:, 2].tolist() == [1, 2, 3]
        assert int(np.count_nonzero(chunk[:, 1] == KIND_SEP_OPPONENT_ANALYSIS)) == 1
        # 无 tiles_left / 历史 token 字段（仅检查不存在未知 kind）。
        for kind in chunk[:, 1].astype(int):
            assert kind in set(range(1, 15)) | set(range(101, 110))


def test_no_event_history_tokens() -> None:
    observations = _first_seat_observations(1)
    batch = riichienv.prepare_current_state_batch(
        [getattr(obs, "native_observation", obs) for obs in observations]
    )
    rows = np.asarray(batch.rows, dtype=np.int32)
    kinds = rows[:, 1].astype(int)
    # 事件历史 token 在旧协议中是 segment 1 的 10 列 factor 行；新协议每行
    # 都有唯一 kind，且不存在任何 20-99 的 kind（旧事件 kind 区间）。
    assert not np.any((kinds >= 20) & (kinds < 100))


def test_own_river_has_no_discard_tokens() -> None:
    observations = _first_seat_observations(6)
    batch = riichienv.prepare_current_state_batch(
        [getattr(obs, "native_observation", obs) for obs in observations]
    )
    rows = np.asarray(batch.rows, dtype=np.int32)
    # RIVER_DISCARD kind=7 只允许 relative_seat 1..3（本编码器不含自身逐张）。
    discard_rows = rows[rows[:, 1] == 7]
    assert np.all(discard_rows[:, 2] >= 1)
    assert np.all(discard_rows[:, 2] <= 3)
