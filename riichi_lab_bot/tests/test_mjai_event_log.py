from __future__ import annotations

import json

from conftest import repository_root

from riichi_lab_bot.telemetry import (
    MjaiEventLogger,
    load_mjai_log_events,
    replay_mjai_log,
)


def _real_mjai_events() -> list[dict]:
    path = repository_root() / "RiichiEnv" / "tests" / "data" / "126_204_0_mjai.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_logger_writes_wrapped_jsonl_and_replays_real_game(tmp_path) -> None:
    events = _real_mjai_events()
    logger = MjaiEventLogger(tmp_path, session="unit")
    written = logger.record_many(events, seat=0, game_id="fixture")
    assert written == len(events)

    files = list(tmp_path.glob("unit-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(events)
    first = json.loads(lines[0])
    assert first["log_no"] == 1
    assert first["game_no"] == 1
    assert first["seat"] == 0
    assert first["event"]["type"] == "start_game"
    last = json.loads(lines[-1])
    assert last["event"]["type"] == "end_game"

    # 回读 wrapper 得到原始事件序列。
    assert load_mjai_log_events(files[0]) == events

    # 完整回放:重建终局分数/顺位/和牌数/流局数等基础指标。
    result = replay_mjai_log(files[0])
    assert result["events"] == len(events)
    assert result["rounds"] == 12
    assert result["start_game_count"] == 1
    assert result["end_game_count"] == 1
    assert result["final_scores"] == [2400, 14900, 38800, 43900]
    assert result["ranks"] == [4, 3, 2, 1]
    assert result["hora_counts"] == [1, 0, 4, 4]
    assert result["ryukyoku_count"] == 3


def test_logger_write_failure_is_swallowed_and_continues(tmp_path) -> None:
    # 根目录位置被文件占用,目录创建必然失败;logger 必须记错并返回 False,
    # 不能把异常抛给对局循环。
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    logger = MjaiEventLogger(blocker / "sub", session="fail")
    assert logger.record({"type": "start_game", "id": 0}) is False
    assert logger.write_errors == 1
