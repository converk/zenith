"""Structured, secret-safe runtime metrics and complete MJAI event logging."""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# 专用 MJAI 事件日志的默认根目录（阶段 6 约定：logs/v19/bot_mjai/）。
DEFAULT_MJAI_LOG_DIR = "logs/v19/bot_mjai"

# MJAI 对局事件全集（含兼容记法的 kita/nuki）；request_action/action_ack 等
# 传输控制消息不属于 MJAI 事件，不进入专用日志。
MJAI_EVENT_TYPES = frozenset(
    {
        "start_game",
        "start_kyoku",
        "tsumo",
        "dahai",
        "chi",
        "pon",
        "daiminkan",
        "ankan",
        "kakan",
        "nuki",
        "kita",
        "dora",
        "reach",
        "reach_accepted",
        "hora",
        "ryukyoku",
        "end_kyoku",
        "end_game",
    }
)


def is_mjai_event(message: dict[str, Any]) -> bool:
    """判断服务端消息是否为 MJAI 对局事件原文。"""
    return isinstance(message, dict) and str(message.get("type", "")) in MJAI_EVENT_TYPES


@dataclass
class SessionMetrics:
    requests: int = 0
    responses: int = 0
    model_actions: int = 0
    fallback_actions: int = 0
    withheld_actions: int = 0
    accepted: int = 0
    rejected: int = 0
    unparseable: int = 0
    stale: int = 0
    defaulted: int = 0
    bank_consumed_ms: int = 0
    inference_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        result = {
            key: value
            for key, value in asdict(self).items()
            if key != "inference_ms"
        }
        values = sorted(self.inference_ms)
        if values:
            result.update(
                {
                    "inference_count": len(values),
                    "inference_mean_ms": statistics.fmean(values),
                    "inference_p50_ms": _percentile(values, 50),
                    "inference_p95_ms": _percentile(values, 95),
                    "inference_max_ms": values[-1],
                }
            )
        else:
            result["inference_count"] = 0
        return result


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _safe_name(value: object, fallback: str) -> str:
    """把 session/game_id 归一化为文件名安全片段。"""
    text = str(value)
    sanitized = re.sub(r"[^\w\-]+", "_", text).strip("_")
    return sanitized or fallback


def _resolve_game_id(event: dict[str, Any], game_no: int) -> str:
    """从 start_game 事件提取 game_id；RiichiLab 的 start_game.id 是座位号，不回退用作局号。

    事件未携带真实局号时使用会话内递增的 game_no，保证文件名唯一。
    """
    candidate = event.get("game_id")
    if isinstance(candidate, (str, int)) and str(candidate):
        return str(candidate)
    maybe_id = event.get("id")
    if isinstance(maybe_id, str) and maybe_id:
        return maybe_id
    if isinstance(maybe_id, int) and not 0 <= maybe_id < 4:
        return str(maybe_id)
    return str(game_no)


def _resolve_seat(event: dict[str, Any], seat: Any) -> Any:
    """解析 wrapper 中的 seat：显式传入优先，其次事件 id/seat 字段。"""
    if seat is not None:
        return seat
    for key in ("id", "seat"):
        value = event.get(key)
        if isinstance(value, int) and 0 <= value < 4:
            return value
    return None


class MjaiEventLogger:
    """完整 MJAI 事件原文日志（与 EventRecorder 元事件分离）。

    每个 MJAI 事件写一行 JSONL，行结构:
    ``{"log_no": ..., "game_no": ..., "seat": ..., "timestamp": ..., "event": <原始dict>}``

    - 文件:``logs/v19/bot_mjai/<session>-<game_id>-<yyyymmdd_hhmmss>.jsonl``；
    - 从 start_game 起按收到的顺序逐行写；写失败只记录错误并继续，绝不中断对局。
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        session: str = "bot",
    ) -> None:
        self.root_dir = Path(root_dir).expanduser() if root_dir else None
        self.session = _safe_name(session, "bot")
        # 会话内局号：每个 start_game +1；game_id 缺省时复用该编号。
        self.game_no = 0
        self.current_path: Path | None = None
        self.current_game_id: str | None = None
        self.current_seat: Any = None
        self.current_log_no = 0
        self.write_errors = 0

    @property
    def enabled(self) -> bool:
        return self.root_dir is not None

    def _begin_game(
        self,
        event: dict[str, Any],
        *,
        seat: Any = None,
        game_id: str | None = None,
    ) -> None:
        """开新局文件；start_game 本身会成为该文件第一行（由随后 record 写）。"""
        if not self.enabled:
            return
        self.game_no += 1
        resolved_id = (
            _safe_name(game_id, str(self.game_no))
            if game_id is not None
            else _safe_name(_resolve_game_id(event, self.game_no), str(self.game_no))
        )
        self.current_game_id = resolved_id
        self.current_seat = _resolve_seat(event, seat)
        self.current_log_no = 0
        self.root_dir = self.root_dir or Path(DEFAULT_MJAI_LOG_DIR)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_path = (
            self.root_dir
            / f"{self.session}-{resolved_id}-{stamp}.jsonl"
        )

    def record(
        self,
        event: dict[str, Any],
        *,
        seat: Any = None,
        game_id: str | None = None,
    ) -> bool:
        """记录一条 MJAI 事件原文；失败记录日志并返回 False，不向对局抛异常。"""
        if not self.enabled or not isinstance(event, dict):
            return False
        try:
            # start_game 开启新局文件；收到非 start_game 且尚无文件时也先开一局。
            if event.get("type") == "start_game" or self.current_path is None:
                self._begin_game(event, seat=seat, game_id=game_id)
            if self.current_path is None:
                return False
            self.current_log_no += 1
            payload = {
                "log_no": self.current_log_no,
                "game_no": self.game_no,
                "seat": self.current_seat,
                "timestamp": time.time(),
                "event": event,
            }
            with self.current_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        except OSError as exc:
            self.write_errors += 1
            logging.getLogger("riichi_lab_bot").error(
                "MJAI event log write failed (%s): %s",
                self.current_path,
                exc,
            )
            return False
        return True

    def record_many(
        self,
        events: list[dict[str, Any]],
        *,
        seat: Any = None,
        game_id: str | None = None,
    ) -> int:
        """一次写入一组事件（本地对局用 env.mjai_log 的全局完整流）。"""
        written = 0
        for event in events:
            if self.record(event, seat=seat, game_id=game_id):
                written += 1
        return written

    def close(self) -> None:
        """关闭当前局文件句柄（本次实现为每行独立打开，close 仅清引用）。"""
        self.current_path = None


class EventRecorder:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path).expanduser() if path else None

    def emit(self, event: str, **fields: Any) -> None:
        payload = {
            "timestamp": time.time(),
            "event": event,
            **fields,
        }
        logging.getLogger("riichi_lab_bot").info(
            "%s", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )


def load_mjai_log_events(path: str | Path) -> list[dict[str, Any]]:
    """读取专用 MJAI JSONL，返回原始事件 dict 列表。

    支持 wrapper 行（本 logger 产物）与裸事件行两种格式。
    """
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed MJAI log line {line_number} at {path}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"MJAI log line {line_number} must be a JSON object"
            )
        event = value.get("event", value)
        if not isinstance(event, dict):
            raise ValueError(
                f"MJAI log line {line_number} has no event object"
            )
        records.append(event)
    return records


def _rank_scores(scores: list[int]) -> list[int]:
    """按分数降序给顺位（并列同顺位，1 为最高）。"""
    result: list[int] = []
    for score in scores:
        result.append(1 + sum(1 for other in scores if other > score))
    return result


def replay_mjai_log(
    path: str | Path,
    *,
    rule: str = "tenhou",
) -> dict[str, Any]:
    """用 riichienv.MjaiReplay 完整回放专用 MJAI JSONL，重建终局与基础指标。

    返回终局分数、顺位、各家和牌数、流局数、局数与逐局分数迁移。
    """
    events = load_mjai_log_events(path)
    if not events:
        raise ValueError(f"MJAI log is empty: {path}")
    stream = "\n".join(
        json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        for event in events
    )
    try:
        from riichienv import MjaiReplay

        replay = MjaiReplay.from_jsonl_string(stream, rule=rule)
    except Exception as exc:  # noqa: BLE001 -- 回放器任何拒绝都归一为可读错误,不吞原因
        raise ValueError(f"MJAI replay failed for {path}: {exc}") from exc
    kyokus = list(replay.take_kyokus())
    if not kyokus:
        raise ValueError(f"MJAI log contains no kyoku: {path}")
    round_features = [kyoku.grp_features() for kyoku in kyokus]
    final_scores = [int(value) for value in round_features[-1]["end_scores"]]
    hora_counts = [0] * 4
    type_counts: Counter[str] = Counter()
    for event in events:
        event_type = str(event.get("type", ""))
        type_counts[event_type] += 1
        if event_type == "hora":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                hora_counts[actor] += 1
    return {
        "path": str(Path(path).resolve()),
        "events": len(events),
        "start_game_count": int(type_counts.get("start_game", 0)),
        "rounds": len(kyokus),
        "final_scores": final_scores,
        "ranks": _rank_scores(final_scores),
        "hora_counts": hora_counts,
        "ryukyoku_count": int(type_counts.get("ryukyoku", 0)),
        "end_game_count": int(type_counts.get("end_game", 0)),
        "round_details": [
            {
                "scores": [int(value) for value in features["scores"]],
                "end_scores": [int(value) for value in features["end_scores"]],
                "delta_scores": [
                    int(value) for value in features["delta_scores"]
                ],
            }
            for features in round_features
        ],
    }
