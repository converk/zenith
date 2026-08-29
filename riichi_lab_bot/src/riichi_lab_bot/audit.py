"""V18 在线输入审计记录器。"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np

from .bridge import PreparedDecision


class InputAuditRecorder:
    """把每次线上决策实际送入模型的 V18 输入落到本地 JSONL。"""

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path).expanduser() if path else None

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def emit_request(
        self,
        *,
        request_id: int,
        observation_base64: str,
        possible_actions: list[dict[str, Any]],
        prepared: PreparedDecision,
        model_action_id: int | None,
        inference_ms: float | None,
        selected_payload: dict[str, Any] | None,
        selected_source: str,
        selected_reason: str | None,
    ) -> None:
        if self.path is None:
            return
        observation = prepared.observation
        fields = getattr(observation, "_fields", {})
        record = {
            "event": "input_audit",
            "request_id": int(request_id),
            "seat": int(prepared.seat),
            "observation_base64": observation_base64,
            "missing_fields": sorted(
                str(value) for value in getattr(observation, "missing_fields", ())
            ),
            "rebuilt_fields": _jsonable(fields),
            "new_events": [
                _parse_event(raw) for raw in list(observation.new_events())
            ],
            "event_context": {
                "last_type": prepared.event_context.last_type,
                "actor": prepared.event_context.actor,
                "pai": prepared.event_context.pai,
            },
            "possible_actions": _jsonable(possible_actions),
            "legal_jsons": [json.loads(value) for value in prepared.legal_jsons],
            "selected": {
                "action_id": model_action_id,
                "payload": _jsonable(selected_payload),
                "source": selected_source,
                "reason": selected_reason,
                "inference_ms": inference_ms,
            },
            "actor": {
                "length": int(prepared.actor_length),
                "factors": prepared.actor_factors.tolist(),
                "numeric": _round_float_rows(prepared.actor_numeric),
            },
            "query": {
                "pair_count": int(prepared.query_pair_count),
                "action_ids": prepared.query_action_ids.tolist(),
            },
            "legal_mask": [
                int(index)
                for index, value in enumerate(prepared.legal_mask.tolist())
                if bool(value)
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def _parse_event(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"_raw": str(raw)}
    return value if isinstance(value, dict) else {"_raw": value}


def _jsonable(value: Any) -> Any:
    """递归转换 numpy/Rust 代理常见值,保证审计文件可 JSON 序列化。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return value


def _round_float_rows(array: np.ndarray) -> list[list[float]]:
    return [
        [round(float(value), 8) for value in row]
        for row in np.asarray(array, dtype=np.float32)
    ]
