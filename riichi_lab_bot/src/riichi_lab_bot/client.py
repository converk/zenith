"""RiichiLab WebSocket validation and ranked clients."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import time
from typing import Any

from .audit import InputAuditRecorder
from .bridge import OnlineStateBridge
from .observation import (
    ObservationView,
    missing_observation_fields,
    normalize_observation_base64,
)
from .policy import PolicyEngine
from .safety import choose_safe_response
from .telemetry import EventRecorder, SessionMetrics

VALIDATION_URL = "wss://game.riichi.dev/ws/validate"
RANKED_URL = "wss://game.riichi.dev/ws/ranked"


class AuthenticationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionResult:
    completed: bool
    validation_passed: bool | None
    end_scores: tuple[int, ...] | None
    metrics: dict[str, Any]


def _deserialize_observation(encoded: str) -> Any:
    try:
        from riichienv import Observation
    except ImportError as exc:
        raise RuntimeError(
            "the local riichienv extension is not installed"
        ) from exc
    observation = Observation.deserialize_from_base64(
        normalize_observation_base64(encoded)
    )
    return ObservationView(
        observation,
        missing_fields=missing_observation_fields(encoded),
    )


async def play_connection(
    *,
    url: str,
    mode: str,
    token: str,
    policy: PolicyEngine,
    recorder: EventRecorder,
    audit_recorder: InputAuditRecorder | None = None,
    deadline_margin_ms: int = 250,
) -> SessionResult:
    from websockets.asyncio.client import connect

    if mode not in {"validate", "ranked"}:
        raise ValueError("mode must be validate or ranked")
    metrics = SessionMetrics()
    bridge: OnlineStateBridge | None = None
    seat: int | None = None
    validation_passed: bool | None = None
    end_scores: tuple[int, ...] | None = None
    completed = False
    recorder.emit("connection_opening", mode=mode, url=url)

    async with connect(
        url,
        additional_headers={"Authorization": f"Bearer {token}"},
        open_timeout=15,
        ping_interval=20,
        ping_timeout=40,
        close_timeout=10,
        max_size=4 * 1024 * 1024,
    ) as websocket:
        while True:
            if (
                mode == "validate"
                and end_scores is not None
                and validation_passed is None
            ):
                try:
                    raw = await asyncio.wait_for(
                        websocket.recv(), timeout=10.0
                    )
                except asyncio.TimeoutError:
                    recorder.emit(
                        "validation_result_timeout", scores=end_scores
                    )
                    break
            else:
                try:
                    raw = await websocket.recv()
                except Exception:
                    if completed:
                        break
                    raise
            received_at = time.perf_counter()
            if isinstance(raw, bytes):
                recorder.emit("binary_frame_ignored", size=len(raw))
                continue
            try:
                message = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                recorder.emit("malformed_server_json_ignored")
                continue
            if not isinstance(message, dict):
                recorder.emit("non_object_server_json_ignored")
                continue
            if "error" in message:
                error = str(message["error"])
                recorder.emit("server_error", message=error)
                if "token" in error.lower() or "inactive" in error.lower():
                    raise AuthenticationError(error)
                raise RuntimeError(error)

            message_type = message.get("type")
            if message_type == "start_game":
                value = message.get("id", 0)
                if not isinstance(value, int) or not 0 <= value < 4:
                    raise RuntimeError(f"invalid start_game seat: {value!r}")
                seat = value
                bridge = OnlineStateBridge(seat)
                recorder.emit("game_started", mode=mode, seat=seat)
                continue

            if message_type == "request_action":
                metrics.requests += 1
                request_id = message.get("request_id")
                encoded = message.get("observation")
                possible = message.get("possible_actions")
                if (
                    not isinstance(request_id, int)
                    or not isinstance(encoded, str)
                    or not isinstance(possible, list)
                    or bridge is None
                    or seat is None
                ):
                    metrics.withheld_actions += 1
                    recorder.emit(
                        "response_withheld",
                        request_id=request_id,
                        reason="malformed request_action or missing start_game",
                    )
                    continue
                possible_actions = [
                    item for item in possible if isinstance(item, dict)
                ]
                primary_action = None
                prepared = None
                primary_error: str | None = None
                inference_ms: float | None = None
                model_action_id: int | None = None
                try:
                    observation = _deserialize_observation(encoded)
                    if int(observation.player_id) != seat:
                        raise ValueError(
                            "request observation seat does not match start_game"
                        )
                    prepared = bridge.prepare(observation)
                    inference = policy.infer(prepared)
                    inference_ms = inference.elapsed_ms
                    model_action_id = inference.action_id
                    metrics.inference_ms.append(inference.elapsed_ms)
                    primary_action = bridge.decode(
                        prepared, inference.action_id
                    )
                except Exception as exc:
                    primary_error = f"{type(exc).__name__}: {exc}"

                if prepared is None:
                    metrics.withheld_actions += 1
                    recorder.emit(
                        "response_withheld",
                        request_id=request_id,
                        reason=primary_error,
                    )
                    continue
                safe = choose_safe_response(
                    prepared,
                    primary_action,
                    possible_actions,
                    request_id,
                    primary_error=primary_error,
                )
                elapsed_ms = (time.perf_counter() - received_at) * 1000.0
                time_info = message.get("time")
                deadline_ms = (
                    time_info.get("deadline_ms")
                    if isinstance(time_info, dict)
                    else None
                )
                if (
                    isinstance(deadline_ms, (int, float))
                    and elapsed_ms + deadline_margin_ms >= float(deadline_ms)
                ):
                    safe = type(safe)(
                        None,
                        "withheld",
                        "local elapsed time reached deadline margin",
                    )
                if safe.payload is None:
                    if audit_recorder is not None:
                        audit_recorder.emit_request(
                            request_id=request_id,
                            observation_base64=encoded,
                            possible_actions=possible_actions,
                            prepared=prepared,
                            model_action_id=model_action_id,
                            inference_ms=inference_ms,
                            selected_payload=None,
                            selected_source=safe.source,
                            selected_reason=safe.reason,
                        )
                    metrics.withheld_actions += 1
                    recorder.emit(
                        "response_withheld",
                        request_id=request_id,
                        elapsed_ms=elapsed_ms,
                        inference_ms=inference_ms,
                        reason=safe.reason,
                    )
                    continue
                if audit_recorder is not None:
                    audit_recorder.emit_request(
                        request_id=request_id,
                        observation_base64=encoded,
                        possible_actions=possible_actions,
                        prepared=prepared,
                        model_action_id=model_action_id,
                        inference_ms=inference_ms,
                        selected_payload=safe.payload,
                        selected_source=safe.source,
                        selected_reason=safe.reason,
                    )
                bridge.record_response(prepared, safe.payload)
                await websocket.send(
                    json.dumps(
                        safe.payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                metrics.responses += 1
                if safe.source == "model":
                    metrics.model_actions += 1
                else:
                    metrics.fallback_actions += 1
                recorder.emit(
                    "action_sent",
                    request_id=request_id,
                    action_id=model_action_id,
                    action_type=safe.payload.get("type"),
                    source=safe.source,
                    elapsed_ms=elapsed_ms,
                    inference_ms=inference_ms,
                )
                continue

            if message_type == "action_ack":
                status = str(message.get("status", ""))
                if hasattr(metrics, status):
                    setattr(metrics, status, getattr(metrics, status) + 1)
                bank = message.get("bank_consumed_ms")
                if isinstance(bank, int):
                    metrics.bank_consumed_ms += bank
                recorder.emit(
                    "action_ack",
                    request_id=message.get("request_id"),
                    status=status,
                    elapsed_ms=message.get("elapsed_ms"),
                    bank_ms=message.get("bank_ms"),
                    reason=message.get("reason"),
                )
                continue

            if message_type == "end_game":
                scores = message.get("scores")
                if isinstance(scores, list) and all(
                    isinstance(value, int) for value in scores
                ):
                    end_scores = tuple(scores)
                completed = True
                recorder.emit(
                    "game_ended",
                    mode=mode,
                    scores=end_scores,
                    metrics=metrics.summary(),
                )
                if mode == "ranked":
                    break
                continue

            if message_type == "validation_result":
                validation_passed = bool(message.get("passed", False))
                recorder.emit(
                    "validation_result",
                    passed=validation_passed,
                    reason=message.get("reason"),
                )
                completed = completed or validation_passed
                break

            recorder.emit(
                "server_event_ignored", event_type=str(message_type)
            )

    return SessionResult(
        completed,
        validation_passed,
        end_scores,
        metrics.summary(),
    )


async def run_ranked(
    *,
    url: str,
    token: str,
    policy: PolicyEngine,
    recorder: EventRecorder,
    games: int | None,
) -> list[SessionResult]:
    results: list[SessionResult] = []
    backoff = 5.0
    while games is None or len(results) < games:
        try:
            result = await play_connection(
                url=url,
                mode="ranked",
                token=token,
                policy=policy,
                recorder=recorder,
            )
            if result.completed:
                results.append(result)
                backoff = 5.0
                if games is None or len(results) < games:
                    await asyncio.sleep(2.0)
            else:
                raise RuntimeError("ranked connection ended before end_game")
        except AuthenticationError:
            raise
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:
            recorder.emit(
                "ranked_reconnect_wait",
                delay_seconds=backoff,
                error=f"{type(exc).__name__}: {exc}",
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 120.0)
    return results
