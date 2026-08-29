from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
import pytest
from riichi_lab_bot.bridge import action_jsons_and_decision_flag
from riichi_lab_bot.client import (
    AuthenticationError,
    SessionResult,
    play_connection,
    run_ranked,
)
from riichi_lab_bot.policy import InferenceResult
from riichi_lab_bot.telemetry import EventRecorder


class FirstLegalPolicy:
    def infer(self, prepared) -> InferenceResult:
        action_id = int(np.flatnonzero(prepared.legal_mask)[0])
        return InferenceResult(action_id, 0.25)


class FakeWebSocket:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = list(messages)
        self.sent: list[dict[str, Any]] = []

    async def recv(self) -> Any:
        if not self.messages:
            raise RuntimeError("mock message stream exhausted")
        return self.messages.pop(0)

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


class FakeConnect:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        websocket = self.websocket

        class Context:
            async def __aenter__(self):
                return websocket

            async def __aexit__(self, *exc):
                return False

        return Context()


def _request_fixture(
    request_id: int = 42, seat: int = 0
) -> tuple[str, list[dict]]:
    from riichienv import RiichiEnv

    observation = RiichiEnv(
        game_mode="4p-red-half", seed=42
    ).reset()[seat]
    legal, _flag = action_jsons_and_decision_flag(observation)
    message = {
        "type": "request_action",
        "request_id": request_id,
        "time": {
            "grace_ms": 3000,
            "bank_ms": 15000,
            "deadline_ms": 18000,
        },
        "possible_actions": [json.loads(value) for value in legal],
        "observation": observation.serialize_to_base64(),
    }
    return json.dumps(message), message["possible_actions"]


def test_deadline_margin_withholds_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _possible = _request_fixture()
    value = json.loads(request)
    value["time"]["deadline_ms"] = 0
    websocket = FakeWebSocket(
        [
            json.dumps({"type": "start_game", "id": 0}),
            json.dumps(value),
            json.dumps(
                {"type": "end_game", "scores": [25000] * 4}
            ),
        ]
    )
    monkeypatch.setattr(
        "websockets.asyncio.client.connect", FakeConnect(websocket)
    )
    result = asyncio.run(
        play_connection(
            url="wss://example.invalid/ranked",
            mode="ranked",
            token="secret",
            policy=FirstLegalPolicy(),
            recorder=EventRecorder(),
        )
    )
    assert result.metrics["withheld_actions"] == 1
    assert websocket.sent == []


def test_observation_seat_mismatch_withholds_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _possible = _request_fixture(seat=0)
    websocket = FakeWebSocket(
        [
            json.dumps({"type": "start_game", "id": 1}),
            request,
            json.dumps(
                {"type": "end_game", "scores": [25000] * 4}
            ),
            json.dumps(
                {"type": "validation_result", "passed": True}
            ),
        ]
    )
    monkeypatch.setattr(
        "websockets.asyncio.client.connect", FakeConnect(websocket)
    )
    result = asyncio.run(
        play_connection(
            url="wss://example.invalid/validate",
            mode="validate",
            token="secret",
            policy=FirstLegalPolicy(),
            recorder=EventRecorder(),
        )
    )
    assert result.metrics["withheld_actions"] == 1
    assert websocket.sent == []


def test_validation_result_and_server_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            json.dumps({"type": "start_game", "id": 0}),
            json.dumps(
                {"type": "end_game", "scores": [25000] * 4}
            ),
            json.dumps(
                {"type": "validation_result", "passed": True}
            ),
        ]
    )
    monkeypatch.setattr(
        "websockets.asyncio.client.connect", FakeConnect(websocket)
    )
    result = asyncio.run(
        play_connection(
            url="wss://example.invalid/validate",
            mode="validate",
            token="secret",
            policy=FirstLegalPolicy(),
            recorder=EventRecorder(),
        )
    )
    assert result.validation_passed is True

    denied = FakeWebSocket(
        [json.dumps({"error": "Token verification failed: Bot is inactive"})]
    )
    monkeypatch.setattr(
        "websockets.asyncio.client.connect", FakeConnect(denied)
    )
    with pytest.raises(AuthenticationError):
        asyncio.run(
            play_connection(
                url="wss://example.invalid/validate",
                mode="validate",
                token="secret",
                policy=FirstLegalPolicy(),
                recorder=EventRecorder(),
            )
        )


def test_ranked_retries_transient_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    waits: list[float] = []

    async def fake_play_connection(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary disconnect")
        return SessionResult(True, None, (25000,) * 4, {})

    async def fake_sleep(delay: float):
        waits.append(delay)

    monkeypatch.setattr(
        "riichi_lab_bot.client.play_connection", fake_play_connection
    )
    monkeypatch.setattr("riichi_lab_bot.client.asyncio.sleep", fake_sleep)
    results = asyncio.run(
        run_ranked(
            url="wss://example.invalid/ranked",
            token="secret",
            policy=FirstLegalPolicy(),
            recorder=EventRecorder(),
            games=1,
        )
    )
    assert len(results) == 1
    assert calls == 2
    assert waits == [5.0]
