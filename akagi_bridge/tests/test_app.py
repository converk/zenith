"""Akagi 云端推理 HTTP 契约测试(不需要原生扩展/模型权重)。"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from akagi_bridge.engine import AkagiCandidate, AkagiDecision, AkagiProtocolError
from akagi_bridge.server import create_app


class StubEngine:
    """只实现 `AkagiDecisionEngine` 的对外契约,用于隔离 HTTP 层。"""

    def __init__(self) -> None:
        self.topk = 3
        self.calls: list[tuple[int, int]] = []
        self.error: Exception | None = None

    def react(self, events: list[Any], player_id: int) -> AkagiDecision:
        self.calls.append((player_id, len(events)))
        if self.error is not None:
            raise self.error
        return AkagiDecision(
            reaction={
                "type": "dahai",
                "actor": player_id,
                "pai": "1m",
                "tsumogiri": False,
            },
            candidates=(
                AkagiCandidate("dahai:1m", 0.8),
                AkagiCandidate("dahai:9p", 0.15),
            ),
            action_id=7,
            elapsed_ms=12.5,
        )


EVENTS = [
    {"type": "start_game", "names": ["a", "b", "c", "d"]},
    {"type": "tsumo", "actor": 0, "pai": "5m"},
]


def _client(engine: StubEngine, api_key: str | None = None) -> TestClient:
    return TestClient(create_app(engine, api_key=api_key))


def test_healthz_is_open() -> None:
    client = _client(StubEngine(), api_key="secret")
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_react_returns_reaction_and_candidates() -> None:
    engine = StubEngine()
    client = _client(engine)
    response = client.post("/v3/react", json={"player_id": 0, "events": EVENTS})
    assert response.status_code == 200
    body = response.json()
    assert body["reaction"] == {
        "type": "dahai",
        "actor": 0,
        "pai": "1m",
        "tsumogiri": False,
    }
    assert body["candidates"][0] == {"action": "dahai:1m", "prob": 0.8}
    assert body["model"] == "zenith-v18"
    assert engine.calls == [(0, len(EVENTS))]


def test_react_echoes_requested_model() -> None:
    client = _client(StubEngine())
    response = client.post(
        "/v3/react",
        json={"model": "custom", "player_id": 2, "events": EVENTS},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "custom"


def test_react_requires_bearer_token_when_configured() -> None:
    client = _client(StubEngine(), api_key="secret")
    payload = {"player_id": 0, "events": EVENTS}
    assert client.post("/v3/react", json=payload).status_code == 401
    assert (
        client.post(
            "/v3/react", json=payload, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/v3/react", json=payload, headers={"Authorization": "Bearer secret"}
        ).status_code
        == 200
    )


def test_protocol_error_maps_to_422() -> None:
    engine = StubEngine()
    engine.error = AkagiProtocolError("no pending decision")
    response = _client(engine).post(
        "/v3/react", json={"player_id": 1, "events": EVENTS}
    )
    # 422 让 Akagi 立刻退回内置本地模型,而不是卡住对局。
    assert response.status_code == 422
    assert "no pending decision" in response.json()["detail"]


def test_internal_error_maps_to_500() -> None:
    engine = StubEngine()
    engine.error = RuntimeError("boom")
    response = _client(engine).post(
        "/v3/react", json={"player_id": 1, "events": EVENTS}
    )
    assert response.status_code == 500


def test_request_validation_rejects_bad_seat_and_empty_events() -> None:
    client = _client(StubEngine())
    assert (
        client.post("/v3/react", json={"player_id": 4, "events": EVENTS}).status_code
        == 422
    )
    assert (
        client.post("/v3/react", json={"player_id": 0, "events": []}).status_code == 422
    )


def test_models_and_key_endpoints() -> None:
    client = _client(StubEngine())
    models = client.get("/v3/models").json()
    assert models["models"][0]["id"] == "zenith-v18"
    assert models["models"][0]["game"] == "4p"
    key = client.get("/v3/key").json()
    assert key["topk"] == 3
    assert key["plan"] == "self-hosted"


def test_recorder_receives_react_events() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        def emit(self, kind: str, **fields: Any) -> None:
            self.events.append((kind, fields))

    recorder = Recorder()
    client = TestClient(create_app(StubEngine(), recorder=recorder))
    client.post("/v3/react", json={"player_id": 3, "events": EVENTS})
    kinds = [kind for kind, _fields in recorder.events]
    assert kinds == ["akagi_react"]
    assert recorder.events[0][1]["player_id"] == 3
