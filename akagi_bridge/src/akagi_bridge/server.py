"""Akagi v3 云端推理 HTTP 服务。

实现 Akagi `bot.api` 需要的端点:

- ``POST /v3/react``  — 决策(Akagi 每次出牌/鸣牌/立直前调用)
- ``GET  /v3/models`` — 模型列表(Akagi 设置页的模型下拉)
- ``GET  /v3/key``    — 密钥状态(Akagi 设置页的"检查密钥")
- ``GET  /healthz``   — 无鉴权存活探针

约定与 Akagi 侧一致:任意非 2xx 响应都会让 Akagi 退回它内置的本地模型,
因此本服务对无法处理的请求一律**显式失败**,而不是猜一个动作。
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .engine import (
    MAX_EVENTS,
    SLOW_DECISION_MS,
    AkagiDecisionEngine,
    AkagiProtocolError,
)

LOGGER = logging.getLogger("akagi_bridge.server")


class ReactRequest(BaseModel):
    """Akagi `POST /v3/react` 的请求体。"""

    model: str | None = None
    player_id: int = Field(ge=0, le=3)
    events: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_EVENTS)


class CandidateModel(BaseModel):
    action: str
    prob: float


class ReactResponse(BaseModel):
    reaction: dict[str, Any] | None
    candidates: list[CandidateModel] = Field(default_factory=list)
    model: str | None = None


class ModelEntry(BaseModel):
    id: str
    game: str = "4p"
    desc: str = ""


class ModelsResponse(BaseModel):
    models: list[ModelEntry] = Field(default_factory=list)


class KeyStatusResponse(BaseModel):
    """自建服务的密钥状态:字段与 Akagi 的 `KeyStatus` 对齐。"""

    plan: str = "self-hosted"
    expires_at: str = ""
    usage_today: int = 0
    rpd: int = 0
    rpm: float = 0.0
    topk: int = 0
    reviews_today: int = 0
    reviews_per_day: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    queue_depth: int = 0
    workers_alive: bool = True


def create_app(
    engine: AkagiDecisionEngine,
    *,
    api_key: str | None = None,
    model_id: str = "zenith-v18",
    model_desc: str = "",
    recorder: Any | None = None,
) -> FastAPI:
    """构建 FastAPI 应用。`api_key=None` 表示不校验 Bearer 令牌。"""
    app = FastAPI(title="Zenith → Akagi bridge", version="0.1.0")
    # 单卡模型推理串行化:同一时刻只跑一次 forward,避免显存/线程争用。
    lock = threading.Lock()

    def _authorize(authorization: str | None) -> None:
        if api_key is None:
            return
        if authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid api key")

    def _emit(kind: str, **fields: Any) -> None:
        if recorder is not None:
            recorder.emit(kind, **fields)

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse()

    @app.get("/v3/key", response_model=KeyStatusResponse)
    def key_status(
        authorization: str | None = Header(default=None),
    ) -> KeyStatusResponse:
        _authorize(authorization)
        return KeyStatusResponse(topk=engine.topk)

    @app.get("/v3/models", response_model=ModelsResponse)
    def models(
        authorization: str | None = Header(default=None),
    ) -> ModelsResponse:
        _authorize(authorization)
        return ModelsResponse(
            models=[ModelEntry(id=model_id, game="4p", desc=model_desc)]
        )

    @app.post("/v3/react", response_model=ReactResponse)
    def react(
        payload: ReactRequest,
        authorization: str | None = Header(default=None),
    ) -> ReactResponse:
        _authorize(authorization)
        with lock:
            try:
                decision = engine.react(payload.events, payload.player_id)
            except AkagiProtocolError as exc:
                _emit(
                    "akagi_protocol_error",
                    player_id=payload.player_id,
                    events=len(payload.events),
                    error=str(exc),
                )
                # 422 ⇒ Akagi 立刻退回内置本地模型,不会让对局卡住。
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:  # noqa: BLE001 -- 任何内部错误都交给 Akagi 兜底
                LOGGER.exception("react failed")
                _emit(
                    "akagi_internal_error",
                    player_id=payload.player_id,
                    events=len(payload.events),
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise HTTPException(
                    status_code=500, detail=f"{type(exc).__name__}: {exc}"
                ) from exc
        if decision.elapsed_ms > SLOW_DECISION_MS:
            LOGGER.warning(
                "slow decision: %.1f ms (player_id=%d, events=%d)",
                decision.elapsed_ms,
                payload.player_id,
                len(payload.events),
            )
        _emit(
            "akagi_react",
            player_id=payload.player_id,
            events=len(payload.events),
            action_id=decision.action_id,
            reaction=decision.reaction,
            elapsed_ms=round(decision.elapsed_ms, 3),
        )
        return ReactResponse(
            reaction=decision.reaction,
            candidates=[
                CandidateModel(action=item.action, prob=item.prob)
                for item in decision.candidates
            ],
            model=payload.model or model_id,
        )

    return app


def serve(
    engine: AkagiDecisionEngine,
    *,
    host: str = "127.0.0.1",
    port: int = 8090,
    api_key: str | None = None,
    model_id: str = "zenith-v18",
    model_desc: str = "",
    recorder: Any | None = None,
) -> None:
    """启动 uvicorn 服务(阻塞直到收到退出信号)。"""
    import uvicorn

    app = create_app(
        engine,
        api_key=api_key,
        model_id=model_id,
        model_desc=model_desc,
        recorder=recorder,
    )
    LOGGER.info(
        "serving Akagi cloud inference on http://%s:%d (auth=%s)",
        host,
        port,
        "on" if api_key else "off",
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
