"""Checkpoint loading and deterministic policy inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .bridge import PreparedDecision
from .model import (
    KyokuTransformerActorCritic,
    ModelConfig,
    NUM_ACTIONS,
    NUMERIC_WIDTH,
    TOKEN_SCHEMA_VERSION,
    TOKEN_WIDTH,
)


def resolve_device(value: str) -> torch.device:
    normalized = value.lower()
    if normalized == "auto":
        normalized = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    normalized = value.lower()
    if normalized == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    if normalized == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requires a BF16-capable CUDA device")
        return torch.bfloat16
    if normalized == "fp32":
        return torch.float32
    raise ValueError("dtype must be auto, fp32, or bf16")


@dataclass(frozen=True)
class InferenceResult:
    action_id: int
    elapsed_ms: float


class PolicyEngine:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint}")
        self.device = resolve_device(device)
        self.autocast_dtype = resolve_dtype(dtype, self.device)
        payload = torch.load(
            self.checkpoint, map_location="cpu", weights_only=False
        )
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a dictionary")
        schema = payload.get("token_schema_version")
        if schema != TOKEN_SCHEMA_VERSION:
            raise ValueError(
                "incompatible token schema: "
                f"checkpoint={schema!r}, runtime={TOKEN_SCHEMA_VERSION}"
            )
        model_config = payload.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("checkpoint is missing model_config")
        state = payload.get("model")
        if not isinstance(state, dict):
            raise ValueError("checkpoint is missing model state")
        self.config = ModelConfig(**model_config)
        self.model = KyokuTransformerActorCritic(self.config)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @property
    def dtype_name(self) -> str:
        return (
            "bf16"
            if self.autocast_dtype == torch.bfloat16
            else "fp32"
        )

    def warmup(self) -> float:
        factors = torch.zeros(
            (1, 1, TOKEN_WIDTH), device=self.device, dtype=torch.long
        )
        numeric = torch.zeros(
            (1, 1, NUMERIC_WIDTH),
            device=self.device,
            dtype=torch.float32,
        )
        legal = torch.ones(
            (1, NUM_ACTIONS), device=self.device, dtype=torch.bool
        )
        lengths = torch.zeros(
            (1,), device=self.device, dtype=torch.long
        )
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype == torch.bfloat16,
        ):
            self.model.forward_policy(factors, numeric, legal, lengths)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - started) * 1000.0

    def infer(self, prepared: PreparedDecision) -> InferenceResult:
        factors = torch.as_tensor(
            prepared.token_factors[None],
            device=self.device,
            dtype=torch.long,
        )
        numeric = torch.as_tensor(
            prepared.token_numeric[None],
            device=self.device,
            dtype=torch.float32,
        )
        legal = torch.as_tensor(
            prepared.legal_mask[None],
            device=self.device,
            dtype=torch.bool,
        )
        lengths = torch.as_tensor(
            [prepared.token_length],
            device=self.device,
            dtype=torch.long,
        )
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype == torch.bfloat16,
        ):
            output = self.model.forward_policy(
                factors, numeric, legal, lengths
            )
            chosen = output["policy_logits"].argmax(-1)
        action_id = int(chosen.item())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return InferenceResult(action_id, elapsed_ms)

