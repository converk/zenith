"""Checkpoint loading and deterministic policy inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from riichi_ppo_v1.model.semantic_validation import assert_actor_token_semantics
from riichi_ppo_v1.model.feature_schema import (
    DECISION_ANALYSIS_VERSION,
    RUST_ANALYSIS_VERSION,
    feature_schema_sha256,
)
from riichi_ppo_v1.sft.checkpoint import load_v13_weights_only
from riichi_ppo_v1.sft.contract import (
    SFT_CONTRACT_VERSION,
    assert_runtime_contract,
)
from riichi_ppo_v1.evaluation.policy_adapter import PPOPolicyAdapter

from .bridge import PreparedDecision
from .model import (
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


def _warmup_inputs() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Build a minimal valid isolated-action-query input.

    The V13 policy head needs one state row followed by one adjacent
    offense/defense pair for exactly one legal action.  The warmup never
    touches a real observation, so its values are only shape/route smoke
    inputs.
    """
    factors = np.zeros((1, 3, TOKEN_WIDTH), dtype=np.uint8)
    factors[0, 0, 0] = 6  # state summary segment
    factors[0, 1, 0] = 7  # action query segment
    factors[0, 1, 1] = 1  # none action kind
    factors[0, 1, 2] = 1  # action_id 0 + 1
    factors[0, 1, 9] = 1  # offense role
    factors[0, 2, 0] = 7
    factors[0, 2, 1] = 1
    factors[0, 2, 2] = 1
    factors[0, 2, 9] = 2  # defense role
    numeric = np.zeros((1, 3, NUMERIC_WIDTH), dtype=np.float32)
    legal = np.zeros((1, NUM_ACTIONS), dtype=np.bool_)
    legal[0, 0] = True
    lengths = np.asarray([3], dtype=np.int64)
    return factors, numeric, legal, lengths


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
        contract = payload.get("sft_contract_version")
        schema = payload.get("token_schema_version")
        ppo_format = int(payload.get("ppo_format_version", 0))
        is_sft = contract == SFT_CONTRACT_VERSION
        is_ppo = ppo_format == 2
        if not is_sft and not is_ppo:
            raise ValueError(
                "incompatible token schema: "
                f"checkpoint={schema!r}, runtime={TOKEN_SCHEMA_VERSION}"
            )
        if is_ppo:
            if int(schema or 0) != TOKEN_SCHEMA_VERSION:
                raise ValueError(
                    "incompatible token schema: "
                    f"checkpoint={schema!r}, runtime={TOKEN_SCHEMA_VERSION}"
                )
            if payload.get("feature_schema_sha256") != feature_schema_sha256():
                raise ValueError("incompatible feature schema hash")
            if int(payload.get("rust_analysis_version", -1)) != RUST_ANALYSIS_VERSION:
                raise ValueError("incompatible Rust analysis version")
            if (
                int(payload.get("decision_analysis_version", -1))
                != DECISION_ANALYSIS_VERSION
            ):
                raise ValueError("incompatible decision-analysis version")
        model_config = payload.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("checkpoint is missing model_config")
        if model_config.get("policy_head_type") != "isolated_action_query":
            raise RuntimeError(
                "v13 checkpoint must explicitly use isolated_action_query"
            )
        assert_runtime_contract()
        if is_ppo:
            self.model = PPOPolicyAdapter.from_checkpoint(
                self.checkpoint, device=self.device
            ).model
            checkpoint_format = "ppo_v2"
        else:
            self.model = load_v13_weights_only(
                self.checkpoint, device=self.device
            )
            checkpoint_format = "sft_v13"
        self.config = self.model.config
        self.metadata: dict[str, Any] = {
            "checkpoint": str(self.checkpoint),
            "checkpoint_format": checkpoint_format,
            "sft_contract_version": contract,
            "ppo_format_version": ppo_format if is_ppo else None,
            "token_schema_version": TOKEN_SCHEMA_VERSION,
            "policy_head_type": model_config["policy_head_type"],
            "model_config": dict(model_config),
        }

    @property
    def dtype_name(self) -> str:
        return (
            "bf16"
            if self.autocast_dtype == torch.bfloat16
            else "fp32"
        )

    def warmup(self) -> float:
        factors, numeric, legal, lengths = _warmup_inputs()
        factors_t = torch.as_tensor(
            factors, device=self.device, dtype=torch.long
        )
        numeric_t = torch.as_tensor(
            numeric, device=self.device, dtype=torch.float32
        )
        legal_t = torch.as_tensor(
            legal, device=self.device, dtype=torch.bool
        )
        lengths_t = torch.as_tensor(
            lengths, device=self.device, dtype=torch.long
        )
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype == torch.bfloat16,
        ):
            self.model.forward_policy(
                factors_t, numeric_t, legal_t, lengths_t
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - started) * 1000.0

    def infer(self, prepared: PreparedDecision) -> InferenceResult:
        assert_actor_token_semantics(
            prepared.token_factors[None],
            prepared.token_numeric[None],
            np.asarray([prepared.token_length], dtype=np.int64),
        )
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
