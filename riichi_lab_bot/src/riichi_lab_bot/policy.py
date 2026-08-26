"""V18 checkpoint 加载与确定性策略推理。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from riichi_ppo_v1.model.encoding_protocol import (
    QUERY_DEFENSE,
    QUERY_OFFENSE,
    QUERY_ROW_ACTION_ID,
    QUERY_ROW_ACTION_TYPE,
    QUERY_ROW_QUERY_TYPE,
)
from riichi_ppo_v1.model.semantic_validation import (
    assert_actor_input_semantics,
)

from .bridge import PreparedDecision
from .model import (
    NUM_ACTIONS,
    NUMERIC_WIDTH,
    QUERY_ROW_WIDTH,
    TOKEN_WIDTH,
    KyokuTransformerActorCritic,
    ModelConfig,
)
# V18 待迁移：bot 的 PreparedDecision 仍使用旧 history/snapshot 布局，本阶段不兼容。


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


def _checkpoint_format(payload: dict[str, Any]) -> str:
    ppo_format = payload.get("ppo_format_version")
    if ppo_format is not None:
        return f"ppo_v{int(ppo_format)}"
    if payload.get("sft_contract_version") is not None:
        return "sft_v18"
    return "v18_weights"


def _warmup_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """构造一条最小但合法的 V18 policy-only 前向样本。"""
    history_factors = np.zeros((1, 1, TOKEN_WIDTH), dtype=np.uint8)
    history_factors[0, 0, 0] = 1
    history_factors[0, 0, 1] = 1
    history_numeric = np.zeros((1, 1, NUMERIC_WIDTH), dtype=np.float32)
    history_lengths = np.asarray([1], dtype=np.int64)

    snapshot_factors = np.zeros(
        (1, SNAPSHOT_FIELD_COUNT, SNAPSHOT_FACTOR_WIDTH), dtype=np.uint8,
    )
    for index, field in enumerate(SNAPSHOT_FIELDS):
        snapshot_factors[0, index, 0] = field.field_id
        snapshot_factors[0, index, 1] = field.relative_seat
    snapshot_numeric = np.zeros(
        (1, SNAPSHOT_FIELD_COUNT, SNAPSHOT_NUMERIC_WIDTH), dtype=np.float32,
    )
    snapshot_lengths = np.asarray([SNAPSHOT_FIELD_COUNT], dtype=np.int64)

    query_rows = np.zeros((1, 2, QUERY_ROW_WIDTH), dtype=np.int32)
    query_rows[0, 0, QUERY_ROW_QUERY_TYPE] = QUERY_OFFENSE
    query_rows[0, 1, QUERY_ROW_QUERY_TYPE] = QUERY_DEFENSE
    query_rows[:, :, QUERY_ROW_ACTION_ID] = 0
    query_rows[:, :, QUERY_ROW_ACTION_TYPE] = 1
    query_action_ids = np.asarray([[0]], dtype=np.int32)
    query_pair_counts = np.asarray([1], dtype=np.int64)
    legal_mask = np.zeros((1, NUM_ACTIONS), dtype=np.bool_)
    legal_mask[0, 0] = True
    assert_actor_input_semantics(
        history_factors,
        history_numeric,
        history_lengths,
        snapshot_factors,
        snapshot_numeric,
        snapshot_lengths,
        query_rows,
        query_action_ids,
        query_pair_counts,
        legal_mask,
    )
    return (
        history_factors,
        history_numeric,
        history_lengths,
        snapshot_factors,
        snapshot_numeric,
        snapshot_lengths,
        query_rows,
        query_action_ids,
        query_pair_counts,
        legal_mask,
    )


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
        raw_config = payload.get("model_config")
        if not isinstance(raw_config, dict):
            raise ValueError("checkpoint is missing model_config")
        if raw_config.get("policy_head_type") != "isolated_action_query":
            raise RuntimeError(
                "V18 bot requires policy_head_type=isolated_action_query"
            )
        try:
            config = ModelConfig.from_mapping(dict(raw_config))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("checkpoint has an invalid model_config") from exc
        state = payload.get("model")
        if not isinstance(state, dict):
            raise RuntimeError("checkpoint is missing model weights")
        model = KyokuTransformerActorCritic(config)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "checkpoint tensor shapes do not match model_config"
            ) from exc
        model.to(self.device)
        model.eval()
        self.model = model
        self.config = config
        self.metadata: dict[str, Any] = {
            "checkpoint": str(self.checkpoint),
            "checkpoint_format": _checkpoint_format(payload),
            "sft_contract_version": payload.get("sft_contract_version"),
            "ppo_format_version": payload.get("ppo_format_version"),
            "token_schema_version": payload.get("token_schema_version"),
            "policy_head_type": raw_config["policy_head_type"],
            "model_config": dict(raw_config),
        }

    @property
    def dtype_name(self) -> str:
        return (
            "bf16"
            if self.autocast_dtype == torch.bfloat16
            else "fp32"
        )

    def warmup(self) -> float:
        inputs = tuple(self._tensor(value) for value in _warmup_inputs())
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype == torch.bfloat16,
        ):
            self.model(*inputs, policy_only=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - started) * 1000.0

    def infer(self, prepared: PreparedDecision) -> InferenceResult:
        assert_actor_input_semantics(
            prepared.history_factors[None],
            prepared.history_numeric[None],
            np.asarray([prepared.history_length], dtype=np.int64),
            prepared.snapshot_factors[None],
            prepared.snapshot_numeric[None],
            np.asarray([prepared.snapshot_length], dtype=np.int64),
            prepared.query_rows[None],
            prepared.query_action_ids[None],
            np.asarray([prepared.query_pair_count], dtype=np.int64),
            prepared.legal_mask[None],
            context_tokens=int(self.config.context_tokens),
        )
        inputs = (
            self._tensor(prepared.history_factors[None]),
            self._tensor(prepared.history_numeric[None]),
            self._tensor(np.asarray([prepared.history_length], dtype=np.int64)),
            self._tensor(prepared.snapshot_factors[None]),
            self._tensor(prepared.snapshot_numeric[None]),
            self._tensor(np.asarray([prepared.snapshot_length], dtype=np.int64)),
            self._tensor(prepared.query_rows[None]),
            self._tensor(prepared.query_action_ids[None]),
            self._tensor(np.asarray([prepared.query_pair_count], dtype=np.int64)),
            self._tensor(prepared.legal_mask[None]),
        )
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype == torch.bfloat16,
        ):
            output = self.model(*inputs, policy_only=True)
            chosen = output["policy_logits"].argmax(-1)
        action_id = int(chosen.item())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return InferenceResult(action_id, elapsed_ms)

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        if value.dtype == np.float32:
            return torch.as_tensor(
                value, device=self.device, dtype=torch.float32
            )
        if value.dtype == np.bool_:
            return torch.as_tensor(
                value, device=self.device, dtype=torch.bool
            )
        return torch.as_tensor(value, device=self.device, dtype=torch.long)
