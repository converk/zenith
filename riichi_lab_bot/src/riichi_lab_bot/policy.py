"""V19 checkpoint 加载与确定性策略推理。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from riichi_ppo_v1.model.architecture import (
    KyokuTransformerActorCritic,
    ModelConfig,
)
from riichi_ppo_v1.model.encoding_protocol import (
    CATEGORY_SCHEMAS,
    KIND_ACTION_DEFENSE_QUERY,
    KIND_ACTION_OFFENSE_QUERY,
    KIND_BOS,
    KIND_OPPONENT_ANALYSIS,
    KIND_PLAYER,
    KIND_RIICHI_CARD,
    KIND_SELF_HAND,
    KIND_SELF_STATE_ANALYSIS,
    KIND_SEP_ACTIONS,
    KIND_SEP_KAMICHA_RIVER,
    KIND_SEP_MELDS,
    KIND_SEP_OPPONENT_ANALYSIS,
    KIND_SEP_PLAYERS,
    KIND_SEP_RIVERS,
    KIND_SEP_SELF_HAND,
    KIND_SEP_SHIMOCHA_RIVER,
    KIND_SEP_TILE_STATE,
    KIND_SEP_TOIMEN_RIVER,
    KIND_TABLE,
    KIND_TILE_STATE,
    SEPARATOR_KINDS,
    SEPARATOR_SEGMENTS,
    TOKEN_NUMERIC_WIDTH,
    TOKEN_ROW_WIDTH,
)
from riichi_ppo_v1.model.schema import NUM_ACTIONS, TILE_KINDS
from riichi_ppo_v1.model.semantic_validation import (
    assert_actor_input_semantics,
)

from .bridge import PreparedDecision


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
        return f"sft_{str(payload['sft_contract_version'])}"
    schema_version = payload.get("token_schema_version")
    return f"v{schema_version}_weights" if schema_version is not None else "weights"


def _warmup_row(kind: int, fields: dict[int, int]) -> np.ndarray:
    """构造一行 warmup token:segment/kind 单源取自协议 schema,离散列按位覆盖。"""
    segment = (
        SEPARATOR_SEGMENTS[kind]
        if kind in SEPARATOR_KINDS.values()
        else CATEGORY_SCHEMAS[kind].segment
    )
    row = np.zeros(TOKEN_ROW_WIDTH, dtype=np.int32)
    row[0], row[1] = segment, kind
    for column, value in fields.items():
        row[column] = value
    return row


def _warmup_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """构造一条最小但语义合法的 V19 policy-only 前向样本。

    行序列严格遵循 ``assert_actor_input_semantics`` 的规范顺序
    (BOS→TABLE→手牌→玩家→牌河(每河末尾恒发 RIICHI_CARD)→副露→牌况→分析→
    Action Query)。未显式覆盖的离散字段取 0(在各类别基数域内);空牌河/无副露
    均为合法形态。
    """
    rows = [
        _warmup_row(KIND_BOS, {}),
        # decision_mode=0(自摸后决策)要求 drawn_is_current=1。
        _warmup_row(KIND_TABLE, {8: 0, 11: 1}),
        _warmup_row(KIND_SEP_SELF_HAND, {}),
        # 手牌至少一行非零牌种且升序;drawn_type=0 时 is_drawn 必须为 0。
        _warmup_row(KIND_SELF_HAND, {2: 1, 3: 1}),
        _warmup_row(KIND_SELF_HAND, {2: 2, 3: 1}),
        _warmup_row(KIND_SELF_STATE_ANALYSIS, {}),
        _warmup_row(KIND_SEP_PLAYERS, {}),
        *(_warmup_row(KIND_PLAYER, {}) for _ in range(4)),
        _warmup_row(KIND_SEP_RIVERS, {}),
        *(
            row
            for river_sep in (
                KIND_SEP_SHIMOCHA_RIVER,
                KIND_SEP_TOIMEN_RIVER,
                KIND_SEP_KAMICHA_RIVER,
            )
            for row in (
                _warmup_row(river_sep, {}),
                # V19:每家用一条恒发射 RIICHI_CARD 收尾(未立直时字段全零)。
                _warmup_row(KIND_RIICHI_CARD, {}),
            )
        ),
        _warmup_row(KIND_SEP_MELDS, {}),
        _warmup_row(KIND_SEP_TILE_STATE, {}),
        # 34 类牌况:tile_type 升序 1..34;unknown=4、all_seen=0 为合法未完全可见形态。
        *(
            _warmup_row(KIND_TILE_STATE, {2: tile_type, 6: 4})
            for tile_type in range(1, TILE_KINDS + 1)
        ),
        _warmup_row(KIND_SEP_OPPONENT_ANALYSIS, {}),
        *(_warmup_row(KIND_OPPONENT_ANALYSIS, {}) for _ in range(3)),
        _warmup_row(KIND_SEP_ACTIONS, {}),
        _warmup_row(KIND_ACTION_OFFENSE_QUERY, {}),
        _warmup_row(KIND_ACTION_DEFENSE_QUERY, {}),
    ]
    actor_factors = np.stack(rows).astype(np.int32, copy=False)[None]
    actor_numeric = np.zeros(
        (1, len(rows), TOKEN_NUMERIC_WIDTH), dtype=np.float32
    )
    actor_lengths = np.asarray([len(rows)], dtype=np.int64)
    query_action_ids = np.asarray([[0]], dtype=np.int32)
    query_pair_counts = np.asarray([1], dtype=np.int64)
    legal_mask = np.zeros((1, NUM_ACTIONS), dtype=np.bool_)
    legal_mask[0, 0] = True
    assert_actor_input_semantics(
        actor_factors,
        actor_numeric,
        actor_lengths,
        None,
        query_action_ids,
        query_pair_counts,
        legal_mask,
    )
    return (
        actor_factors,
        actor_numeric,
        actor_lengths,
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
            raise ValueError("checkpoint payload must be a dictionary")  # noqa: TRY004
        raw_config = payload.get("model_config")
        if not isinstance(raw_config, dict):
            raise ValueError("checkpoint is missing model_config")  # noqa: TRY004
        if raw_config.get("policy_head_type") != "current_state_snapshot":
            raise RuntimeError(
                "V19 bot requires policy_head_type=current_state_snapshot"
            )
        try:
            config = ModelConfig.from_mapping(dict(raw_config))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("checkpoint has an invalid model_config") from exc
        state = payload.get("model")
        if not isinstance(state, dict):
            raise ValueError("checkpoint is missing model weights")  # noqa: TRY004
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
        # 每次决策的二次语义校验是安全契约:桥接装配结果在此按 checkpoint
        # 的 context_tokens 复核,任一侧漂移都 fail closed。
        assert_actor_input_semantics(
            prepared.actor_factors[None],
            prepared.actor_numeric[None],
            np.asarray([prepared.actor_length], dtype=np.int64),
            None,
            prepared.query_action_ids[None],
            np.asarray([prepared.query_pair_count], dtype=np.int64),
            prepared.legal_mask[None],
            context_tokens=int(self.config.context_tokens),
        )
        inputs = (
            self._tensor(prepared.actor_factors[None]),
            self._tensor(prepared.actor_numeric[None]),
            self._tensor(np.asarray([prepared.actor_length], dtype=np.int64)),
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
