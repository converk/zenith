"""确定性 1v3 评测的 V16/V17 策略边界。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from ..model import KyokuTransformerActorCritic, ModelConfig
from ..model.bridge import BatchedStateBridge, Decision


@dataclass(frozen=True)
class V16PreparedPolicyBatch:
    history_factors: np.ndarray
    history_numeric: np.ndarray
    history_lengths: np.ndarray
    snapshot_kinds: np.ndarray
    snapshot_cat: np.ndarray
    snapshot_num: np.ndarray
    snapshot_lengths: np.ndarray
    query_rows: np.ndarray
    query_action_ids: np.ndarray
    query_pair_counts: np.ndarray
    legal: np.ndarray


class PolicyAdapter(Protocol):
    contract_id: str
    model: torch.nn.Module
    requires_decision_analysis: bool

    def prepare(
        self, bridge: BatchedStateBridge, decisions: list[Decision], analysis: Any | None = None,
    ) -> V16PreparedPolicyBatch: ...

    def masked_logits(self, batch: V16PreparedPolicyBatch) -> torch.Tensor: ...

    def metadata(self) -> dict[str, Any]: ...


class V16PolicyAdapter:
    """V16 三段编码的策略边界:评测只做 policy-only 前向。"""

    contract_id = "riichi-runtime-v16"
    requires_decision_analysis = False

    def __init__(self, model: torch.nn.Module, device: torch.device, checkpoint: Path) -> None:
        self.model = model
        self.device = device
        self.checkpoint = checkpoint.resolve()

    def prepare(
        self, bridge: BatchedStateBridge, decisions: list[Decision], analysis: Any | None = None,
    ) -> V16PreparedPolicyBatch:
        del analysis
        batch = bridge.prepare_v16(decisions)
        return V16PreparedPolicyBatch(
            batch.history_factors,
            batch.history_numeric,
            batch.history_lengths,
            batch.snapshot_kinds,
            batch.snapshot_cat,
            batch.snapshot_num,
            batch.snapshot_lengths,
            batch.query_rows,
            batch.query_action_ids,
            batch.query_pair_counts,
            batch.legal_mask,
        )

    @torch.inference_mode()
    def masked_logits(self, batch: V16PreparedPolicyBatch) -> torch.Tensor:
        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(value).to(
                self.device, non_blocking=self.device.type == "cuda",
            )

        use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            output = self.model.forward_v16(
                tensor(batch.history_factors),
                tensor(batch.history_numeric),
                tensor(batch.history_lengths),
                tensor(batch.snapshot_kinds),
                tensor(batch.snapshot_cat),
                tensor(batch.snapshot_num),
                tensor(batch.snapshot_lengths),
                tensor(batch.query_rows),
                tensor(batch.query_action_ids),
                tensor(batch.query_pair_counts),
                tensor(batch.legal),
                policy_only=True,
            )
        return output["policy_logits"].float()

    def metadata(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "checkpoint": str(self.checkpoint),
            "model_config": asdict(self.model.config),
        }


def load_policy_adapter(
    path: str | Path, *, device: torch.device | str,
) -> PolicyAdapter:
    """加载 V16/V17 symmetric_action_query checkpoint。"""
    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid policy checkpoint: {checkpoint}")
    raw_config = payload.get("model_config")
    if not isinstance(raw_config, dict):
        raise RuntimeError("policy checkpoint is missing model_config")
    try:
        config = ModelConfig.from_mapping(dict(raw_config))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("policy checkpoint has an invalid model_config") from exc
    if config.policy_head_type != "symmetric_action_query":
        raise RuntimeError("only V16/V17 symmetric_action_query checkpoints are supported")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise RuntimeError("policy checkpoint is missing model weights")
    model = KyokuTransformerActorCritic(config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("policy checkpoint tensors do not match model_config") from exc
    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()
    return V16PolicyAdapter(model, device_obj, checkpoint)
