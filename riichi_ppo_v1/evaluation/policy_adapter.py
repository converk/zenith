"""确定性 1v3 评测的 V18 策略边界。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from ..model import KyokuTransformerActorCritic, ModelConfig
from ..model.bridge import BatchedStateBridge, Decision


@dataclass(frozen=True)
class PreparedPolicyBatch:
    actor_factors: np.ndarray
    actor_numeric: np.ndarray
    actor_lengths: np.ndarray
    query_rows: np.ndarray
    query_action_ids: np.ndarray
    query_pair_counts: np.ndarray
    legal: np.ndarray
    critic_factors: np.ndarray
    critic_lengths: np.ndarray


class PolicyAdapter(Protocol):
    contract_id: str
    model: torch.nn.Module
    requires_decision_analysis: bool

    def prepare(
        self, bridge: BatchedStateBridge, decisions: list[Decision], analysis: Any | None = None,
    ) -> PreparedPolicyBatch: ...

    def masked_logits(self, batch: PreparedPolicyBatch) -> torch.Tensor: ...

    def metadata(self) -> dict[str, Any]: ...


class V18PolicyAdapter:
    """V18 三段编码的策略边界:评测只做 policy-only 前向。"""

    contract_id = "riichi-runtime-v18"
    requires_decision_analysis = False

    def __init__(self, model: torch.nn.Module, device: torch.device, checkpoint: Path) -> None:
        self.model = model
        self.device = device
        self.checkpoint = checkpoint.resolve()

    def prepare(
        self, bridge: BatchedStateBridge, decisions: list[Decision], analysis: Any | None = None,
    ) -> PreparedPolicyBatch:
        del analysis
        batch = bridge.prepare(decisions)
        return PreparedPolicyBatch(
            batch.actor_factors,
            batch.actor_numeric,
            batch.actor_lengths,
            batch.query_rows,
            batch.query_action_ids,
            batch.query_pair_counts,
            batch.legal_mask,
            batch.critic_factors,
            batch.critic_lengths,
        )

    @torch.inference_mode()
    def masked_logits(self, batch: PreparedPolicyBatch) -> torch.Tensor:
        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(value).to(
                self.device, non_blocking=self.device.type == "cuda",
            )

        use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            output = self.model(
                tensor(batch.actor_factors).long(),
                tensor(batch.actor_numeric),
                tensor(batch.actor_lengths).long(),
                tensor(batch.query_action_ids),
                tensor(batch.query_pair_counts).long(),
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
    """加载严格 V18 current_state_snapshot checkpoint。"""
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
    if config.policy_head_type != "current_state_snapshot":
        raise RuntimeError("only V18 current_state_snapshot checkpoints are supported")
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
    return V18PolicyAdapter(model, device_obj, checkpoint)
