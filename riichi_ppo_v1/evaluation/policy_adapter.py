"""确定性 1v3 评测的 V19 策略边界(含信念指标面)。"""

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
    """评测 policy-only 前向所需的张量;critic 字段不进入评测路径。"""

    actor_factors: np.ndarray
    actor_numeric: np.ndarray
    actor_lengths: np.ndarray
    query_action_ids: np.ndarray
    query_pair_counts: np.ndarray
    legal: np.ndarray


class PolicyAdapter(Protocol):
    contract_id: str
    model: torch.nn.Module
    requires_decision_analysis: bool

    def prepare(
        self, bridge: BatchedStateBridge, decisions: list[Decision], analysis: Any | None = None,
    ) -> PreparedPolicyBatch: ...

    def outputs(self, batch: PreparedPolicyBatch) -> dict[str, torch.Tensor]: ...

    def masked_logits(self, batch: PreparedPolicyBatch) -> torch.Tensor: ...

    def metadata(self) -> dict[str, Any]: ...


class V19PolicyAdapter:
    """V19 三段编码 + 信念网络的策略边界:评测做 policy-only 前向。

    V19 模型在 ``policy_only`` 前向下也返回信念五头输出(信念是模型内部
    产物,与策略共用同一条前向路径),评测因此能对学习模型与 SFT 对手同时
    度量「对 SFT 对手的信念校准」(设计训练分册 §5.3)。
    """

    contract_id = "riichi-runtime-v19"
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
            batch.query_action_ids,
            batch.query_pair_counts,
            batch.legal_mask,
        )

    @torch.inference_mode()
    def outputs(self, batch: PreparedPolicyBatch) -> dict[str, torch.Tensor]:
        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(value).to(
                self.device, non_blocking=self.device.type == "cuda",
            )

        use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            return self.model(
                tensor(batch.actor_factors).long(),
                tensor(batch.actor_numeric),
                tensor(batch.actor_lengths).long(),
                tensor(batch.query_action_ids),
                tensor(batch.query_pair_counts).long(),
                tensor(batch.legal),
                policy_only=True,
                # 评测路径信念是模型自身输出,不做公共层梯度缩放(1.0)。
                belief_public_grad_scale=1.0,
                # 1v3 与训练/rollout 一致：逐动作读出开启（SFT/PPO 产品路径）。
                belief_readout_enabled=True,
                belief_readout_detach=True,
            )

    @torch.inference_mode()
    def masked_logits(self, batch: PreparedPolicyBatch) -> torch.Tensor:
        return self.outputs(batch)["policy_logits"].float()

    def metadata(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "checkpoint": str(self.checkpoint),
            "model_config": asdict(self.model.config),
        }


def load_policy_adapter(
    path: str | Path, *, device: torch.device | str,
) -> PolicyAdapter:
    """加载严格 V19 current_state_snapshot checkpoint。"""
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
        raise RuntimeError("only V19 current_state_snapshot checkpoints are supported")
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
    return V19PolicyAdapter(model, device_obj, checkpoint)
