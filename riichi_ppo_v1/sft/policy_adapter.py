"""Version-independent policy boundary for deterministic evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from ..model.bridge import BatchedStateBridge, Decision
from .checkpoint import load_v13_weights_only
from .contract import SFT_CONTRACT_VERSION, assert_runtime_contract


@dataclass(frozen=True)
class PreparedPolicyBatch:
    factors: np.ndarray
    numeric: np.ndarray
    lengths: np.ndarray
    legal: np.ndarray


class PolicyAdapter(Protocol):
    contract_id: str
    model: torch.nn.Module

    def prepare(
        self, bridge: BatchedStateBridge, decisions: list[Decision], analysis: Any | None = None,
    ) -> PreparedPolicyBatch: ...

    def masked_logits(self, batch: PreparedPolicyBatch) -> torch.Tensor: ...

    def metadata(self) -> dict[str, Any]: ...


class TorchPolicyAdapter:
    contract_id = ""

    def __init__(self, model: torch.nn.Module, device: torch.device, checkpoint: Path) -> None:
        self.model = model
        self.device = device
        self.checkpoint = checkpoint.resolve()

    @torch.inference_mode()
    def masked_logits(self, batch: PreparedPolicyBatch) -> torch.Tensor:
        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(value).to(self.device, non_blocking=self.device.type == "cuda")

        use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            output = self.model.forward_policy(
                tensor(batch.factors), tensor(batch.numeric),
                tensor(batch.legal), tensor(batch.lengths),
            )
        return output["policy_logits"].float()

    def metadata(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "checkpoint": str(self.checkpoint),
            "model_config": vars(self.model.config),
        }


class V13PolicyAdapter(TorchPolicyAdapter):
    contract_id = SFT_CONTRACT_VERSION

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, device: torch.device | str,
    ) -> "V13PolicyAdapter":
        return cls(load_v13_weights_only(path, device=device), torch.device(device), Path(path))

    def prepare(
        self, bridge: BatchedStateBridge, decisions: list[Decision], analysis: Any | None = None,
    ) -> PreparedPolicyBatch:
        assert_runtime_contract()
        factors, numeric, lengths, legal, _generations, _critic, _critic_lengths = (
            bridge.prepare(decisions, analysis)
        )
        return PreparedPolicyBatch(factors, numeric, lengths, legal)


def load_policy_adapter(
    path: str | Path, *, device: torch.device | str,
) -> PolicyAdapter:
    """Dispatch once at the checkpoint boundary; evaluators stay version-free."""
    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid policy checkpoint: {checkpoint}")
    contract = payload.get("sft_contract_version")
    schema = payload.get("token_schema_version")
    if contract == SFT_CONTRACT_VERSION or (contract is None and schema == 13):
        return V13PolicyAdapter.from_checkpoint(checkpoint, device=device)
    if contract is None and schema == 11:
        from ..legacy.v11 import V11PolicyAdapter

        return V11PolicyAdapter.from_checkpoint(checkpoint, device=device)
    raise RuntimeError("checkpoint has no supported v11/v13 evaluation contract")
