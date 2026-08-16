"""确定性评测的版本无关策略边界(V16 与存量 v13 契约并存,按 checkpoint 分发)。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from ..model.bridge import BatchedStateBridge, Decision
from ..model import KyokuTransformerActorCritic, ModelConfig
from ..sft.checkpoint import load_v13_weights_only
from ..sft.contract import SFT_CONTRACT_VERSION, assert_runtime_contract


@dataclass(frozen=True)
class PreparedPolicyBatch:
    factors: np.ndarray
    numeric: np.ndarray
    lengths: np.ndarray
    legal: np.ndarray


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
    ) -> PreparedPolicyBatch: ...

    def masked_logits(self, batch: PreparedPolicyBatch) -> torch.Tensor: ...

    def metadata(self) -> dict[str, Any]: ...


class TorchPolicyAdapter:
    contract_id = ""
    requires_decision_analysis = True

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


class PPOPolicyAdapter(V13PolicyAdapter):
    """Load a fresh PPO v2 checkpoint for deterministic greedy evaluation.

    PPO checkpoints share the v13 token/feature contract and policy head, so
    the evaluation preparation path is identical to the SFT adapter; only the
    checkpoint payload layout (``model`` + ``model_config`` + PPO metadata)
    differs.
    """

    contract_id = "ppo_v2"

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, device: torch.device | str,
    ) -> "PPOPolicyAdapter":
        checkpoint = Path(path)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid PPO checkpoint: {checkpoint}")
        if int(payload.get("ppo_format_version", 0)) != 2:
            raise RuntimeError(
                f"unsupported PPO checkpoint format: {payload.get('ppo_format_version')!r}"
            )
        raw_config = payload.get("model_config")
        if not isinstance(raw_config, dict):
            raise RuntimeError("PPO checkpoint is missing model_config")
        if raw_config.get("policy_head_type") != "isolated_action_query":
            raise RuntimeError("PPO checkpoint must use isolated_action_query")
        try:
            config = ModelConfig(**dict(raw_config))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("PPO checkpoint has an invalid model_config") from exc
        state = payload.get("model")
        if not isinstance(state, dict):
            raise RuntimeError("PPO checkpoint is missing model weights")
        model = KyokuTransformerActorCritic(config)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError("PPO checkpoint tensor shapes do not match model_config") from exc
        model.to(torch.device(device))
        model.eval()
        return cls(model, torch.device(device), checkpoint)


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
            "model_config": vars(self.model.config),
        }


def load_policy_adapter(
    path: str | Path, *, device: torch.device | str,
) -> PolicyAdapter:
    """在 checkpoint 边界做一次分发,评测器保持版本无关。"""
    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid policy checkpoint: {checkpoint}")
    raw_config = payload.get("model_config")
    if isinstance(raw_config, dict) and raw_config.get("policy_head_type") == "symmetric_action_query":
        try:
            config = ModelConfig(**dict(raw_config))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("V16 checkpoint has an invalid model_config") from exc
        state = payload.get("model")
        if not isinstance(state, dict):
            raise RuntimeError("V16 checkpoint is missing model weights")
        model = KyokuTransformerActorCritic(config)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError("V16 checkpoint tensor shapes do not match model_config") from exc
        model.to(torch.device(device))
        model.eval()
        return V16PolicyAdapter(model, torch.device(device), checkpoint)
    contract = payload.get("sft_contract_version")
    if contract == SFT_CONTRACT_VERSION:
        return V13PolicyAdapter.from_checkpoint(checkpoint, device=device)
    if int(payload.get("ppo_format_version", 0)) == 2:
        return PPOPolicyAdapter.from_checkpoint(checkpoint, device=device)
    raise RuntimeError("checkpoint has no supported v13 evaluation contract")
