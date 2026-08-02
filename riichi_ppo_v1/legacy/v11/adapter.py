"""V11 implementation of the shared evaluation policy interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...model.bridge import Decision
from ...sft.policy_adapter import PreparedPolicyBatch, TorchPolicyAdapter
from ...training.rewards import DecisionAnalysisBatch, EfficiencyAnalyzer
from .contract import V11_CONTRACT_ID
from .encoder import prepare_v11
from .model import load_v11_weights_only


class V11PolicyAdapter(TorchPolicyAdapter):
    contract_id = V11_CONTRACT_ID

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.analyzer = EfficiencyAnalyzer(131_072)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, device: torch.device | str,
    ) -> "V11PolicyAdapter":
        return cls(load_v11_weights_only(path, device=device), torch.device(device), Path(path))

    def prepare(
        self, bridge: Any, decisions: list[Decision], analysis: Any | None = None,
    ) -> PreparedPolicyBatch:
        legacy_analysis = DecisionAnalysisBatch.build_legacy_v11(
            decisions,
            analyzer=self.analyzer,
            public=getattr(analysis, "_public", None),
        )
        factors, numeric, lengths, legal, _generations = prepare_v11(
            bridge, decisions, legacy_analysis,
        )
        return PreparedPolicyBatch(factors, numeric, lengths, legal)
