"""V18 模型参数与 state-key 的统一审计。"""

from __future__ import annotations

from typing import Any

from torch import nn


def parameter_report(model: nn.Module) -> dict[str, Any]:
    by_root: dict[str, int] = {}
    trainable = 0
    total = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        root = name.split(".", 1)[0]
        by_root[root] = by_root.get(root, 0) + count
        total += count
        if parameter.requires_grad:
            trainable += count
    keys = tuple(sorted(model.state_dict()))
    forbidden = tuple(
        key for key in keys
        if "q_scorer" in key or "candidate_q" in key or "dueling_q" in key
    )
    return {
        "total": total,
        "trainable": trainable,
        "by_root": dict(sorted(by_root.items())),
        "state_key_count": len(keys),
        "forbidden_q_keys": forbidden,
    }


def assert_v18_parameter_contract(model: nn.Module) -> dict[str, Any]:
    report = parameter_report(model)
    if not 4_900_000 <= report["total"] <= 5_100_000:
        raise RuntimeError(f"V18 parameter count is outside 4.9M–5.1M: {report['total']}")
    if report["forbidden_q_keys"]:
        raise RuntimeError("V18 state contains forbidden Q keys")
    return report
