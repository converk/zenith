"""V19 模型参数与 state-key 的统一审计。"""

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
        or "snapshot_embeddings" in key or "query_embedding" in key or "history" in key
    )
    # 分项：嵌入 / shared / actor / critic / head。
    # 分组与 sft/actor_bc.py 的 _CRITIC_ROOTS 保持一致：value_head 属于 critic 冻结组。
    grouped: dict[str, int] = {"embedding": 0, "shared": 0, "actor": 0, "critic": 0, "head": 0}
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        if name.startswith("token_embedding") or ".embedding" in name or "Embedding" in type(parameter).__name__:
            grouped["embedding"] += count
        elif name.startswith("public_backbone"):
            grouped["shared"] += count
        elif name.startswith("actor_backbone"):
            grouped["actor"] += count
        elif name.startswith("critic_backbone") or name.startswith("value_query") or name.startswith("value_head"):
            grouped["critic"] += count
        else:
            grouped["head"] += count
    return {
        "total": total,
        "trainable": trainable,
        "by_root": dict(sorted(by_root.items())),
        "by_group": dict(sorted(grouped.items())),
        "state_key_count": len(keys),
        "forbidden_q_keys": forbidden,
    }


def assert_v19_parameter_contract(model: nn.Module) -> dict[str, Any]:
    report = parameter_report(model)
    if report["total"] > 7_200_000:
        raise RuntimeError(f"V19 parameter count exceeds 7.2M: {report['total']}")
    if report["forbidden_q_keys"]:
        raise RuntimeError("V19 state contains forbidden Q/legacy keys")
    return report
