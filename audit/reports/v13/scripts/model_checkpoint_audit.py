import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.sft.train import _model_config, load_config

cfg = load_config(Path("riichi_ppo_v1/configs/sft.yaml"))
model = KyokuTransformerActorCritic(_model_config(cfg))
frozen_roots = {"critic_embedding", "critic_backbone", "value_head", "value_query"}
for name, parameter in model.named_parameters():
    if name.split(".", 1)[0] in frozen_roots:
        parameter.requires_grad_(False)
params = list(model.named_parameters())
optimizer = torch.optim.AdamW([p for _, p in params if p.requires_grad])
by_root = defaultdict(lambda: {"total": 0, "trainable": 0, "tensors": 0})
for name, p in params:
    root = name.split(".", 1)[0]
    by_root[root]["total"] += p.numel()
    by_root[root]["trainable"] += p.numel() if p.requires_grad else 0
    by_root[root]["tensors"] += 1
report = {
    "model_config": asdict(model.config),
    "total_parameters": sum(p.numel() for _, p in params),
    "trainable_parameters": sum(p.numel() for _, p in params if p.requires_grad),
    "frozen_parameters": sum(p.numel() for _, p in params if not p.requires_grad),
    "optimizer_parameters": sum(p.numel() for g in optimizer.param_groups for p in g["params"]),
    "parameter_tensors": len(params), "state_dict_tensors": len(model.state_dict()),
    "modules": dict(sorted(by_root.items())),
    "checkpoints": {},
}
for filename in ("latest.pt", "best.pt"):
    payload = torch.load(Path("checkpoints/train_riichi_v13/sft") / filename, map_location="cpu", weights_only=False)
    shape_mismatch = {}
    current = model.state_dict()
    for name, value in payload["model"].items():
        if name not in current or tuple(value.shape) != tuple(current[name].shape):
            shape_mismatch[name] = {"checkpoint": list(value.shape), "current": list(current[name].shape) if name in current else None}
    optimizer_ids = {int(i) for g in payload["optimizer"]["param_groups"] for i in g["params"]}
    report["checkpoints"][filename] = {
        "global_step": payload["global_step"], "epoch": payload["epoch"],
        "state_dict_tensors": len(payload["model"]), "shape_mismatches": shape_mismatch,
        "missing_current_tensors": sorted(set(current) - set(payload["model"])),
        "optimizer_parameter_ids": len(optimizer_ids), "optimizer_state_entries": len(payload["optimizer"]["state"]),
        "data_cursor": payload.get("data_cursor"), "rank_steps": payload.get("rank_steps"),
    }
(Path(__file__).resolve().parents[1]/"model_checkpoint.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
