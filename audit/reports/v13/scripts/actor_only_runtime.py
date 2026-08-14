import json
from pathlib import Path

import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.sft.data import iter_split_samples
from riichi_ppo_v1.sft.train import _model_config, collate_samples, load_config

cfg = load_config(Path("riichi_ppo_v1/configs/sft.yaml"))
device = torch.device("cuda:0")
model = KyokuTransformerActorCritic(_model_config(cfg)).to(device)
frozen_roots = {"critic_embedding", "critic_backbone", "value_head", "value_query"}
for name, parameter in model.named_parameters():
    if name.split(".", 1)[0] in frozen_roots:
        parameter.requires_grad_(False)
optimized = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(optimized, lr=1.5e-4, fused=True)
stream = iter_split_samples(Path("datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16"), "train", seed=1, shuffle=False, include_critic=False)
rows = [next(stream), next(stream)]
batch = collate_samples(rows, device, include_critic=False)
calls = {"critic_embedding": 0, "critic_backbone": 0, "value_head": 0}
hooks = []
for name in calls:
    hooks.append(getattr(model, name).register_forward_hook(lambda _m, _i, _o, n=name: calls.__setitem__(n, calls[n] + 1)))
optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(batch["token_factors"], batch["token_numeric"], batch["legal_mask"], batch["token_lengths"], policy_only=True)
    loss = torch.nn.functional.cross_entropy(output["policy_logits"].float(), batch["actions"])
loss.backward()
grad_none = {name: p.grad is None for name, p in model.named_parameters() if name.split(".", 1)[0] in frozen_roots}
finite_legal = torch.isfinite(output["policy_logits"])[batch["legal_mask"]].all().item()
negative_inf_illegal = torch.isneginf(output["policy_logits"])[~batch["legal_mask"]].all().item()
optimizer.step()
report = {
    "device": str(device), "device_name": torch.cuda.get_device_name(device),
    "bf16_supported": torch.cuda.is_bf16_supported(), "logits_shape": list(output["policy_logits"].shape),
    "loss_finite": bool(torch.isfinite(loss)), "critic_forward_calls": calls,
    "all_frozen_grads_none": all(grad_none.values()), "frozen_grad_none": grad_none,
    "legal_logits_finite": bool(finite_legal), "illegal_logits_negative_inf": bool(negative_inf_illegal),
    "optimizer_fused": optimizer.defaults.get("fused"), "optimizer_state_entries": len(optimizer.state),
    "optimizer_parameter_tensors": sum(len(g["params"]) for g in optimizer.param_groups),
    "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
}
(Path(__file__).resolve().parents[1]/"actor_only_runtime.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
