"""V18 GRP checkpoint 在验证集上的基线指标(CE + eu_mae),供 V19 重训对照。"""
from pathlib import Path

import torch

from riichi_ppo_v1.model.grp import GRPModel
from riichi_ppo_v1.training.grp.train import evaluate_validation_loss

payload = torch.load(
    "checkpoints/train_riichi_v18/grp/best.pt",
    map_location="cpu", weights_only=False,
)
mc = payload["model_config"]
model = GRPModel(input_size=mc["input_size"], hidden_size=mc["hidden"], num_layers=mc["layers"])
model.load_state_dict(payload["model"], strict=True)
model = model.to("cuda")
ce, eu_mae, total = evaluate_validation_loss(
    model,
    Path("datasets/tenhou_grp_2024_2025_v18"),
    "validation",
    torch.device("cuda"),
    2048,
)
print(f"V18 baseline: validation/loss={ce:.4f} validation/eu_mae={eu_mae:.4f} samples={int(total)}")
