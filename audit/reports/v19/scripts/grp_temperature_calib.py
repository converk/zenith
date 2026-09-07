"""GRP 推理温度标定:验证集折半,标定折拟合 T,独立测试折评估。

  用法: python grp_temperature_calib.py [checkpoint 路径, 默认 V18]
  输出各温度的 eu_mae 与最优 T;更换 grp_checkpoint 后须以新 checkpoint
  重新标定并同步 v19_ppo.yaml 的 grp_temperature。
"""
from pathlib import Path
import sys

import torch

from riichi_ppo_v1.model.grp import (
    GRPModel,
    expected_utility_from_logits,
    true_expected_utility,
)
from riichi_ppo_v1.training.grp.prepare import iter_grp_samples

DEFAULT_CHECKPOINT = "checkpoints/train_riichi_v18/grp/best.pt"
TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

checkpoint = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
mc = payload["model_config"]
model = GRPModel(
    input_size=mc["input_size"], hidden_size=mc["hidden"], num_layers=mc["layers"],
)
model.load_state_dict(payload["model"], strict=True)
model = model.to("cuda").eval()

calib: list[tuple[torch.Tensor, torch.Tensor]] = []  # 标定折(偶数位)
test: list[tuple[torch.Tensor, torch.Tensor]] = []   # 测试折(奇数位)
buffer: list = []

with torch.no_grad():
    for row in iter_grp_samples(Path("datasets/tenhou_grp_2024_2025_v18"), "validation"):
        buffer.append(row)
        if len(buffer) < 4096:
            continue
        maximum = max(len(f) for f, _ in buffer)
        features = torch.zeros((len(buffer), maximum, 21))
        lengths = torch.empty(len(buffer), dtype=torch.long)
        ranks = torch.empty((len(buffer), 4), dtype=torch.long)
        for i, (f, r) in enumerate(buffer):
            features[i, : len(f)] = torch.from_numpy(f)
            lengths[i] = len(f)
            ranks[i] = torch.from_numpy(r)
        logits = model(features.to("cuda"), lengths.to("cuda"))
        eu_true = true_expected_utility(ranks.to("cuda"))
        for i in range(len(buffer)):
            item = (logits[i].cpu(), eu_true[i].cpu())
            (calib if (len(calib) + len(test)) % 2 == 0 else test).append(item)
        buffer = []

def mae(items, temp: float) -> float:
    total = 0.0
    for logits1, eu_true1 in items:
        eu_pred = expected_utility_from_logits(
            logits1.to("cuda").unsqueeze(0) / temp, model.perms,
        )
        total += float((eu_pred[0] - eu_true1.to("cuda")).abs().sum())
    return total / (4 * len(items))

print(f"checkpoint: {checkpoint}")
print(f"标定折 {len(calib)} 样本 / 测试折 {len(test)} 样本")
print()
print("=== 标定折扫 T ===")
best_t, best_v = 1.0, float("inf")
for temp in TEMPERATURES:
    v = mae(calib, temp)
    print(f"  T={temp:>4}: eu_mae={v:.4f}")
    if v < best_v:
        best_t, best_v = temp, v
print(f"标定折最优 T={best_t}")
print()
print("=== 独立测试折最终评估 ===")
base = mae(test, 1.0)
calibrated = mae(test, best_t)
print(f"  T=1.0(无标定): eu_mae={base:.4f}")
print(f"  T={best_t}(标定)  : eu_mae={calibrated:.4f}({(calibrated - base) / base * 100:+.2f}%)")
