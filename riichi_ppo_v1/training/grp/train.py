"""V16 GRP 离线训练入口:prefix→最终排名 CE、冻结与 σ_GRP 固化。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from ...model.grp import GRPModel, GRP_CATEGORIES, GRP_NUMERIC_FEATURES, GRP_UTILITY, expected_value
from .prepare import iter_grp_samples


DEFAULT_CONFIG = {
    "seed": 1,
    "device": "cuda",
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "shuffle_buffer_samples": 65536,
    "log_interval_steps": 100,
    "checkpoint_dir": "checkpoints/train_riichi_v16/grp",
}


def load_config(path: Path | None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if path is not None:
        with path.open(encoding="utf-8") as file:
            overlay = yaml.safe_load(file)
        if not isinstance(overlay, dict):
            raise ValueError("GRP config must be a mapping")
        config.update(overlay)
    return config


def collate(rows: list[tuple[np.ndarray, np.ndarray, int]], device: torch.device) -> dict:
    maximum = max(len(categorical) for categorical, _numeric, _rank in rows)
    categorical = torch.zeros((len(rows), maximum, len(GRP_CATEGORIES)), dtype=torch.long)
    numeric = torch.zeros((len(rows), maximum, GRP_NUMERIC_FEATURES))
    lengths = torch.empty(len(rows), dtype=torch.long)
    ranks = torch.empty(len(rows), dtype=torch.long)
    for index, (cat, num, rank) in enumerate(rows):
        length = len(cat)
        categorical[index, :length] = torch.from_numpy(cat)
        numeric[index, :length] = torch.from_numpy(num)
        lengths[index] = length
        ranks[index] = rank
    return {
        "categorical": categorical.to(device),
        "numeric": numeric.to(device),
        "lengths": lengths.to(device),
        "ranks": ranks.to(device),
    }


def prefix_loss(logits: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    """对每个 prefix 独立监督最终排名:L=(1/K)Σ CE(P(s_{0:k}), rank_final)。"""
    return F.cross_entropy(
        logits.transpose(1, 2),
        ranks[:, None].expand_as(logits[..., 0]),
    )


def evaluate(
    model: GRPModel, dataset: Path, split: str, device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total = correct = 0
    buffer: list[tuple[np.ndarray, np.ndarray, int]] = []
    with torch.no_grad():
        for row in iter_grp_samples(dataset, split):
            buffer.append(row)
            if len(buffer) < 64:
                continue
            batch = collate(buffer, device)
            logits = model(batch["categorical"], batch["numeric"], batch["lengths"])
            ends = batch["lengths"] - 1
            predicted = logits[torch.arange(logits.shape[0], device=device), ends].argmax(dim=1)
            ranks = batch["ranks"]
            correct += int((predicted == ranks).sum())
            total += len(buffer)
            buffer.clear()
        if buffer:
            batch = collate(buffer, device)
            logits = model(batch["categorical"], batch["numeric"], batch["lengths"])
            ends = batch["lengths"] - 1
            predicted = logits[torch.arange(logits.shape[0], device=device), ends].argmax(dim=1)
            correct += int((predicted == batch["ranks"]).sum())
            total += len(buffer)
    return correct / max(total, 1), float(total)


def _sigma_grp(
    model: GRPModel, dataset: Path, split: str, device: torch.device,
) -> float:
    """在验证数据上离线统计 GRP 小局 delta 的标准差(训练期只读)。"""
    deltas: list[float] = []
    utility = torch.tensor(GRP_UTILITY, device=device)
    model.eval()
    buffer: list[tuple[np.ndarray, np.ndarray, int]] = []

    def consume(rows: list[tuple[np.ndarray, np.ndarray, int]]) -> None:
        batch = collate(rows, device)
        logits = model(batch["categorical"], batch["numeric"], batch["lengths"])
        values = torch.softmax(logits, dim=-1) @ utility
        for row in range(values.shape[0]):
            length = int(batch["lengths"][row])
            if length > 1:
                deltas.extend((values[row, 1:length] - values[row, :length - 1]).cpu().tolist())

    with torch.no_grad():
        for row in iter_grp_samples(dataset, split):
            buffer.append(row)
            if len(buffer) < 64:
                continue
            consume(buffer)
            buffer.clear()
        if buffer:
            consume(buffer)
    return float(np.std(np.asarray(deltas, dtype=np.float32))) if deltas else 1.0


def train_grp(dataset: Path, config: dict) -> None:
    device = torch.device(str(config["device"]))
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    model = GRPModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    output = Path(str(config["checkpoint_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    step = 0
    best_accuracy = 0.0
    buffer: list[tuple[np.ndarray, np.ndarray, int]] = []
    rng = random.Random(int(config["seed"]))
    batch_size = max(1, int(config["batch_size"]))
    buffer_capacity = max(int(config["shuffle_buffer_samples"]), batch_size)

    def drain(*, flush: bool) -> None:
        nonlocal step
        rng.shuffle(buffer)
        usable = len(buffer) if flush else len(buffer) - len(buffer) % batch_size
        for start in range(0, usable, batch_size):
            _train_step(model, optimizer, buffer[start:start + batch_size], device, config)
            step += 1
            if step % int(config["log_interval_steps"]) == 0:
                print(f"grp step={step}", flush=True)
        del buffer[:usable]

    for epoch in range(int(config["epochs"])):
        for categorical, numeric, rank in iter_grp_samples(dataset, "train"):
            buffer.append((categorical, numeric, rank))
            if len(buffer) < buffer_capacity:
                continue
            drain(flush=False)
        drain(flush=False)
    drain(flush=True)
    accuracy, total = evaluate(model, dataset, "validation", device)
    print(json.dumps({"validation/rank_accuracy": accuracy, "validation/samples": total}), flush=True)
    if accuracy <= 0.25:
        raise RuntimeError(
            f"GRP validation accuracy {accuracy:.4f} is not better than uniform random 0.25"
        )
    model.freeze()
    sigma_grp = _sigma_grp(model, dataset, "validation", device)
    _write_normalization(dataset, sigma_grp)
    payload = {
        "model": model.state_dict(),
        "config": config,
        "model_config": {
            "categories": list(GRP_CATEGORIES),
            "numeric_features": GRP_NUMERIC_FEATURES,
        },
        "training_stage": "grp",
        "global_step": step,
        "sigma_grp": sigma_grp,
    }
    temporary = output / "best.pt.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, output / "best.pt")
    (output / "config_snapshot.json").write_text(
        json.dumps(payload["model_config"] | {"config": config}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"sigma_grp": sigma_grp, "checkpoint": str(output / "best.pt")}), flush=True)


def _train_step(
    model: GRPModel,
    optimizer: torch.optim.Optimizer,
    rows: list[tuple[np.ndarray, np.ndarray, int]],
    device: torch.device,
    config: dict,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = collate(rows, device)
    logits = model(batch["categorical"], batch["numeric"], batch["lengths"])
    loss = prefix_loss(logits, batch["ranks"])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
    optimizer.step()


def _write_normalization(dataset: Path, sigma_grp: float) -> None:
    path = dataset / "dataset.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["normalization"]["sigma_grp"] = sigma_grp
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/tenhou_grp_2024_2025_v16"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.epochs is not None:
        config["epochs"] = args.epochs
    train_grp(args.dataset, config)


if __name__ == "__main__":
    main()
