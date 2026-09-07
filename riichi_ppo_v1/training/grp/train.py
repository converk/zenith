"""GRP 离线训练入口(Mortal 方案):24 类排列 CE + E[U] 辅助回归、validation-loss best。

输入为 21 维边界状态(见 ``model.grp.GRP_INPUT_LAYOUT``);每半庄的全部 prefix
独立监督最终排列标签;batch、AdamW 与 lr 等由配置提供(DEFAULT_CONFIG 默认
batch 512、lr=1e-5);validation loss 最低的 checkpoint 才落盘(best.pt)。

损失 = 排列 CE + ``eu_loss_coef`` × MSE(E[U](logits), 真实排名 utility):
PPO 奖励直接消费期望 utility E[U],而排列 CE 只监督分类,辅助回归把训练
目标与奖励用途对齐;``eu_loss_coef=0`` 退回纯 CE。验证输出 CE loss 与
E[U] 平均绝对误差(eu_mae,奖励质量的直接度量);checkpoint 选择仍以
CE 为准。训练结束后重载 CE 最优权重再冻结重写 best.pt(保证落盘权重与
best_loss.json 的 validation_loss 标签一致,不无条件用最终权重覆盖中间
最优)。模型结构超参只取自 ``model.grp`` 常量,配置快照随 checkpoint 保存。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from ...model.grp import (
    GRP_HIDDEN,
    GRP_INPUT_LAYOUT,
    GRP_INPUT_SIZE,
    GRP_LAYERS,
    GRP_NUM_CLASSES,
    GRP_UTILITY,
    GRPModel,
    expected_utility_from_logits,
    true_expected_utility,
)
from .prepare import iter_grp_samples

DEFAULT_CONFIG = {
    "seed": 1,
    "device": "cuda",
    "epochs": 30,
    "batch_size": 512,
    "learning_rate": 1e-5,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    # E[U] 辅助回归系数:0 = 纯排列 CE(旧行为);>0 = CE + coef × MSE(E[U])。
    "eu_loss_coef": 1.0,
    "shuffle_buffer_samples": 65536,
    "log_interval_steps": 100,
    "val_interval_steps": 500,
    "checkpoint_dir": "checkpoints/train_riichi_current/grp",
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


def collate(rows: list[tuple[np.ndarray, np.ndarray]], device: torch.device) -> dict:
    maximum = max(len(features) for features, _ranks in rows)
    features = torch.zeros((len(rows), maximum, GRP_INPUT_SIZE), dtype=torch.float32)
    lengths = torch.empty(len(rows), dtype=torch.long)
    rank_by_player = torch.empty((len(rows), 4), dtype=torch.long)
    for index, (features_seq, ranks) in enumerate(rows):
        length = len(features_seq)
        features[index, :length] = torch.from_numpy(features_seq)
        lengths[index] = length
        rank_by_player[index] = torch.from_numpy(ranks)
    return {
        "features": features.to(device),
        "lengths": lengths.to(device),
        "rank_by_player": rank_by_player.to(device),
    }


def prefix_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """24 类排列 CE:每个训练样本(单个 prefix)独立监督最终排列。"""
    return F.cross_entropy(logits, labels)


def evaluate_validation_loss(
    model: GRPModel, dataset: Path, split: str, device: torch.device,
    batch_size: int,
) -> tuple[float, float, float]:
    """验证集平均 CE loss、每玩家 E[U] 平均绝对误差与样本数。

    eu_mae 是奖励质量的直接度量(奖励 = ΔE[U]),与 checkpoint 选择的
    CE loss 相互独立、恒同步输出。
    """
    model.eval()
    total_loss = 0.0
    total_eu_mae = 0.0
    total = 0
    buffer: list[tuple[np.ndarray, np.ndarray]] = []

    def consume(rows: list[tuple[np.ndarray, np.ndarray]]) -> None:
        nonlocal total_loss, total_eu_mae, total
        if not rows:
            return
        batch = collate(rows, device)
        labels = model.get_label(batch["rank_by_player"])
        logits = model(batch["features"], batch["lengths"])
        loss = F.cross_entropy(logits, labels, reduction="sum")
        eu_pred = expected_utility_from_logits(logits, model.perms)
        eu_true = true_expected_utility(batch["rank_by_player"])
        eu_mae_sum = (eu_pred - eu_true).abs().sum()
        total_loss += float(loss.detach().item())
        total_eu_mae += float(eu_mae_sum.detach().item())
        total += len(rows)

    with torch.no_grad():
        for row in iter_grp_samples(dataset, split):
            buffer.append(row)
            if len(buffer) < batch_size:
                continue
            consume(buffer)
            buffer.clear()
        consume(buffer)
    model.train()
    return (
        total_loss / max(total, 1),
        total_eu_mae / max(4 * total, 1),
        float(total),
    )


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
    best_validation_loss = float("inf")
    buffer: list[tuple[np.ndarray, np.ndarray]] = []
    rng = random.Random(int(config["seed"]))
    batch_size = max(1, int(config["batch_size"]))
    buffer_capacity = max(int(config["shuffle_buffer_samples"]), batch_size)
    val_interval = max(1, int(config["val_interval_steps"]))

    def drain(*, flush: bool) -> None:
        nonlocal step, best_validation_loss
        rng.shuffle(buffer)
        usable = len(buffer) if flush else len(buffer) - len(buffer) % batch_size
        for start in range(0, usable, batch_size):
            _train_step(model, optimizer, buffer[start:start + batch_size], device, config)
            step += 1
            if step % int(config["log_interval_steps"]) == 0:
                print(f"grp step={step}", flush=True)
            if step % val_interval == 0:
                val_loss, val_eu_mae, val_samples = evaluate_validation_loss(
                    model, dataset, "validation", device, batch_size,
                )
                print(
                    json.dumps({
                        "step": step,
                        "validation/loss": float(val_loss),
                        "validation/eu_mae": float(val_eu_mae),
                        "validation/samples": float(val_samples),
                    }),
                    flush=True,
                )
                if _maybe_save_best(
                    model, config, output,
                    step, val_loss, best_validation_loss,
                ):
                    best_validation_loss = float(val_loss)
        del buffer[:usable]

    for _epoch in range(int(config["epochs"])):
        for features_seq, ranks in iter_grp_samples(dataset, "train"):
            buffer.append((features_seq, ranks))
            if len(buffer) < buffer_capacity:
                continue
            drain(flush=False)
        drain(flush=False)
    drain(flush=True)

    val_loss, val_eu_mae, val_samples = evaluate_validation_loss(
        model, dataset, "validation", device, batch_size,
    )
    print(json.dumps({
        "validation/loss": float(val_loss),
        "validation/eu_mae": float(val_eu_mae),
        "validation/samples": float(val_samples),
    }), flush=True)
    saved = _maybe_save_best(
        model, config, output, step, val_loss, best_validation_loss,
    )
    if not saved and _latest_best_validation_loss(output) is None:
        # 极小数据集无中间验证时,至少保证有一个产物。
        _maybe_save_best(model, config, output, step, val_loss, float("inf"))
    # 训练全部结束:重载 CE 最优权重后再冻结重写 best.pt,保证落盘权重与
    # best_loss.json 的 validation_loss 标签一致;修复旧实现无条件用最终权重
    # 覆盖中间最优 checkpoint 的问题。best.pt 不存在时才回退当前权重。
    model.freeze()
    best_payload = output / "best.pt"
    best_recorded = _latest_best_validation_loss(output)
    if (
        best_payload.is_file()
        and best_recorded is not None
        and best_recorded < float(val_loss)
    ):
        best_state = torch.load(
            best_payload, map_location=device, weights_only=False,
        )
        model.load_state_dict(best_state["model"])
        print(json.dumps({
            "reload_best_step": int(best_state.get("global_step", -1)),
            "reload_best_validation_loss": float(best_recorded),
        }), flush=True)
    frozen_loss = float(_latest_best_validation_loss(output) or val_loss)
    _write_best_checkpoint(model, config, output, step, frozen_loss)
    (output / "config_snapshot.json").write_text(
        json.dumps({
            "model_config": {
                "input_size": GRP_INPUT_SIZE,
                "hidden": GRP_HIDDEN,
                "layers": GRP_LAYERS,
                "feature_layout": list(GRP_INPUT_LAYOUT),
                "num_classes": GRP_NUM_CLASSES,
                "utility": list(GRP_UTILITY),
            },
            "config": config,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "checkpoint": str(output / "best.pt"),
        "best_validation_loss": float(frozen_loss),
    }), flush=True)


def _write_best_checkpoint(
    model: GRPModel,
    config: dict,
    output: Path,
    step: int,
    validation_loss: float,
) -> None:
    """写 best.pt(不修改 best_loss.json);用于训练结束冻结后的重写。"""
    payload = {
        "model": model.state_dict(),
        "config": config,
        "model_config": {
            "input_size": GRP_INPUT_SIZE,
            "hidden": GRP_HIDDEN,
            "layers": GRP_LAYERS,
            "feature_layout": list(GRP_INPUT_LAYOUT),
            "num_classes": GRP_NUM_CLASSES,
            "utility": list(GRP_UTILITY),
        },
        "training_stage": "grp",
        "global_step": int(step),
        "validation_loss": float(validation_loss),
    }
    temporary = output / "best.pt.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, output / "best.pt")


def _latest_best_validation_loss(output: Path) -> float | None:
    """从 ``best_loss.json`` 读取已保存最低 val loss(无则 None)。"""
    path = output / "best_loss.json"
    if not path.is_file():
        return None
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))["validation_loss"])
    except (ValueError, KeyError, TypeError):
        return None


def _maybe_save_best(
    model: GRPModel,
    config: dict,
    output: Path,
    step: int,
    validation_loss: float,
    best_validation_loss: float,
) -> bool:
    """validation loss 更低才覆盖 best.pt;并记录 best_loss.json。"""
    if float(validation_loss) >= float(best_validation_loss) - 1e-12:
        return False
    _write_best_checkpoint(model, config, output, step, validation_loss)
    (output / "best_loss.json").write_text(
        json.dumps({"validation_loss": float(validation_loss), "step": int(step)})
        + "\n", encoding="utf-8",
    )
    return True


def _train_step(
    model: GRPModel,
    optimizer: torch.optim.Optimizer,
    rows: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    config: dict,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = collate(rows, device)
    logits = model(batch["features"], batch["lengths"])
    labels = model.get_label(batch["rank_by_player"])
    loss = prefix_loss(logits, labels)
    # E[U] 辅助回归:排列 CE 只监督分类,而 PPO 奖励直接消费期望 utility;
    # 该项把训练目标与奖励用途对齐(eu_loss_coef=0 时退回纯 CE)。
    eu_loss_coef = float(config.get("eu_loss_coef") or 0.0)
    if eu_loss_coef > 0.0:
        eu_pred = expected_utility_from_logits(logits, model.perms)
        eu_true = true_expected_utility(batch["rank_by_player"])
        loss = loss + eu_loss_coef * F.mse_loss(eu_pred, eu_true)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
    optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/tenhou_grp_2024_2025_v18"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eu-loss-coef", type=float, dest="eu_loss_coef")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.eu_loss_coef is not None:
        config["eu_loss_coef"] = args.eu_loss_coef
    train_grp(args.dataset, config)


if __name__ == "__main__":
    main()
