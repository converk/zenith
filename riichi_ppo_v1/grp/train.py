"""Offline GRP training and validation entry point (CPU-friendly)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .dataset import iter_grp_rows
from .model import (
    FEATURE_DIM,
    NUM_PLAYERS,
    RankPredictor,
    reward_from_rank_probs,
    save_grp_checkpoint,
)


def _default_zip_paths() -> list[str]:
    return [
        "datasets/tenhou-to-mjai/2024.zip",
        "datasets/tenhou-to-mjai/2025.zip",
    ]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-paths", nargs="*", default=None)
    parser.add_argument("--subset-denominator", type=int, default=5)
    parser.add_argument("--subset-remainders", type=int, nargs="*", default=[0])
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--val-games", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="checkpoints/train_riichi_ppo_goal_grp/grp_rank_predictor.pt")
    parser.add_argument("--validation-json", default="audit/reports/ppo_rl_goal_run_20260808/e3_grp/grp_validation.json")
    return parser.parse_args()


def _load_split(
    zip_paths: list[str],
    *,
    subset_denominator: int,
    subset_remainders: list[int],
    max_games: int | None,
    val_games: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    deltas: list[float] = []
    for row, label, delta in iter_grp_rows(
        zip_paths,
        subset_denominator=subset_denominator,
        subset_remainders=subset_remainders,
        max_games=max_games,
    ):
        rows.append(row)
        labels.append(label)
        deltas.append(delta)
    rows_arr = np.asarray(rows, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64)
    deltas_arr = np.asarray(deltas, dtype=np.float32)
    permutation = rng.permutation(len(rows_arr))
    rows_arr, labels_arr, deltas_arr = (
        rows_arr[permutation],
        labels_arr[permutation],
        deltas_arr[permutation],
    )
    val_count = val_games if val_games is not None else max(1, len(rows_arr) // 10)
    val_count = min(val_count, len(rows_arr))
    return (
        rows_arr[:-val_count],
        labels_arr[:-val_count],
        deltas_arr[:-val_count],
        rows_arr[-val_count:],
        labels_arr[-val_count:],
        deltas_arr[-val_count:],
    )


def _accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(labels == predictions))


def _ece(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    confidence = np.max(probs, axis=-1)
    correct = labels == np.argmax(probs, axis=-1)
    bin_boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    count = 0
    for low, high in zip(bin_boundaries[:-1], bin_boundaries[1:], strict=True):
        if high == 1.0:
            mask = confidence >= low
        else:
            mask = (confidence >= low) & (confidence < high)
        if not mask.any():
            continue
        acc = float(np.mean(correct[mask]))
        conf = float(np.mean(confidence[mask]))
        total += len(mask) * abs(acc - conf)
        count += len(mask)
    return float(total / max(count, 1))


def main() -> None:
    args = _args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    zip_paths = args.zip_paths or _default_zip_paths()
    (
        train_x,
        train_y,
        train_delta,
        val_x,
        val_y,
        val_delta,
    ) = _load_split(
        zip_paths,
        subset_denominator=args.subset_denominator,
        subset_remainders=args.subset_remainders,
        max_games=args.max_games,
        val_games=args.val_games,
        seed=args.seed,
    )
    print(
        f"loaded train={len(train_x)} val={len(val_x)} "
        f"train_label_dist={np.bincount(train_y, minlength=4).tolist()}",
        flush=True,
    )
    model = RankPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    criterion = torch.nn.CrossEntropyLoss()
    steps_per_epoch = max(1, (len(train_x) + args.batch_size - 1) // args.batch_size)
    for epoch in range(int(args.epochs)):
        order = np.random.permutation(len(train_x))
        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            features = torch.as_tensor(train_x[index], dtype=torch.float32)
            labels = torch.as_tensor(train_y[index], dtype=torch.long)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            batches += 1
            if batches % 200 == 0:
                print(
                    f"epoch={epoch + 1} batch={batches}/{steps_per_epoch} loss={total_loss / batches:.4f}",
                    flush=True,
                )
    model.eval()
    val_probs = model.predict_rank_probs(val_x)
    val_pred = np.argmax(val_probs, axis=-1)
    val_acc = _accuracy(val_y, val_pred)
    val_top2 = float(np.mean([
        int(val_y[row]) in np.argsort(val_probs[row])[-2:]
        for row in range(len(val_y))
    ]))
    val_ece = _ece(val_y, val_probs)
    pts_weight = [10, 4, -4, -10]
    grp_rewards = np.asarray([
        reward_from_rank_probs(probs, pts_weight=pts_weight)
        for probs in val_probs
    ])
    point_delta = np.clip(val_delta, -24.0, 24.0)
    correlation = float(spearmanr(grp_rewards, point_delta).statistic)
    baseline = 0.25
    result = {
        "train_rows": int(len(train_x)),
        "val_rows": int(len(val_x)),
        "val_rank_accuracy": val_acc,
        "val_rank_accuracy_baseline": baseline,
        "val_top2_accuracy": float(val_top2),
        "val_ece": val_ece,
        "grp_point_delta_spearman": correlation,
        "pts_weight": pts_weight,
        "val_label_distribution": np.bincount(val_y, minlength=4).tolist(),
        "subset_denominator": args.subset_denominator,
        "subset_remainders": args.subset_remainders,
        "model": "mlp_20_128_64_4",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_grp_checkpoint(
        model,
        str(output_path),
        validation=result,
        train_config=vars(args),
    )
    validation_path = Path(args.validation_json)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
