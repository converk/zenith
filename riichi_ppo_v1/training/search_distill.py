"""Search-distillation dataset: precomputed states plus top-K action targets.

The dataset is produced by ``riichi_ppo_v1.sft.search_head_to_head`` with
``--record-distill-dir``: for every searchable decision of the search team it
stores the exact policy input tensors (token factors/numeric/lengths/legal
mask) and a target action distribution derived from the root-search rollout
values.  The PPO learner samples batches from this dataset and adds
``lambda * KL(policy || search_target)`` to the update loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def search_distill_cross_entropy(
    policy_logits: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    """Forward-KL (cross-entropy) of the target under the policy, per row.

    ``target_probs`` is a sparse distribution (top-3 search candidates, zero
    elsewhere).  Reverse KL(policy || target) is degenerate with sparse targets
    (its partial-sum surrogate can be negative), so distillation minimises
    ``-sum_a target(a) log policy(a)``, which is the standard imitation loss.
    """
    log_probs = torch.log_softmax(policy_logits.float(), dim=-1)
    safe_log_probs = torch.where(
        torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs)
    )
    return -(target_probs.float() * safe_log_probs).sum(-1)


class SearchDistillDataset:
    """In-memory loader over ``part_*.npz`` files produced by the harness."""

    def __init__(
        self,
        directory: str | Path,
        *,
        max_samples: int | None = None,
    ) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ValueError(f"search-distill directory does not exist: {self.directory}")
        self.rows: list[dict[str, np.ndarray]] = []
        for path in sorted(self.directory.glob("part_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                factors = data["factors"]
                numeric = data["numeric"]
                lengths = data["lengths"]
                legal = data["legal"]
                target_probs = data["target_probs"]
            for row in range(int(len(lengths))):
                length = int(lengths[row])
                self.rows.append({
                    "factors": np.ascontiguousarray(
                        factors[row, :length], dtype=np.uint8
                    ),
                    "numeric": np.ascontiguousarray(
                        numeric[row, :length], dtype=np.float32
                    ),
                    "length": length,
                    "legal": np.ascontiguousarray(legal[row], dtype=np.bool_),
                    "target_probs": np.ascontiguousarray(
                        target_probs[row], dtype=np.float32
                    ),
                })
            if max_samples is not None and len(self.rows) >= int(max_samples):
                break
        if not self.rows:
            raise ValueError(f"no search-distill parts found in {self.directory}")

    def __len__(self) -> int:
        return len(self.rows)

    def sample_batch(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> dict[str, np.ndarray]:
        """Return a padded numpy batch of ``size`` random samples."""
        size = max(1, int(size))
        indices = rng.integers(0, len(self.rows), size=size)
        selected = [self.rows[index] for index in indices]
        max_length = max(row["length"] for row in selected)
        batch: dict[str, np.ndarray] = {
            "factors": np.zeros((size, max_length, 10), dtype=np.uint8),
            "numeric": np.zeros((size, max_length, 8), dtype=np.float32),
            "lengths": np.zeros(size, dtype=np.int64),
            "legal": np.zeros((size, 241), dtype=np.bool_),
            "target_probs": np.zeros((size, 241), dtype=np.float32),
        }
        for row_index, row in enumerate(selected):
            batch["factors"][row_index, : row["length"]] = row["factors"]
            batch["numeric"][row_index, : row["length"]] = row["numeric"]
            batch["lengths"][row_index] = row["length"]
            batch["legal"][row_index] = row["legal"]
            batch["target_probs"][row_index] = row["target_probs"]
        return batch

    def metadata(self) -> dict[str, Any]:
        meta_path = self.directory / "meta.json"
        if meta_path.is_file():
            import json

            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {}
