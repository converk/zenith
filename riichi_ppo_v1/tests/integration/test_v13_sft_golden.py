from __future__ import annotations

from itertools import islice
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from riichi_ppo_v1.sft.checkpoint import load_v13_weights_only
from riichi_ppo_v1.sft.data import iter_split_samples
from riichi_ppo_v1.sft.train import (
    collate_samples, group_classification_loss, rule_teacher_loss,
)


ROOT = Path(__file__).parents[3]
DATASET = ROOT / "datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16"
CHECKPOINT = ROOT / "checkpoints/train_riichi_v13/sft/best_heuristic.pt"


@pytest.mark.skipif(
    not DATASET.is_dir() or not CHECKPOINT.is_file(),
    reason="formal v13 fixture unavailable",
)
def test_formal_v13_loader_logits_actions_and_losses_are_golden() -> None:
    rows = list(islice(iter_split_samples(
        DATASET, "validation", seed=1, shuffle=False, include_critic=False,
    ), 4))
    assert [
        (row.year, row.game_id, row.kyoku_index, row.seat, row.decision_index)
        for row in rows
    ] == [
        (2024, "2024010102gm-00a9-0000-9164c72c", 0, 0, index)
        for index in range(4)
    ]
    batch = collate_samples(rows, torch.device("cpu"), include_critic=False)
    assert batch["token_lengths"].tolist() == [57, 62, 70, 75]
    assert batch["actions"].tolist() == [61, 19, 37, 65]
    model = load_v13_weights_only(CHECKPOINT, device="cpu")
    with torch.inference_mode():
        output = model.forward_policy(
            batch["token_factors"], batch["token_numeric"],
            batch["legal_mask"], batch["token_lengths"],
        )
    logits = output["policy_logits"].float()
    assert logits.argmax(-1).tolist() == [61, 19, 37, 65]
    torch.testing.assert_close(
        output["raw_policy_logits"][torch.arange(4), logits.argmax(-1)],
        torch.tensor([3.4513016, 4.2165408, 3.8457525, 0.7265501]),
        rtol=1e-6, atol=1e-6,
    )
    losses = torch.stack((
        F.cross_entropy(logits, batch["actions"]),
        group_classification_loss(logits, batch["actions"]),
        rule_teacher_loss(logits, batch["teacher_masks"]),
    ))
    torch.testing.assert_close(
        losses,
        torch.tensor([0.27561104, 0.0, 5.0784116]),
        rtol=1e-6, atol=1e-6,
    )
