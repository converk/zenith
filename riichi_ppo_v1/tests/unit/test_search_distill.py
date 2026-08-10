"""Unit tests for the search-distillation dataset round-trip."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from riichi_ppo_v1.training.search_distill import search_distill_cross_entropy
from riichi_ppo_v1.training.search_distill import SearchDistillDataset
from riichi_ppo_v1.sft.search_head_to_head import DistillRecorder


def _write_part(directory, rows):
    max_length = max(row["length"] for row in rows)
    count = len(rows)
    factors = np.zeros((count, max_length, 10), dtype=np.uint8)
    numeric = np.zeros((count, max_length, 8), dtype=np.float32)
    lengths = np.zeros(count, dtype=np.int64)
    legal = np.zeros((count, 241), dtype=np.bool_)
    target_probs = np.zeros((count, 241), dtype=np.float32)
    for row_index, row in enumerate(rows):
        factors[row_index, : row["length"]] = row["factors"]
        numeric[row_index, : row["length"]] = row["numeric"]
        lengths[row_index] = row["length"]
        legal[row_index] = row["legal"]
        target_probs[row_index] = row["target_probs"]
    np.savez_compressed(
        directory / "part_00000.npz",
        factors=factors,
        numeric=numeric,
        lengths=lengths,
        legal=legal,
        target_probs=target_probs,
    )


def test_dataset_round_trip(tmp_path):
    rows = [
        {
            "factors": np.random.default_rng(1).integers(
                0, 4, size=(11, 10), dtype=np.uint8
            ),
            "numeric": np.random.default_rng(2).random((11, 8), dtype=np.float32),
            "length": 11,
            "legal": np.zeros(241, dtype=np.bool_),
            "target_probs": np.zeros(241, dtype=np.float32),
        },
        {
            "factors": np.random.default_rng(3).integers(
                0, 4, size=(17, 10), dtype=np.uint8
            ),
            "numeric": np.random.default_rng(4).random((17, 8), dtype=np.float32),
            "length": 17,
            "legal": np.zeros(241, dtype=np.bool_),
            "target_probs": np.zeros(241, dtype=np.float32),
        },
    ]
    rows[0]["legal"][1:6] = True
    rows[0]["target_probs"][[2, 3, 4]] = np.asarray([0.5, 0.3, 0.2], dtype=np.float32)
    rows[1]["legal"][10:14] = True
    rows[1]["target_probs"][[10, 11]] = np.asarray([0.7, 0.3], dtype=np.float32)
    _write_part(tmp_path, rows)

    dataset = SearchDistillDataset(tmp_path)
    assert len(dataset) == 2
    rng = np.random.default_rng(7)
    batch = dataset.sample_batch(rng, size=4)
    assert batch["factors"].shape == (4, 17, 10)
    assert batch["numeric"].shape == (4, 17, 8)
    assert batch["lengths"].shape == (4,)
    assert batch["legal"].shape == (4, 241)
    assert batch["target_probs"].shape == (4, 241)
    for row_index in range(batch["target_probs"].shape[0]):
        probs = batch["target_probs"][row_index]
        if np.isclose(probs[[2, 3, 4]].sum(), 1.0):
            np.testing.assert_allclose(
                probs[[2, 3, 4]],
                np.asarray([0.5, 0.3, 0.2], dtype=np.float32),
            )
        elif np.isclose(probs[[10, 11]].sum(), 1.0):
            np.testing.assert_allclose(
                probs[[10, 11]],
                np.asarray([0.7, 0.3], dtype=np.float32),
            )
        else:
            raise AssertionError("sampled target distribution was corrupted")


def test_dataset_rejects_missing_directory(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        SearchDistillDataset(tmp_path / "missing")


def test_dataset_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no search-distill parts"):
        SearchDistillDataset(tmp_path)


def test_recorder_round_trip_preserves_variable_lengths(tmp_path):
    recorder = DistillRecorder(tmp_path, tau=2.0)
    recorder.add(
        factors=np.zeros((57, 10), dtype=np.uint8),
        numeric=np.zeros((57, 8), dtype=np.float32),
        length=57,
        legal=np.zeros(241, dtype=np.bool_),
        target_probs=np.zeros(241, dtype=np.float32),
    )
    recorder.add(
        factors=np.zeros((63, 10), dtype=np.uint8),
        numeric=np.zeros((63, 8), dtype=np.float32),
        length=63,
        legal=np.zeros(241, dtype=np.bool_),
        target_probs=np.zeros(241, dtype=np.float32),
    )
    recorder.close(seed_base=1)
    dataset = SearchDistillDataset(tmp_path)
    assert len(dataset) == 2
    assert dataset.rows[0]["factors"].shape == (57, 10)
    assert dataset.rows[0]["numeric"].shape == (57, 8)
    assert dataset.rows[1]["factors"].shape == (63, 10)
    assert dataset.metadata()["seed_base"] == 1


def test_cross_entropy_loss_is_positive_and_correct():
    logits = torch.tensor([[1.0, 2.0, 3.0, -10.0, -10.0]])
    targets = torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.0]])
    ce = search_distill_cross_entropy(logits, targets)
    assert ce.shape == (1,)
    assert bool(ce[0] >= 0.0)
    expected = -0.5 * torch.log_softmax(logits, dim=-1)[0, 1:3].sum()
    torch.testing.assert_close(ce, expected[None], atol=1e-6, rtol=1e-6)


def test_cross_entropy_never_negative_on_sparse_targets():
    rng = torch.Generator().manual_seed(3)
    logits = torch.randn(64, 241, generator=rng)
    targets = torch.zeros(64, 241)
    for row in range(64):
        ids = torch.randperm(241, generator=rng)[:3]
        targets[row, ids] = torch.softmax(
            torch.randn(3, generator=rng), dim=0
        )
    ce = search_distill_cross_entropy(logits, targets)
    assert bool(torch.all(ce >= -1e-6))


def test_cross_entropy_handles_masked_illegal_logits():
    logits = torch.tensor([[1.0, 2.0, -float("inf"), -float("inf")]])
    targets = torch.tensor([[0.3, 0.7, 0.0, 0.0]])
    ce = search_distill_cross_entropy(logits, targets)
    assert bool(torch.isfinite(ce).all())
    assert bool(ce[0] >= 0.0)
