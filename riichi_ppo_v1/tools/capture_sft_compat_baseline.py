"""Capture field-level SFT compatibility fixtures for structural refactors.

This is an audit tool, not part of the training call path.  Outputs belong in
temporary storage and deliberately include arrays rather than opaque hashes so
that a mismatch can be localized to a token row, logit, or loss component.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import tarfile

import numpy as np
import torch
from torch.nn import functional as F

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
from riichi_ppo_v1.sft.data import encode_kyoku, iter_split_samples
from riichi_ppo_v1.sft.train import (
    collate_samples,
    group_classification_loss,
    rule_teacher_loss,
)


def _first_raw_kyoku(dataset: Path) -> tuple[str, str]:
    shard = sorted((dataset / "validation").glob("validation-*.tar"))[0]
    with tarfile.open(shard, "r") as archive:
        member = next(item for item in archive.getmembers() if item.isfile())
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"cannot extract {shard}:{member.name}")
        payload = extracted.read()
    content = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    return f"{shard}:{member.name}", content.decode("utf-8")


def _load(path: Path) -> KyokuTransformerActorCritic:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(payload["model_config"])
    model = KyokuTransformerActorCritic(ModelConfig(**config))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def _sample_arrays(prefix: str, sample: object, output: dict[str, np.ndarray]) -> None:
    output[f"{prefix}_factors"] = sample.token_factors
    output[f"{prefix}_numeric"] = sample.token_numeric
    output[f"{prefix}_legal"] = sample.legal_mask
    output[f"{prefix}_teacher"] = sample.teacher_mask
    output[f"{prefix}_action"] = np.asarray(sample.action, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--encoded", type=Path, required=True)
    parser.add_argument("--v13-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(1)
    identity, content = _first_raw_kyoku(args.raw)
    raw_v13 = encode_kyoku(content, include_critic=False)
    encoded = list(
        zip(
            range(4),
            iter_split_samples(
                args.encoded, "validation", shuffle=False, include_critic=False,
            ),
        )
    )
    rows = [sample for _index, sample in encoded]
    batch = collate_samples(rows, torch.device("cpu"), include_critic=False)
    v13 = _load(args.v13_checkpoint)
    with torch.inference_mode():
        v13_output = v13.forward_policy(
            batch["token_factors"], batch["token_numeric"],
            batch["legal_mask"], batch["token_lengths"],
        )
    logits = v13_output["policy_logits"].float()
    losses = {
        "policy_ce": float(F.cross_entropy(logits, batch["actions"])),
        "group": float(group_classification_loss(logits, batch["actions"])),
        "rule": float(rule_teacher_loss(logits, batch["teacher_masks"])),
    }
    arrays: dict[str, np.ndarray] = {
        "batch_factors": batch["token_factors"].numpy(),
        "batch_numeric": batch["token_numeric"].numpy(),
        "batch_lengths": batch["token_lengths"].numpy(),
        "batch_legal": batch["legal_mask"].numpy(),
        "batch_actions": batch["actions"].numpy(),
        "batch_teachers": batch["teacher_masks"].numpy(),
        "v13_logits": v13_output["raw_policy_logits"].numpy(),
        "v13_actions": logits.argmax(-1).numpy(),
    }
    _sample_arrays("raw_v13", raw_v13[0], arrays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), **arrays)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "raw_identity": identity,
                "raw_v13_samples": len(raw_v13),
                "encoded_identities": [
                    [sample.year, sample.game_id, sample.kyoku_index, sample.seat, sample.decision_index]
                    for sample in rows
                ],
                "losses": losses,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(args.output.with_suffix(".json").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
