from __future__ import annotations

import atexit
import os
import sys
import tempfile
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
REPOSITORY = PROJECT.parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def repository_root() -> Path:
    return PROJECT.parent


def default_checkpoint() -> Path:
    override = os.environ.get("BOT_TEST_CHECKPOINT")
    if override:
        return Path(override).expanduser().resolve()
    from riichi_ppo_v1.model import KyokuTransformerActorCritic

    path = Path(tempfile.gettempdir()) / f"riichi_lab_bot_v18_{os.getpid()}.pt"
    if not path.exists():
        model = KyokuTransformerActorCritic()
        torch.save({
            "model_config": vars(model.config),
            "model": model.state_dict(),
            "token_schema_version": 18,
        }, path)
        atexit.register(path.unlink, missing_ok=True)
    return path


def v16_sft_checkpoint() -> Path:
    return (
        repository_root()
        / "checkpoints"
        / "train_riichi_v16"
        / "sft"
        / "best.pt"
    )


def v17_ppo_checkpoint() -> Path:
    return (
        repository_root()
        / "checkpoints"
        / "train_riichi_v17"
        / "archive_20260819_V1run1"
        / "ppo"
        / "latest.pt"
    )
