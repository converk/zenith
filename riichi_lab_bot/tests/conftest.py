from __future__ import annotations

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def repository_root() -> Path:
    return PROJECT.parent


def default_checkpoint() -> Path:
    return (
        repository_root()
        / "checkpoints"
        / "train_riichi_v10_sft"
        / "best_heuristic.pt"
    )

