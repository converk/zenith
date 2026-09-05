"""V19 actor-only SFT（Actor BC + 信念监督联合）CLI 入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .trainer import (
    DEFAULT_CONFIG,
    collate_samples,
    evaluate,
    length_bucketed_batches,
    load_config,
    train_worker,
    validate_config,
)
from .trainer import (
    main as train_main,
)

__all__ = (
    "DEFAULT_CONFIG",
    "collate_samples",
    "evaluate",
    "length_bucketed_batches",
    "load_config",
    "train_worker",
    "validate_config",
    "main",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.dataset is not None:
        config["dataset"] = str(args.dataset)
    if args.output is not None:
        config["checkpoint_dir"] = str(args.output)
    if "dataset" not in config:
        raise SystemExit("--dataset is required when the config does not define dataset")
    if "checkpoint_dir" not in config:
        raise SystemExit("--output is required when the config does not define checkpoint_dir")
    validate_config(config)
    train_main(config, dataset=args.dataset, output=args.output)


if __name__ == "__main__":
    main()
