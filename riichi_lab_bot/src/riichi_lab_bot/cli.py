"""Command-line entry points."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from .client import (
    RANKED_URL,
    VALIDATION_URL,
    play_connection,
    run_ranked,
)
from .local_play import play_local_game
from .policy import PolicyEngine
from .telemetry import EventRecorder


def _default_checkpoint() -> str:
    override = os.environ.get("RIICHI_CHECKPOINT")
    if override:
        return override
    repository = Path(__file__).resolve().parents[3]
    return str(
        repository
        / "checkpoints"
        / "train_riichi_v13_sft"
        / "best_heuristic.pt"
    )


def _common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--checkpoint", default=_default_checkpoint()
    )
    common.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:index (default: auto)",
    )
    common.add_argument(
        "--dtype",
        choices=("auto", "fp32", "bf16"),
        default="auto",
    )
    common.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    common.add_argument("--jsonl-log", default=None)
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riichi-lab-bot",
        description="Standalone Zenith checkpoint client for RiichiLab",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()

    local = subparsers.add_parser(
        "local", parents=[common], help="run local RiichiEnv games"
    )
    local.add_argument("--games", type=int, default=3)
    local.add_argument("--seed", type=int, default=20260730)
    local.add_argument("--max-steps", type=int, default=4000)

    validate = subparsers.add_parser(
        "validate", parents=[common], help="run one validation game"
    )
    validate.add_argument("--url", default=VALIDATION_URL)

    ranked = subparsers.add_parser(
        "ranked", parents=[common], help="join ranked matchmaking"
    )
    ranked.add_argument("--url", default=RANKED_URL)
    group = ranked.add_mutually_exclusive_group()
    group.add_argument("--games", type=int, default=1)
    group.add_argument("--forever", action="store_true")
    return parser


def _load_policy(args: argparse.Namespace, recorder: EventRecorder) -> PolicyEngine:
    policy = PolicyEngine(
        args.checkpoint, device=args.device, dtype=args.dtype
    )
    warmup_ms = policy.warmup()
    recorder.emit(
        "model_loaded",
        checkpoint=str(policy.checkpoint),
        device=str(policy.device),
        dtype=policy.dtype_name,
        token_schema_version=policy.metadata["token_schema_version"],
        sft_contract_version=policy.metadata["sft_contract_version"],
        policy_head_type=policy.metadata["policy_head_type"],
        warmup_ms=warmup_ms,
    )
    return policy


def _online_token() -> str:
    token = os.environ.get("RIICHI_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "RIICHI_BOT_TOKEN is required for online commands"
        )
    return token


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    recorder = EventRecorder(args.jsonl_log)
    try:
        policy = _load_policy(args, recorder)
        if args.command == "local":
            if args.games < 1:
                raise ValueError("--games must be positive")
            results = []
            for game in range(args.games):
                result = play_local_game(
                    policy,
                    game=game + 1,
                    seed=args.seed + game,
                    max_steps=args.max_steps,
                    recorder=recorder,
                )
                payload = {
                    **result.__dict__,
                    "warmup": game == 0,
                    "decisions_per_second": (
                        result.metrics["requests"]
                        / max(result.elapsed_seconds, 1e-9)
                    ),
                }
                recorder.emit("local_game_result", **payload)
                results.append(payload)
            measured = results[1:] if len(results) > 1 else results
            summary = {
                "games": len(results),
                "warmup_games": 1 if len(results) > 1 else 0,
                "measured_games": len(measured),
                "measured_elapsed_seconds": sum(
                    item["elapsed_seconds"] for item in measured
                ),
                "measured_decisions": sum(
                    item["metrics"]["requests"] for item in measured
                ),
            }
            summary["measured_decisions_per_second"] = (
                summary["measured_decisions"]
                / max(summary["measured_elapsed_seconds"], 1e-9)
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return

        token = _online_token()
        if args.command == "validate":
            result = asyncio.run(
                play_connection(
                    url=args.url,
                    mode="validate",
                    token=token,
                    policy=policy,
                    recorder=recorder,
                )
            )
            print(
                json.dumps(
                    result.__dict__, ensure_ascii=False, indent=2
                )
            )
            if result.validation_passed is not True:
                raise SystemExit(2)
            return

        games = None if args.forever else args.games
        if games is not None and games < 1:
            raise ValueError("--games must be positive")
        results = asyncio.run(
            run_ranked(
                url=args.url,
                token=token,
                policy=policy,
                recorder=recorder,
                games=games,
            )
        )
        print(
            json.dumps(
                [result.__dict__ for result in results],
                ensure_ascii=False,
                indent=2,
            )
        )
    except KeyboardInterrupt:
        recorder.emit("interrupted")
        raise SystemExit(130) from None
    except SystemExit:
        raise
    except Exception as exc:
        recorder.emit(
            "fatal_error", error=f"{type(exc).__name__}: {exc}"
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
