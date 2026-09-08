"""命令行入口:独立启动 Akagi 云端推理服务。

    akagi-bridge --checkpoint checkpoints/train_riichi_v18/ppo/best.pt \
      --device cuda:0 --dtype fp32 --host 0.0.0.0 --port 8090 \
      --api-key "$AKAGI_API_KEY"
"""

from __future__ import annotations

import argparse
import logging
import os

from .engine import AkagiDecisionEngine
from .server import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="akagi-bridge",
        description="Akagi v3 cloud-inference server for Zenith V18 checkpoints",
    )
    parser.add_argument(
        "--checkpoint", default=os.environ.get("RIICHI_CHECKPOINT")
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:index (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp32", "bf16"),
        default="auto",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("AKAGI_API_KEY"),
        help="expected bearer token (default: $AKAGI_API_KEY, empty = no auth)",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--model-id", default="zenith-v18")
    parser.add_argument(
        "--rule",
        choices=("tenhou", "mjsoul"),
        default="tenhou",
        help="rule used when rebuilding the Observation from mjai events",
    )
    parser.add_argument("--jsonl-log", default=None)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint 或环境变量 RIICHI_CHECKPOINT 必须提供")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from riichi_lab_bot.policy import PolicyEngine
    from riichi_lab_bot.telemetry import EventRecorder

    recorder = EventRecorder(args.jsonl_log)
    policy = PolicyEngine(
        args.checkpoint, device=args.device, dtype=args.dtype
    )
    warmup_ms = policy.warmup()
    recorder.emit(
        "model_loaded",
        checkpoint=str(policy.checkpoint),
        checkpoint_format=policy.metadata["checkpoint_format"],
        device=str(policy.device),
        dtype=policy.dtype_name,
        token_schema_version=policy.metadata["token_schema_version"],
        sft_contract_version=policy.metadata["sft_contract_version"],
        policy_head_type=policy.metadata["policy_head_type"],
        warmup_ms=warmup_ms,
    )
    engine = AkagiDecisionEngine(policy, topk=args.topk, rule=args.rule)
    serve(
        engine,
        host=args.host,
        port=args.port,
        api_key=args.api_key or None,
        model_id=args.model_id,
        model_desc=(
            f"{policy.metadata['checkpoint_format']} "
            f"{policy.metadata['token_schema_version']} "
            f"({policy.dtype_name} on {policy.device})"
        ),
        recorder=recorder,
    )


if __name__ == "__main__":
    main()
