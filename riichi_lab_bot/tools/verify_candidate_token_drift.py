"""Empirically verify the bot and training paths emit identical V13 tokens.

The training `BatchedStateBridge.prepare(decisions, analysis)` injects six
``segment=6`` state rows plus one ``segment=7`` offense/defense query pair
per legal action (see
``riichi_ppo_v1/training/rewards/decision.py:DecisionAnalysisBatch.state_tokens``
and ``candidate_tokens``).  The bot's ``OnlineStateBridge.prepare`` must
reproduce the exact same token sequence and therefore the same model argmax.

This script runs one local RiichiEnv hanchan, encodes every decision with
both paths using the *same* checkpoint, and reports the token/argmax
disagreement count.  The acceptance requirement is ``disagreements = 0``.

Usage (from the workspace root):
    CUDA_DEVICE=2,3 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python \\
        riichi_lab_bot/tools/verify_candidate_token_drift.py \\
        --model checkpoints/train_riichi_v13_sft/best_heuristic.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from riichi_lab_bot.bridge import OnlineStateBridge
from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision
from riichi_ppo_v1.sft.head_to_head import _bf16_supported, _load_model, _tensor
from riichi_ppo_v1.training.rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)


def _play_one_hanchan(
    model: Any,
    device: torch.device,
    use_bf16: bool,
    *,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """Play one hanchan; for each model-seat decision, compute argmax under
    both the bot's OnlineStateBridge and the training BatchedStateBridge,
    using the same checkpoint."""
    from riichienv import BatchedRiichiEnv
    import riichi

    envs = BatchedRiichiEnv(1, seed=seed, step_threads=1, game_mode="4p-red-half")
    train_bridge = BatchedStateBridge(riichi.MjaiKyokuStateMachineManager(1), 1)
    bot_bridges = {seat: OnlineStateBridge(seat) for seat in range(4)}
    pending_events = {seat: [] for seat in range(4)}
    analyzer = EfficiencyAnalyzer(131_072)
    public = PublicStateTracker(1)

    observations = list(envs.reset())
    train_bridge.sync(observations)
    public.update(train_bridge.last_events)

    decisions_checked = 0
    disagreements = 0
    same_actions = 0
    bot_only_samples = 0
    train_only_samples = 0
    sample_diffs: list[dict[str, Any]] = []
    bot_token_lengths: list[int] = []
    train_token_lengths: list[int] = []

    active_envs = {0}
    for _step in range(max_steps):
        actions_by_env: list[dict[int, Any]] = [{}]
        decisions = []
        # Gather pending events into each seat's observation (bot path).
        for seat, obs in observations[0].items():
            pending_events[int(seat)].extend(obs.new_events())

        for seat, original in observations[0].items():
            if not original.legal_actions():
                continue
            seat_i = int(seat)
            # Bot path: build observation with events and prepare.
            from riichi_lab_bot.local_play import observation_with_events
            bot_obs = observation_with_events(original, pending_events[seat_i])
            pending_events[seat_i].clear()
            prepared_bot = bot_bridges[seat_i].prepare(bot_obs)

            # Training path: prepare with analysis -> candidate tokens.
            train_decision = Decision(0, seat_i, original)
            analysis = DecisionAnalysisBatch.build(
                [train_decision], analyzer=analyzer, public=public,
            )
            (tf, tn, tl, tm, _tg, _tc, _tcl) = train_bridge.prepare(
                [train_decision], analysis
            )

            # Run both through the same model policy head.
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    bot_logits = model.forward_policy(
                        _tensor(prepared_bot.token_factors[None], device),
                        _tensor(prepared_bot.token_numeric[None], device),
                        _tensor(prepared_bot.legal_mask[None], device),
                        torch.tensor([prepared_bot.token_length], device=device, dtype=torch.long),
                    )["policy_logits"].to(torch.float32)
                    train_logits = model.forward_policy(
                        _tensor(tf, device),
                        _tensor(tn, device),
                        _tensor(tm, device),
                        _tensor(tl, device),
                    )["policy_logits"].to(torch.float32)

            bot_argmax = int(bot_logits.argmax(-1).item())
            # Train_logits shape is [1, 241]; argmax.
            train_argmax = int(train_logits.argmax(-1).item())

            bot_token_lengths.append(int(prepared_bot.token_length))
            train_token_lengths.append(int(tl[0]))

            decisions_checked += 1
            if bot_argmax != train_argmax:
                disagreements += 1
                if len(sample_diffs) < 10:
                    sample_diffs.append({
                        "step": _step,
                        "seat": seat_i,
                        "bot_action_id": bot_argmax,
                        "train_action_id": train_argmax,
                        "bot_token_len": int(prepared_bot.token_length),
                        "train_token_len": int(tl[0]),
                        "bot_top3": _top3(bot_logits),
                        "train_top3": _top3(train_logits),
                    })
            else:
                same_actions += 1

            # Step env with either action (use bot argmax to advance).
            action = bot_bridges[seat_i].decode(prepared_bot, bot_argmax)
            actions_by_env[0][seat_i] = action

        observations = list(envs.step_batch(actions_by_env))
        train_bridge.sync(observations)
        public.update(train_bridge.last_events)
        if bool(envs.done()[0]):
            break
    else:
        raise RuntimeError(f"hanchan exceeded {max_steps} steps")

    return {
        "seed": seed,
        "decisions_checked": decisions_checked,
        "disagreements": disagreements,
        "same_actions": same_actions,
        "disagreement_rate_pct": 100.0 * disagreements / max(1, decisions_checked),
        "bot_only_samples": bot_only_samples,
        "train_only_samples": train_only_samples,
        "avg_bot_token_len": float(np.mean(bot_token_lengths)) if bot_token_lengths else 0.0,
        "avg_train_token_len": float(np.mean(train_token_lengths)) if train_token_lengths else 0.0,
        "sample_diffs": sample_diffs,
    }


def _top3(logits: torch.Tensor) -> list[tuple[int, float]]:
    vals, idx = torch.topk(logits.flatten(), k=min(3, logits.numel()))
    return [(int(i), float(v)) for i, v in zip(idx.tolist(), vals.tolist())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = _load_model(args.model, device)
    use_bf16 = _bf16_supported(device)

    started = time.perf_counter()
    report = _play_one_hanchan(
        model, device, use_bf16, seed=args.seed, max_steps=args.max_steps
    )
    elapsed = time.perf_counter() - started

    print("\n" + "=" * 80)
    print(f"seed={args.seed} decisions={report['decisions_checked']} elapsed={elapsed:.1f}s")
    print(f"disagreements: {report['disagreements']} / {report['decisions_checked']} "
          f"({report['disagreement_rate_pct']:.2f}%)")
    print(f"same actions: {report['same_actions']}")
    print(f"avg bot token len:   {report['avg_bot_token_len']:.1f}")
    print(f"avg train token len: {report['avg_train_token_len']:.1f}")
    if report["sample_diffs"]:
        print("\nfirst disagreement examples:")
        for diff in report["sample_diffs"][:5]:
            print(f"  step={diff['step']} seat={diff['seat']}: "
                  f"bot={diff['bot_action_id']} train={diff['train_action_id']} "
                  f"(bot_len={diff['bot_token_len']}, train_len={diff['train_token_len']})")
            print(f"    bot_top3:   {diff['bot_top3']}")
            print(f"    train_top3: {diff['train_top3']}")

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
