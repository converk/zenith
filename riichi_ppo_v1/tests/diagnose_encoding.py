"""Inspect semantic public and centralized-critic model inputs from RiichiEnv."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from typing import Any

from ..model.bridge import BatchedStateBridge, Decision
from ..model.semantic_validation import (
    assert_actor_token_semantics,
    assert_critic_token_semantics,
    summarize_tokens,
)


def run_diagnostics(games: int = 3, seed: int = 20260719, max_steps: int = 2500, *, include_public_state: bool = True) -> dict[str, Any]:
    """Run deterministic games and return a human/audit-friendly semantic report."""
    import riichi
    from riichienv import BatchedRiichiEnv

    rng = random.Random(seed)
    started = time.perf_counter()
    reports: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for game in range(games):
        game_started = time.perf_counter()
        env = BatchedRiichiEnv(1, seed=seed + game, step_threads=1)
        bridge = BatchedStateBridge(
            riichi.MjaiKyokuStateMachineManager(1), 1, critic_include_public_state=include_public_state,
        )
        observations = list(env.reset())
        bridge.sync(observations)
        decision_reports: list[dict[str, Any]] = []
        for step in range(max_steps):
            decisions = [Decision(0, int(seat), observation) for seat, observation in observations[0].items() if observation.legal_actions()]
            if not decisions:
                if env.done()[0]:
                    break
                raise AssertionError(f"environment stalled in game={game}, step={step}")
            factors, numeric, lengths, masks, _generation, critic, critic_lengths = bridge.prepare(decisions)
            assert_actor_token_semantics(factors, numeric, lengths)
            assert_critic_token_semantics(critic, critic_lengths, include_public_state=include_public_state)
            for row, decision in enumerate(decisions):
                public = summarize_tokens(factors[row], int(lengths[row]))
                private = summarize_tokens(critic[row], int(critic_lengths[row]))
                decision_reports.append({
                    "step": step, "seat": decision.seat_id, "legal_actions": int(masks[row].sum()),
                    "public": public, "critic": private,
                })
                totals.update(public["by_segment_kind_field"])
                totals.update({f"critic:{key}": value for key, value in private["by_segment_kind_field"].items()})
            actions = {decision.seat_id: rng.choice(decision.observation.legal_actions()) for decision in decisions}
            observations = list(env.step_batch([actions]))
            bridge.sync(observations)
            if env.done()[0]:
                break
        else:
            raise AssertionError(f"game={game} exceeded max_steps={max_steps}")
        reports.append({"game": game, "decisions": decision_reports, "elapsed_s": time.perf_counter() - game_started})
    return {
        "games": games, "seed": seed, "include_public_state": include_public_state,
        "warmup_game": 0, "elapsed_s": time.perf_counter() - started,
        "post_warmup_elapsed_s": [report["elapsed_s"] for report in reports[1:]],
        "token_totals": dict(sorted(totals.items())), "games_detail": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--no-critic-public-state", action="store_false", dest="include_public_state")
    args = parser.parse_args()
    report = run_diagnostics(args.games, args.seed, args.max_steps, include_public_state=args.include_public_state)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(f"games={report['games']} seed={report['seed']} critic_public={report['include_public_state']} elapsed_s={report['elapsed_s']:.3f}")
    for game in report["games_detail"]:
        print(f"game={game['game']} decisions={len(game['decisions'])} elapsed_s={game['elapsed_s']:.3f}" + (" warmup" if game["game"] == 0 else ""))
        for decision in game["decisions"][:8]:
            print(f"  step={decision['step']} seat={decision['seat']} legal={decision['legal_actions']} public={decision['public']} critic={decision['critic']}")
    print("token_totals=" + json.dumps(report["token_totals"], sort_keys=True))


if __name__ == "__main__":
    main()
