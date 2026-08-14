"""In-process SFT policy evaluation against fixed heuristic opponents."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..model.bridge import NUM_PLAYERS, BatchedStateBridge
from ..training.metrics import SemanticMetrics
from ..training.opponents.heuristic import HeuristicPolicy
from ..training.rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)
from ..training.worker import active_decisions
from .evaluation_cases import DEFENSE, EFFICIENCY, evaluation_cases
from ..evaluation.policy_adapter import PolicyAdapter, V13PolicyAdapter
from .contract import SFT_FINAL_EVAL_HANCHAN_COUNT


def _phase(discard_count: int) -> str:
    if int(discard_count) < 24:
        return "early"
    if int(discard_count) < 48:
        return "middle"
    return "late"


@torch.inference_mode()
def evaluate_against_heuristics(
    model: nn.Module | PolicyAdapter,
    device: torch.device,
    config: dict[str, Any],
    *,
    hanchan_count: int | None = None,
    cycle: int = 0,
) -> dict[str, float]:
    """Play deterministic rotating-seat hanchans and return SFT-prefixed metrics."""
    try:
        import riichi
        from riichienv import BatchedRiichiEnv
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "install the local riichi and RiichiEnv extensions before SFT evaluation"
        ) from exc

    count = int(
        config.get(
            "heuristic_evaluation_hanchan_count",
            SFT_FINAL_EVAL_HANCHAN_COUNT,
        )
        if hanchan_count is None else hanchan_count
    )
    cases = evaluation_cases(
        int(config.get("heuristic_evaluation_seed_base", 0)),
        count,
        cycle=int(cycle),
    )
    parallel_hanchans = max(
        1, int(config.get("heuristic_evaluation_parallel_hanchan_count", 1)),
    )
    metrics = SemanticMetrics()
    if hasattr(model, "prepare") and hasattr(model, "masked_logits"):
        adapter = model
        policy_model = adapter.model
    else:
        policy_model = model
        adapter = V13PolicyAdapter(policy_model, device, Path("in-memory-v13"))
    was_training = policy_model.training
    policy_model.eval()
    started = time.perf_counter()

    for batch_start in range(0, count, parallel_hanchans):
        case_batch = cases[batch_start : batch_start + parallel_hanchans]
        batch_size = len(case_batch)
        envs = BatchedRiichiEnv(
            batch_size,
            # evaluation_cases deliberately emits consecutive seeds, which is
            # also BatchedRiichiEnv's per-table seed schedule.
            seed=int(case_batch[0][0]),
            step_threads=batch_size,
            game_mode=str(config.get("heuristic_evaluation_game_mode", "4p-red-half")),
        )
        bridge = BatchedStateBridge(riichi.MjaiKyokuStateMachineManager(batch_size), batch_size)
        observations = list(envs.reset())
        bridge.sync(observations)
        public = PublicStateTracker(batch_size)
        public.update(bridge.last_events)
        analyzer = EfficiencyAnalyzer(
            int(config.get("heuristic_evaluation_cache_capacity", 131_072)),
        )
        heuristics = {
            EFFICIENCY: HeuristicPolicy(analyzer, public, defensive=False),
            DEFENSE: HeuristicPolicy(analyzer, public, defensive=True),
        }
        policies_by_env: list[list[str]] = []
        candidate_seats: list[int] = []
        for _seed, candidate_seat, opponents in case_batch:
            policies: list[str] = []
            opponent_iter = iter(opponents)
            for seat in range(NUM_PLAYERS):
                policies.append(
                    "candidate" if seat == int(candidate_seat) else next(opponent_iter)
                )
            policies_by_env.append(policies)
            candidate_seats.append(int(candidate_seat))

        start_scores = [[int(value) for value in row] for row in envs.scores()]
        initial_scores = [row.copy() for row in start_scores]
        current_dealers = [
            int(observations[index][candidate_seat].oya)
            for index, candidate_seat in enumerate(candidate_seats)
        ]
        match_kyokus = [0] * batch_size
        match_discards = [0] * batch_size
        active_envs = set(range(batch_size))
        for _step in range(int(config.get("heuristic_evaluation_max_steps", 4000))):
            actions_by_env: list[dict[int, Any]] = [{} for _ in range(batch_size)]
            decisions = active_decisions(observations, active_envs)
            analysis = (
                DecisionAnalysisBatch.build(
                    decisions, analyzer=analyzer, public=public,
                )
                if decisions else None
            )
            candidate_decisions = [
                decision for decision in decisions
                if policies_by_env[decision.env_index][decision.seat_id] == "candidate"
            ]
            for policy_name, policy in heuristics.items():
                policy_decisions = [
                    decision for decision in decisions
                    if policies_by_env[decision.env_index][decision.seat_id] == policy_name
                ]
                for decision, action in zip(
                    policy_decisions,
                    policy.select_batch(policy_decisions, analysis),
                    strict=True,
                ):
                    actions_by_env[decision.env_index][decision.seat_id] = action

            if candidate_decisions:
                prepared = adapter.prepare(bridge, candidate_decisions, analysis)
                legal = prepared.legal
                action_ids = adapter.masked_logits(prepared).argmax(-1).tolist()
                candidate_actions = bridge.decode(candidate_decisions, action_ids)
                for decision, action_id, action, legal_row in zip(
                    candidate_decisions, action_ids, candidate_actions, legal, strict=True,
                ):
                    env_index = decision.env_index
                    candidate_seat = candidate_seats[env_index]
                    tile = getattr(action, "tile", None)
                    tile_type = int(tile) // 4 if tile is not None else -1
                    threat = public.has_riichi_threat(env_index, candidate_seat)
                    decision_phase = _phase(int(public.discard_counts[env_index]))
                    metrics.record_decision(
                        int(action_id), legal_row, threat=threat,
                        genbutsu_to_all=(tile_type >= 0 and public.is_genbutsu_to_all_riichi(env_index, tile_type)),
                        genbutsu_count=(public.genbutsu_coverage(env_index, tile_type) if tile_type >= 0 else 0),
                        seat=candidate_seat, dealer=candidate_seat == current_dealers[env_index],
                        phase=decision_phase, prior_riichi_count=int(public.riichi[env_index].sum()),
                    )
                    row = analysis.for_decision(decision) if analysis is not None else None
                    candidate = row.candidate_for(action) if row is not None else None
                    if candidate is None and row is not None and int(action_id) == 75:
                        candidate = min(row.candidates, key=lambda item: item.rank, default=None)
                    if candidate is not None and row is not None and row.best_rank is not None:
                        discard_regret, call_regret = row.selected_regrets(action)
                        shanten_gap = int(candidate.structural_shanten) - int(row.best_rank[0])
                        best_ukeire = max((item.ukeire for item in row.candidates if item.structural_shanten == row.best_rank[0]), default=0)
                        ukeire_loss = (max(0, best_ukeire - int(candidate.ukeire)) / max(best_ukeire, 1) if shanten_gap == 0 else 1.0)
                        metrics.record_efficiency(reward=float(discard_regret + call_regret), shanten_gap=shanten_gap, ukeire_loss=ukeire_loss, selected_shanten=int(candidate.structural_shanten), selected_effective_shanten=int(candidate.effective_shanten))
                        metrics.record_rule_quality(candidate, accepted_call=76 <= int(action_id) <= 170, bad_call=call_regret < 0.0, best_rank=row.best_rank, alternatives=row.candidates)
                    metrics.record_rule_action(int(action_id), threat=threat, bad_kan=threat)
                    actions_by_env[env_index][decision.seat_id] = action

            observations = list(envs.step_batch(actions_by_env))
            end_kyoku, _end_game = bridge.sync(observations)
            public.update(bridge.last_events)
            scores_by_env = [[int(value) for value in row] for row in envs.scores()]
            for env_index in list(active_envs):
                candidate_seat = candidate_seats[env_index]
                scores = scores_by_env[env_index]
                if bool(end_kyoku[env_index]):
                    candidate_discards = int(public.completed_discard_counts_by_seat[env_index, candidate_seat])
                    metrics.record_kyoku([candidate_seat], [scores[seat] - start_scores[env_index][seat] for seat in range(NUM_PLAYERS)], bridge.last_events[env_index], discard_count=candidate_discards, open_meld_count=int(public.completed_open_meld_counts_by_seat[env_index, candidate_seat]), dealer_seat=current_dealers[env_index], phase=_phase(int(public.completed_discard_counts[env_index])))
                    start_scores[env_index] = scores
                    match_kyokus[env_index] += 1
                    match_discards[env_index] += candidate_discards
                    current_dealers[env_index] = int(observations[env_index][candidate_seat].oya)
                if bool(envs.done()[env_index]):
                    metrics.record_match_result(
                        candidate_seat,
                        scores,
                        kyoku_count=match_kyokus[env_index],
                        discard_count=match_discards[env_index],
                        point_delta=(
                            scores[candidate_seat]
                            - initial_scores[env_index][candidate_seat]
                        ),
                    )
                    active_envs.remove(env_index)
            if not active_envs:
                break
        else:
            raise RuntimeError(f"SFT heuristic evaluation batch={batch_start // parallel_hanchans} exceeded max steps")

        elapsed = time.perf_counter() - started
        print(
            f"heuristic_evaluation game={batch_start + batch_size}/{count} "
            f"parallel_hanchans={batch_size} elapsed_s={elapsed:.2f}",
            flush=True,
        )

    summary = metrics.summary("heuristic_eval")
    kyoku_points = np.asarray(metrics.kyoku_points, dtype=np.float64)
    summary["heuristic_eval/kyoku/point_delta_mean_stderr"] = (
        float(kyoku_points.std(ddof=1) / math.sqrt(len(kyoku_points)))
        if len(kyoku_points) > 1 else 0.0
    )
    summary["heuristic_eval/performance/elapsed_s"] = time.perf_counter() - started
    summary["heuristic_eval/performance/hanchan_per_s"] = (
        count / max(summary["heuristic_eval/performance/elapsed_s"], 1e-9)
    )
    if was_training:
        policy_model.train()
    return summary
