"""Deterministic fixed-baseline evaluation for the current PPO policy."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

try:
    import ray
except ImportError:  # pragma: no cover - imported by the training entry point
    ray = None

from ..model.bridge import BatchedStateBridge, Decision, NUM_PLAYERS
from .metrics import SemanticMetrics
from .opponents.heuristic import HeuristicPolicy
from .rewards import (
    DiscardAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
    selected_efficiency_rewards,
)
from .worker import active_decisions


EFFICIENCY = "heuristic_efficiency"
DEFENSE = "heuristic_defense"


def evaluation_cases(seed_base: int, hanchan_count: int, *, cycle: int = 0) -> list[tuple[int, int, tuple[str, str, str]]]:
    """Create an exact deterministic hanchan budget with cyclic seat rotation."""
    count = int(hanchan_count)
    if count <= 0:
        raise ValueError("hanchan_count must be positive")
    offset = (2 * int(cycle)) % NUM_PLAYERS
    result = []
    for index in range(count):
        recipe = (EFFICIENCY, DEFENSE, EFFICIENCY) if index % 2 == 0 else (DEFENSE, EFFICIENCY, DEFENSE)
        result.append((int(seed_base) + index, (index + offset) % NUM_PLAYERS, recipe))
    return result


def merge_evaluation_summaries(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    values = list(rows)
    if not values:
        return {}
    count_keys = {
        "eval/kyoku/count", "eval/match/count", "eval/action/decision_count",
        "eval/action/riichi_opportunity_count", "eval/action/call_opportunity_count",
        "eval/defense/threat_discard_count",
    }
    result: dict[str, float] = {}
    for name in {name for row in values for name in row}:
        samples = [float(row[name]) for row in values if name in row]
        if name in count_keys:
            result[name] = float(sum(samples))
            continue
        if name.startswith("eval/kyoku/"):
            weights = [float(row.get("eval/kyoku/count", 0.0)) for row in values if name in row]
        elif name.startswith("eval/match/"):
            weights = [float(row.get("eval/match/count", 0.0)) for row in values if name in row]
        elif name == "eval/action/riichi_opportunity_accept_rate":
            weights = [float(row.get("eval/action/riichi_opportunity_count", 0.0)) for row in values if name in row]
        elif name == "eval/action/call_opportunity_accept_rate":
            weights = [float(row.get("eval/action/call_opportunity_count", 0.0)) for row in values if name in row]
        elif name.startswith(("eval/action/", "eval/efficiency/")):
            weights = [float(row.get("eval/action/decision_count", 0.0)) for row in values if name in row]
        else:
            weights = [1.0] * len(samples)
        result[name] = float(np.average(samples, weights=weights)) if sum(weights) else float(np.mean(samples))
    kyoku_scores = [float(row.get("eval/kyoku/point_delta_mean", 0.0)) for row in values]
    result["eval/kyoku/point_delta_mean_stderr"] = (
        float(np.std(kyoku_scores, ddof=1) / np.sqrt(len(kyoku_scores))) if len(kyoku_scores) > 1 else 0.0
    )
    return result


if ray is not None:
    @ray.remote(num_cpus=1)
    class EvaluationWorker:
        """CPU-only worker; all candidate-policy inference remains on GPU actors."""
        def __init__(self, worker_id: int, config: dict[str, Any], inference: Any) -> None:
            try:
                import riichi
                from riichienv import BatchedRiichiEnv
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("install local riichi and RiichiEnv extensions before evaluating") from exc
            self.worker_id, self.config, self.inference = int(worker_id), dict(config), inference
            self._riichi, self._env_type = riichi, BatchedRiichiEnv

        def evaluate(self, seed: int, candidate_seat: int, opponents: tuple[str, str, str]) -> dict[str, Any]:
            envs = self._env_type(1, seed=int(seed), step_threads=1, game_mode=self.config["game_mode"])
            bridge = BatchedStateBridge(self._riichi.MjaiKyokuStateMachineManager(1), 1)
            observations = list(envs.reset()); bridge.sync(observations)
            public = PublicStateTracker(1); public.update(bridge.last_events)
            efficiency = EfficiencyAnalyzer(int(self.config.get("reward_cache_capacity", 131_072)))
            heuristics = {EFFICIENCY: HeuristicPolicy(efficiency, public, defensive=False), DEFENSE: HeuristicPolicy(efficiency, public, defensive=True)}
            policies = []
            opponent_iter = iter(opponents)
            for seat in range(NUM_PLAYERS):
                policies.append("candidate" if seat == int(candidate_seat) else next(opponent_iter))
            start_scores = [int(value) for value in envs.scores()[0]]
            metrics = SemanticMetrics()
            match_kyokus = 0
            match_discards = 0
            for _step in range(int(self.config.get("evaluation_max_steps", 4000))):
                actions_by_env: list[dict[int, Any]] = [{}]
                decisions = active_decisions(observations)
                analysis_indices = [
                    index for index, decision in enumerate(decisions)
                    if policies[decision.seat_id] in heuristics or policies[decision.seat_id] == "candidate"
                ]
                analysis = DiscardAnalysisBatch.build([decisions[index] for index in analysis_indices], analyzer=efficiency, public=public) if analysis_indices else None
                for index, decision in enumerate(decisions):
                    policy = policies[decision.seat_id]
                    if policy == "candidate":
                        factors, numeric, lengths, legal, _generations, critic, critic_lengths = bridge.prepare([decision])
                        response = ray.get(self.inference.infer.remote(
                            worker_id=100_000 + self.worker_id, namespace="eval", batch_indices=[decision.batch_index],
                            token_factors=factors, token_numeric=numeric, critic_factors=critic, critic_lengths=critic_lengths,
                            legal_mask=legal, token_lengths=lengths, greedy=True,
                        ))
                        action_id = int(response["action_ids"][0])
                        action = bridge.decode([decision], [action_id])[0]
                        tile = getattr(action, "tile", None); tile_type = int(tile) // 4 if tile is not None else -1
                        threat = public.has_riichi_threat(0, decision.seat_id)
                        metrics.record_decision(action_id, legal[0], threat=threat,
                                                genbutsu_to_all=(tile_type >= 0 and public.is_genbutsu_to_all_riichi(0, tile_type)),
                                                genbutsu_count=(public.genbutsu_coverage(0, tile_type) if tile_type >= 0 else 0))
                        efficiency_reward = selected_efficiency_rewards(
                            [decision], [action], analyzer=efficiency, public=public, analysis=analysis,
                        )[0]
                        analysis_row = analysis.for_decision(decision) if analysis is not None else None
                        if analysis_row is not None:
                            candidate = next(
                                (item for item in analysis_row.candidates if getattr(item.action, "tile", None) == tile),
                                None,
                            )
                            if candidate is not None and analysis_row.best_shanten is not None:
                                shanten_gap = int(candidate.shanten) - int(analysis_row.best_shanten)
                                ukeire_loss = (
                                    max(0, int(analysis_row.best_ukeire) - int(candidate.ukeire))
                                    / max(int(analysis_row.best_ukeire), 1)
                                    if shanten_gap == 0 else 1.0
                                )
                                metrics.record_efficiency(
                                    reward=float(efficiency_reward),
                                    shanten_gap=shanten_gap,
                                    ukeire_loss=ukeire_loss,
                                )
                    else:
                        action = heuristics[policy].select_batch([decision], analysis)[0]
                    actions_by_env[0][decision.seat_id] = action
                observations = list(envs.step_batch(actions_by_env))
                end_kyoku, _end_game = bridge.sync(observations); public.update(bridge.last_events)
                scores = [int(value) for value in envs.scores()[0]]
                if bool(end_kyoku[0]):
                    metrics.record_kyoku(
                        [candidate_seat], [scores[seat] - start_scores[seat] for seat in range(NUM_PLAYERS)],
                        bridge.last_events[0],
                        discard_count=int(public.completed_discard_counts[0]),
                        open_meld_count=int(public.completed_open_meld_counts[0]),
                    )
                    start_scores = scores
                    match_kyokus += 1
                    match_discards += int(public.completed_discard_counts[0])
                if bool(envs.done()[0]):
                    metrics.record_match_result(
                        candidate_seat, scores, kyoku_count=match_kyokus, discard_count=match_discards,
                    )
                    return {"metrics": metrics.summary("eval"), "seed": int(seed), "candidate_seat": int(candidate_seat), "opponents": list(opponents)}
            raise RuntimeError("evaluation game exceeded evaluation_max_steps")
else:
    EvaluationWorker = None
