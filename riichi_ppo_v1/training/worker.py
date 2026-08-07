"""Ray rollout actors.  Workers own environments and Rust state-machine slots."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import ray
except ImportError:  # imported lazily by the command line program
    ray = None

from ..model.bridge import NUM_PLAYERS, BatchedStateBridge, Decision
from .metrics import SemanticMetrics
from .profiling import StageProfiler
from .rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
    terminal_kyoku_reward,
)
from .trajectory import Transition, finish_kyoku


def active_decisions(
    observations_by_env: list[dict[int, Any]], active_envs: set[int] | None = None,
) -> list[Decision]:
    """Return every seat that currently has at least one legal action."""
    return [
        Decision(env_index, seat, observation)
        for env_index, observations in enumerate(observations_by_env)
        if active_envs is None or env_index in active_envs
        for seat, observation in observations.items()
        if observation.legal_actions()
    ]


if ray is not None:
    @ray.remote
    class RolloutWorker:
        def __init__(
            self,
            worker_id: int,
            config: dict[str, Any],
            inference: Any,
        ) -> None:
            try:
                import riichi
                from riichienv import BatchedRiichiEnv
            except ImportError as exc:
                raise RuntimeError("install local riichi and RiichiEnv extensions before starting workers") from exc
            self.config = config
            self.worker_id = int(worker_id)
            self.inference = inference
            # Environment workers intentionally never import CUDA or own a
            # policy model. The single inference actor owns GPU execution.
            self.num_envs = int(config["envs_per_worker"])
            self.envs = BatchedRiichiEnv(
                self.num_envs,
                seed=int(config["seed"]) + worker_id * 1_000,
                step_threads=int(config.get("env_step_threads", 4)),
                game_mode=config["game_mode"],
            )
            self.state_machine = riichi.MjaiKyokuStateMachineManager(self.num_envs)
            self.profiler = StageProfiler(enabled=bool(config.get("profile_enabled", True)))
            self.bridge = BatchedStateBridge(
                self.state_machine,
                self.num_envs,
                self.profiler,
                critic_include_public_state=bool(config.get("critic_include_public_state", False)),
            )
            self.observations = list(self.envs.reset())
            self.bridge.sync(self.observations)
            self.public = PublicStateTracker(self.num_envs)
            self.public.update(self.bridge.last_events)
            self.efficiency = EfficiencyAnalyzer(int(config.get("reward_cache_capacity", 131_072)))
            self.start_scores = [[int(x) for x in scores] for scores in self.envs.scores()]
            self.match_kyoku_counts = [0] * self.num_envs
            self.pending: list[list[list[Transition]]] = [
                [[] for _ in range(NUM_PLAYERS)] for _ in range(self.num_envs)
            ]
            self.model_decisions = 0
            self.recorded_decisions = 0
            self.kyoku_reward_clip_points = int(
                config.get("kyoku_reward_clip_points", 24_000)
            )
            if self.kyoku_reward_clip_points <= 0:
                raise ValueError("kyoku_reward_clip_points must be positive")
            self.deferred_reset_indices: set[int] = set()
            self.semantic = SemanticMetrics()
            self.lineups: list[tuple[str, str, str, str]] = [("current",) * NUM_PLAYERS for _ in range(self.num_envs)]

        def set_rollout_context(
            self,
            update: int,
        ) -> None:
            """Install the standard all-current self-play lineup."""
            del update
            self.lineups = [("current",) * NUM_PLAYERS for _ in range(self.num_envs)]

        def _submit_model_actions(
            self,
            decisions: list[Decision],
            namespace: str,
            greedy: bool,
            analysis_batch: DecisionAnalysisBatch | None = None,
        ) -> tuple[Any, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            with self.profiler.stage("rollout/model_state_prepare"):
                (
                    token_factors,
                    token_numeric,
                    token_lengths,
                    legal,
                    _history_generations,
                    critic_factors,
                    critic_lengths,
                ) = self.bridge.prepare(decisions, analysis_batch)
            request = self.inference.infer.remote(
                worker_id=self.worker_id,
                namespace=namespace,
                batch_indices=[decision.batch_index for decision in decisions],
                token_factors=token_factors,
                token_numeric=token_numeric,
                critic_factors=critic_factors,
                critic_lengths=critic_lengths,
                legal_mask=legal,
                token_lengths=token_lengths,
                greedy=greedy,
            )
            return request, (token_factors, token_numeric, token_lengths, legal, critic_factors, critic_lengths)

        def _model_actions(
            self,
            decisions: list[Decision],
            prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            result: dict[str, Any],
            record: bool,
            analysis_batch: DecisionAnalysisBatch | None = None,
        ) -> tuple[list[Any], list[Transition | None]]:
            token_factors, token_numeric, token_lengths, legal, critic_factors, critic_lengths = prepared
            action_ids = [int(value) for value in result["action_ids"]]
            logprobs = [float(value) for value in result["logprobs"]]
            values = [float(value) for value in result["values"]]
            with self.profiler.stage("rollout/model_action_decode"):
                actions = self.bridge.decode(decisions, action_ids)
            self.model_decisions += len(decisions)
            transitions: list[Transition | None] = [None] * len(decisions)
            if record:
                with self.profiler.stage("rollout/transition_materialize"):
                    for row, action_id in enumerate(action_ids):
                        decision = decisions[row]
                        analysis_row = analysis_batch.for_decision(decision) if analysis_batch is not None else None
                        transition = Transition(
                            token_factors[row, : token_lengths[row]].copy(),
                            token_numeric[row, : token_lengths[row]].copy(), int(token_lengths[row]),
                            legal[row].copy(), action_id, logprobs[row], values[row],
                            critic_factors=critic_factors[row, : critic_lengths[row]].copy(),
                            critic_length=int(critic_lengths[row]),
                        )
                        transitions[row] = transition
                        threat = self.public.has_riichi_threat(decision.env_index, decision.seat_id)
                        tile = getattr(actions[row], "tile", None)
                        tile_type = int(tile) // 4 if tile is not None else -1
                        self.semantic.record_decision(
                            action_id, legal[row], threat=threat,
                            genbutsu_to_all=(tile_type >= 0 and self.public.is_genbutsu_to_all_riichi(decision.env_index, tile_type)),
                            genbutsu_count=(self.public.genbutsu_coverage(decision.env_index, tile_type) if tile_type >= 0 else 0),
                        )
                        if analysis_row is not None:
                            chosen = actions[row]
                            candidate = analysis_row.candidate_for(chosen)
                            if candidate is not None and analysis_row.best_rank is not None:
                                shanten_gap = int(candidate.structural_shanten) - int(analysis_row.best_rank[0])
                                best_ukeire = max(
                                    (item.ukeire for item in analysis_row.candidates
                                     if item.structural_shanten == analysis_row.best_rank[0]),
                                    default=0,
                                )
                                ukeire_loss = (
                                    max(0, best_ukeire - int(candidate.ukeire)) / max(best_ukeire, 1)
                                    if shanten_gap == 0 else 1.0
                                )
                                self.semantic.record_efficiency(
                                reward=0.0,
                                    shanten_gap=shanten_gap,
                                    ukeire_loss=ukeire_loss,
                                )
                                self.semantic.record_rule_quality(
                                    candidate,
                                    accepted_call=76 <= action_id <= 170,
                                    bad_call=False,
                                    best_rank=analysis_row.best_rank,
                                    alternatives=analysis_row.candidates,
                                )
                        self.recorded_decisions += 1
            return actions, transitions

        def _finish_games(self, done_indices: list[int]) -> None:
            """Discard all four completed-seat trajectories before reset."""
            if not done_indices:
                return
            for env_index in done_indices:
                self.pending[env_index] = [[] for _ in range(NUM_PLAYERS)]

        def _reset_games(self, done_indices: list[int]) -> None:
            if not done_indices:
                return
            with self.profiler.stage("env/reset_completed_native"):
                self.observations = list(self.envs.reset_indices(done_indices))
            with self.profiler.stage("rollout/event_sync_after_reset"):
                self.bridge.sync(self.observations)
            with self.profiler.stage("rollout/public_state_update"):
                self.public.update(self.bridge.last_events)
            scores_by_env = self.envs.scores()
            for env_index in done_indices:
                self.start_scores[env_index] = [int(x) for x in scores_by_env[env_index]]
                self.match_kyoku_counts[env_index] = 0

        def _advance_once(
            self,
            active_envs: set[int] | None = None,
        ) -> tuple[list[Transition], list[float], int, list[int], list[int]]:
            completed: list[Transition] = []
            rewards: list[float] = []
            active = set(range(self.num_envs)) if active_envs is None else active_envs
            actions_by_env: list[dict[int, Any]] = [{} for _ in range(self.num_envs)]
            with self.profiler.stage("rollout/scan_observations_and_legal_actions"):
                decisions = active_decisions(self.observations, active)
            if decisions:
                for policy_decisions in (decisions,):
                    with self.profiler.stage("rollout/reward_analysis"):
                        analysis_batch = DecisionAnalysisBatch.build(
                            policy_decisions, analyzer=self.efficiency, public=self.public,
                            profiler=self.profiler,
                        )
                    request, prepared = self._submit_model_actions(
                        policy_decisions, "rollout", False,
                        analysis_batch,
                    )
                    with self.profiler.stage("inference/rpc_wait"):
                        result = ray.get(request)
                    record_policy = True
                    actions, transitions = self._model_actions(
                        policy_decisions, prepared, result, record_policy, analysis_batch,
                    )
                    for decision, action, transition in zip(
                        policy_decisions, actions, transitions, strict=True,
                    ):
                        actions_by_env[decision.env_index][decision.seat_id] = action
                        if transition is not None:
                            self.pending[decision.env_index][decision.seat_id].append(transition)

            with self.profiler.stage("env/step_batch_native"):
                self.observations = list(self.envs.step_batch(actions_by_env))
            with self.profiler.stage("rollout/event_sync_after_step"):
                end_kyoku, _end_game = self.bridge.sync(self.observations)
            with self.profiler.stage("rollout/public_state_update"):
                self.public.update(self.bridge.last_events)
            scores_by_env = self.envs.scores()
            done = self.envs.done()
            done_indices = [index for index, value in enumerate(done) if value and index in active]
            completed_kyokus = 0
            ended_kyoku_indices: list[int] = []
            for env_index in range(self.num_envs):
                if env_index in active and end_kyoku[env_index]:
                    completed_kyokus += 1
                    ended_kyoku_indices.append(env_index)
                    self.match_kyoku_counts[env_index] += 1
                    scores = [int(x) for x in scores_by_env[env_index]]
                    current_seats = [
                        seat for seat, policy in enumerate(self.lineups[env_index])
                        if policy == "current"
                    ]
                    self.semantic.record_kyoku(
                        current_seats,
                        [scores[seat] - self.start_scores[env_index][seat] for seat in range(NUM_PLAYERS)],
                        self.bridge.last_events[env_index],
                        discard_count=int(self.public.completed_discard_counts[env_index]),
                        open_meld_count=int(self.public.completed_open_meld_counts[env_index]),
                    )
                    for seat in range(NUM_PLAYERS):
                        reward = terminal_kyoku_reward(
                            scores[seat] - self.start_scores[env_index][seat],
                            self.kyoku_reward_clip_points,
                        )
                        pending = self.pending[env_index][seat]
                        if self.lineups[env_index][seat] == "current":
                            with self.profiler.stage("rollout/finish_kyoku_gae"):
                                if pending:
                                    pending[-1].kyoku_reward = reward
                                    pending[-1].reward = reward
                                completed.extend(finish_kyoku(
                                    pending, float(self.config["gamma"]), float(self.config["gae_lambda"]),
                                ))
                        rewards.append(reward)
                        self.pending[env_index][seat] = []
                    self.start_scores[env_index] = scores
                    if done[env_index]:
                        # Self-play is symmetric: record the physical hanchan
                        # once, rather than duplicating its length for four seats.
                        self.semantic.record_match_length(self.match_kyoku_counts[env_index])
            if done_indices:
                self._finish_games(done_indices)
            return completed, rewards, completed_kyokus, ended_kyoku_indices, done_indices

        def collect(
            self,
            update: int | None = None,
        ) -> tuple[list[Transition], dict[str, float]]:
            if update is not None:
                self.set_rollout_context(int(update))
            # ``kyokus_per_worker`` is the worker-level rollout drain target.
            # The native environment advances tables in parallel, so a worker
            # can exceed it by one in-flight completion wave; that bounded
            # overshoot is preferred to discarding partial kyoku trajectories.
            target = int(self.config["kyokus_per_worker"])
            transitions: list[Transition] = []
            rewards: list[float] = []
            kyokus = 0
            drain_kyokus = 0
            drain_steps = 0
            draining = False
            active_envs = set(range(self.num_envs))
            self.profiler.reset()
            self.semantic = SemanticMetrics()
            for lineup in self.lineups:
                current_seats = [seat for seat, policy in enumerate(lineup) if policy == "current"]
                self.semantic.record_lineup(lineup, current_seats)
            self.model_decisions = 0
            self.recorded_decisions = 0
            if self.deferred_reset_indices:
                self._reset_games(sorted(self.deferred_reset_indices))
                self.deferred_reset_indices.clear()
            started = time.perf_counter()
            while active_envs:
                step, new_rewards, new_kyokus, ended_kyokus, done_indices = self._advance_once(
                    active_envs=active_envs,
                )
                transitions.extend(step)
                rewards.extend(new_rewards)
                kyokus += new_kyokus
                if not draining and kyokus >= target:
                    draining = True
                if draining:
                    # A table is frozen immediately after its in-flight kyoku
                    # closes, so it cannot start collecting an extra kyoku.
                    frozen = set(ended_kyokus) | set(done_indices)
                    drain_kyokus += new_kyokus
                    drain_steps += 1
                    active_envs.difference_update(frozen)
                    self.deferred_reset_indices.update(done_indices)
                else:
                    self._reset_games(done_indices)
            elapsed = time.perf_counter() - started
            stats = {
                "kyokus": float(kyokus),
                "sampled_rewards": float(len(rewards)),
                "reward_mean": float(np.mean(rewards)),
                "rollout_s": elapsed,
                "transitions_per_s": float(len(transitions) / max(elapsed, 1e-9)),
                "model_decisions": float(self.model_decisions),
                "recorded_decisions": float(self.recorded_decisions),
                "sampled_seats_per_game": float(
                    sum(policy == "current" for lineup in self.lineups for policy in lineup)
                    / max(self.num_envs, 1)
                ),
                "reward_schedule/kyoku_weight": 1.0,
                "drain_kyokus": float(drain_kyokus),
                "drain_steps": float(drain_steps),
            }
            stats.update(self.profiler.delta({}, prefix="timing"))
            stats.update(self.efficiency.metrics())
            stats.update(self.public.metrics())
            for transition in transitions:
                self.semantic.record_transition_reward(transition)
            stats.update(self.semantic.summary())
            return transitions, stats

else:
    RolloutWorker = None
