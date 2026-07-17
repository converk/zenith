"""Ray rollout actors.  Workers own environments and Rust state-machine slots."""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np

try:
    import ray
except ImportError:  # imported lazily by the command line program
    ray = None

from ..model.bridge import BatchedStateBridge, Decision, NUM_PLAYERS
from .curriculum import STAGES, Curriculum, CurriculumSnapshot
from .opponents.heuristic import HeuristicPolicy
from .opponents.history import compatible_history, rollout_cohort
from .opponents.lineup import CURRENT, DEFENSE, EFFICIENCY, Lineup, LineupSampler
from .profiling import StageProfiler
from .rewards import DiscardAnalysisBatch, EfficiencyAnalyzer, PublicStateTracker, selected_efficiency_rewards
from .trajectory import Transition, finish_kyoku
from .metrics import SemanticMetrics


def sample_training_seats(rng: random.Random) -> tuple[int, int]:
    """Choose two distinct seats for one complete hanchan."""
    return tuple(sorted(rng.sample(range(NUM_PLAYERS), 2)))


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


def should_record_decision(decision: Decision, sampled_seats: list[tuple[int, int]] | list[Lineup]) -> bool:
    owner = sampled_seats[decision.env_index]
    return owner.is_learner(decision.seat_id) if isinstance(owner, Lineup) else decision.seat_id in owner


def rank_reward(rank: int) -> float:
    value = int(rank)
    index = value - 1 if 1 <= value <= 4 else value
    if not 0 <= index < 4:
        raise ValueError(f"invalid four-player rank {rank!r}")
    return (12.0, 4.0, -4.0, -12.0)[index]


if ray is not None:
    @ray.remote
    class RolloutWorker:
        def __init__(self, worker_id: int, config: dict[str, Any], inference: Any) -> None:
            try:
                import riichi
                from riichienv import BatchedRiichiEnv
            except ImportError as exc:
                raise RuntimeError("install local riichi and RiichiEnv extensions before starting workers") from exc
            self.config = config
            self.worker_id = int(worker_id)
            self.inference = inference
            self.rng = random.Random(int(config["seed"]) + worker_id * 10_000)
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
            self.curriculum = Curriculum(int(config["total_updates"]))
            self.curriculum_snapshot: CurriculumSnapshot = self.curriculum.snapshot(0)
            self.history: tuple[str, ...] = ()
            self.lineup_sampler = LineupSampler(self.rng)
            self.generations = [0] * self.num_envs
            self.lineups: list[Lineup] = [
                self.lineup_sampler.sample(self.curriculum_snapshot, self.history, index, 0)
                for index in range(self.num_envs)
            ]
            # Kept as a compatibility alias for existing diagnostic callers.
            self.sampled_seats = [tuple(seat for seat in range(NUM_PLAYERS) if lineup.is_learner(seat)) for lineup in self.lineups]
            self.observations = list(self.envs.reset())
            self.bridge.sync(self.observations)
            self.public = PublicStateTracker(self.num_envs)
            self.public.update(self.bridge.last_events)
            self.efficiency = EfficiencyAnalyzer(int(config.get("reward_cache_capacity", 131_072)))
            self.heuristic_efficiency = HeuristicPolicy(self.efficiency, self.public, defensive=False)
            self.heuristic_defense = HeuristicPolicy(self.efficiency, self.public, defensive=True)
            self.start_scores = [[int(x) for x in scores] for scores in self.envs.scores()]
            self.pending: list[list[list[Transition]]] = [
                [[] for _ in range(NUM_PLAYERS)] for _ in range(self.num_envs)
            ]
            self.model_decisions = 0
            self.recorded_decisions = 0
            self.deferred_reset_indices: set[int] = set()
            self.semantic = SemanticMetrics()

        def set_rollout_context(self, update: int) -> None:
            """Freeze reward weights and available historical paths for one rollout."""
            next_snapshot = self.curriculum.snapshot(int(update))
            changed = next_snapshot.stage.name != self.curriculum_snapshot.stage.name
            self.curriculum_snapshot = next_snapshot
            self.history = rollout_cohort(
                compatible_history(
                    self.config["checkpoint_dir"],
                    max_entries=int(self.config.get("history_pool_size", 48)),
                ),
                seed=int(self.config["seed"]), update=int(update), size=2,
            )
            # A stage boundary is also a hanchan boundary.  This gives every
            # lineup a fixed opponent set for its whole match and makes the
            # fixed percentage schedule take effect immediately, rather than
            # waiting for a long-running table to finish naturally.
            if changed:
                self.generations = [generation + 1 for generation in self.generations]
                self.lineups = [
                    self.lineup_sampler.sample(self.curriculum_snapshot, self.history, index, self.generations[index])
                    for index in range(self.num_envs)
                ]
                self.sampled_seats = [tuple(seat for seat in range(NUM_PLAYERS) if lineup.is_learner(seat)) for lineup in self.lineups]
                self.pending = [[[] for _ in range(NUM_PLAYERS)] for _ in range(self.num_envs)]
                self.deferred_reset_indices.clear()
                self.observations = list(self.envs.reset())
                self.bridge.sync(self.observations)
                self.public.reset(list(range(self.num_envs)))
                self.public.update(self.bridge.last_events)
                self.start_scores = [[int(x) for x in scores] for scores in self.envs.scores()]

        def _model_actions(
            self,
            decisions: list[Decision],
            namespace: str,
            greedy: bool,
            record: bool,
            policy_id: str = CURRENT,
            analysis_batch: DiscardAnalysisBatch | None = None,
        ) -> tuple[list[Any], list[Transition | None]]:
            with self.profiler.stage("rollout/model_state_prepare"):
                (
                    token_factors,
                    token_numeric,
                    token_lengths,
                    legal,
                    _history_generations,
                    critic_factors,
                    critic_lengths,
                ) = self.bridge.prepare(decisions)
            with self.profiler.stage("inference/rpc_wait"):
                result = ray.get(self.inference.infer.remote(
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
                    policy_id=policy_id,
                ))
            action_ids = [int(value) for value in result["action_ids"]]
            logprobs = [float(value) for value in result["logprobs"]]
            values = [float(value) for value in result["values"]]
            with self.profiler.stage("rollout/model_action_decode"):
                actions = self.bridge.decode(decisions, action_ids)
            self.model_decisions += len(decisions)
            transitions: list[Transition | None] = [None] * len(decisions)
            if record:
                if float(self.curriculum_snapshot.weights[0]) != 0.0:
                    efficiency_rewards = selected_efficiency_rewards(
                        decisions, actions, analyzer=self.efficiency, public=self.public, analysis=analysis_batch,
                    )
                else:
                    efficiency_rewards = [0.0] * len(decisions)
                with self.profiler.stage("rollout/transition_materialize"):
                    for row, (action_id, efficiency_reward) in enumerate(zip(action_ids, efficiency_rewards, strict=True)):
                        decision = decisions[row]
                        if not self.lineups[decision.env_index].is_learner(decision.seat_id):
                            continue
                        transition = Transition(
                            token_factors[row, : token_lengths[row]].copy(),
                            token_numeric[row, : token_lengths[row]].copy(), int(token_lengths[row]),
                            legal[row].copy(), action_id, logprobs[row], values[row],
                            critic_factors=critic_factors[row, : critic_lengths[row]].copy(),
                            critic_length=int(critic_lengths[row]),
                        )
                        transition.efficiency_reward = float(efficiency_reward)
                        transition.reward_weights = self.curriculum_snapshot.weights
                        transition.refresh_reward()
                        transitions[row] = transition
                        threat = self.public.has_riichi_threat(decision.env_index, decision.seat_id)
                        tile = getattr(actions[row], "tile", None)
                        tile_type = int(tile) // 4 if tile is not None else -1
                        self.semantic.record_decision(
                            action_id, legal[row], threat=threat,
                            genbutsu_to_all=(tile_type >= 0 and self.public.is_genbutsu_to_all_riichi(decision.env_index, tile_type)),
                            genbutsu_count=(self.public.genbutsu_coverage(decision.env_index, tile_type) if tile_type >= 0 else 0),
                        )
                        if analysis_batch is not None:
                            analysis_row = analysis_batch.for_decision(decision)
                            chosen = actions[row]
                            chosen_tile = getattr(chosen, "tile", None)
                            candidate = next((item for item in analysis_row.candidates
                                              if getattr(item.action, "tile", None) == chosen_tile), None)
                            if candidate is not None and analysis_row.best_shanten is not None:
                                shanten_gap = int(candidate.shanten) - int(analysis_row.best_shanten)
                                ukeire_loss = (max(0, int(analysis_row.best_ukeire) - int(candidate.ukeire))
                                               / max(int(analysis_row.best_ukeire), 1)) if shanten_gap == 0 else 1.0
                                self.semantic.record_efficiency(reward=float(efficiency_reward), shanten_gap=shanten_gap, ukeire_loss=ukeire_loss)
                        self.recorded_decisions += 1
            return actions, transitions

        def _finish_games(self, done_indices: list[int]) -> list[int]:
            """Record hanchan-level results but leave native resets to the caller."""
            if not done_indices:
                return []
            ranks_by_env = self.envs.ranks()
            ranks: list[int] = []
            for env_index in done_indices:
                for seat in range(NUM_PLAYERS):
                    if not self.lineups[env_index].is_learner(seat):
                        continue
                    ranks.append(int(ranks_by_env[env_index][seat]))
                self.generations[env_index] += 1
                self.lineups[env_index] = self.lineup_sampler.sample(
                    self.curriculum_snapshot, self.history, env_index, self.generations[env_index],
                )
                self.sampled_seats[env_index] = tuple(seat for seat in range(NUM_PLAYERS) if self.lineups[env_index].is_learner(seat))
                self.pending[env_index] = [[] for _ in range(NUM_PLAYERS)]
            return ranks

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

        def _advance_once(
            self,
            greedy: bool = False,
            record: bool = True,
            active_envs: set[int] | None = None,
            reset_completed: bool = True,
        ) -> tuple[list[Transition], list[float], list[int], int, list[int], list[int]]:
            completed: list[Transition] = []
            rewards: list[float] = []
            ranks: list[int] = []
            active = set(range(self.num_envs)) if active_envs is None else active_envs
            actions_by_env: list[dict[int, Any]] = [{} for _ in range(self.num_envs)]
            with self.profiler.stage("rollout/scan_observations_and_legal_actions"):
                decisions = active_decisions(self.observations, active)
            if decisions:
                selected_actions: list[Any | None] = [None] * len(decisions)
                selected_transitions: list[Transition | None] = [None] * len(decisions)
                by_policy: dict[str, list[int]] = {}
                for index, decision in enumerate(decisions):
                    by_policy.setdefault(self.lineups[decision.env_index].policies[decision.seat_id], []).append(index)
                analysis_batch: DiscardAnalysisBatch | None = None
                needs_analysis = [
                    index for index, decision in enumerate(decisions)
                    if (
                        self.lineups[decision.env_index].policies[decision.seat_id] in {EFFICIENCY, DEFENSE}
                        or (
                            record
                            and self.lineups[decision.env_index].policies[decision.seat_id] == CURRENT
                            and float(self.curriculum_snapshot.weights[0]) != 0.0
                        )
                    )
                ]
                if needs_analysis:
                    with self.profiler.stage("rollout/reward_analysis"):
                        analysis_batch = DiscardAnalysisBatch.build(
                            [decisions[index] for index in needs_analysis],
                            analyzer=self.efficiency,
                            public=self.public,
                        )
                for policy_id, indices in by_policy.items():
                    subset = [decisions[index] for index in indices]
                    if policy_id == EFFICIENCY:
                        with self.profiler.stage("rollout/heuristic_select"):
                            actions = self.heuristic_efficiency.select_batch(subset, analysis_batch)
                        transitions = [None] * len(subset)
                    elif policy_id == DEFENSE:
                        with self.profiler.stage("rollout/heuristic_select"):
                            actions = self.heuristic_defense.select_batch(subset, analysis_batch)
                        transitions = [None] * len(subset)
                    else:
                        actions, transitions = self._model_actions(
                            subset, "eval" if greedy else "rollout", greedy,
                            record and policy_id == CURRENT, policy_id, analysis_batch,
                        )
                    for index, action, transition in zip(indices, actions, transitions, strict=True):
                        selected_actions[index] = action
                        selected_transitions[index] = transition
                for decision, action, transition in zip(decisions, selected_actions, selected_transitions, strict=True):
                    if action is None:
                        raise RuntimeError("mixed-policy rollout left a decision unresolved")
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
            ranks_by_env = self.envs.ranks() if done_indices else ()
            completed_kyokus = 0
            ended_kyoku_indices: list[int] = []
            for env_index in range(self.num_envs):
                if env_index in active and end_kyoku[env_index]:
                    completed_kyokus += 1
                    ended_kyoku_indices.append(env_index)
                    scores = [int(x) for x in scores_by_env[env_index]]
                    self.semantic.record_kyoku(
                        (seat for seat in range(NUM_PLAYERS) if self.lineups[env_index].is_learner(seat)),
                        [scores[seat] - self.start_scores[env_index][seat] for seat in range(NUM_PLAYERS)],
                        self.bridge.last_events[env_index],
                    )
                    for seat in range(NUM_PLAYERS):
                        if not self.lineups[env_index].is_learner(seat):
                            continue
                        reward = float(np.clip((scores[seat] - self.start_scores[env_index][seat]), -12_000, 12_000) / 1_000.0)
                        pending = self.pending[env_index][seat]
                        if record:
                            with self.profiler.stage("rollout/finish_kyoku_gae"):
                                if pending:
                                    pending[-1].kyoku_reward = reward
                                    if env_index in done_indices:
                                        pending[-1].rank_reward = rank_reward(int(ranks_by_env[env_index][seat]))
                                    pending[-1].refresh_reward()
                                completed.extend(finish_kyoku(pending, 0.0, float(self.config["gamma"]), float(self.config["gae_lambda"])))
                        rewards.append(float(sum(weight * component for weight, component in zip(
                            self.curriculum_snapshot.weights,
                            (0.0, reward, rank_reward(int(ranks_by_env[env_index][seat])) if env_index in done_indices else 0.0),
                            strict=True,
                        ))))
                        self.pending[env_index][seat] = []
                    self.start_scores[env_index] = scores
            if done_indices:
                for env_index in done_indices:
                    learner = [seat for seat in range(NUM_PLAYERS) if self.lineups[env_index].is_learner(seat)]
                    self.semantic.record_match(
                        learner,
                        [int(ranks_by_env[env_index][seat]) for seat in learner],
                        [int(scores_by_env[env_index][seat]) for seat in learner],
                    )
                ranks.extend(self._finish_games(done_indices))
                if reset_completed:
                    self._reset_games(done_indices)
            return completed, rewards, ranks, completed_kyokus, ended_kyoku_indices, done_indices

        def collect(self, update: int | None = None) -> tuple[list[Transition], dict[str, float]]:
            if update is not None:
                self.set_rollout_context(int(update))
            # ``kyokus_per_worker`` is the worker-level rollout drain target.
            # The native environment advances tables in parallel, so a worker
            # can exceed it by one in-flight completion wave; that bounded
            # overshoot is preferred to discarding partial kyoku trajectories.
            target = int(self.config["kyokus_per_worker"])
            transitions: list[Transition] = []
            rewards: list[float] = []
            ranks: list[int] = []
            kyokus = 0
            drain_kyokus = 0
            drain_steps = 0
            draining = False
            active_envs = set(range(self.num_envs))
            self.profiler.reset()
            self.semantic = SemanticMetrics()
            for index, lineup in enumerate(self.lineups):
                self.semantic.record_lineup(lineup.policies, (seat for seat in range(NUM_PLAYERS) if lineup.is_learner(seat)))
            self.model_decisions = 0
            self.recorded_decisions = 0
            if self.deferred_reset_indices:
                self._reset_games(sorted(self.deferred_reset_indices))
                self.deferred_reset_indices.clear()
            started = time.perf_counter()
            while active_envs:
                step, new_rewards, new_ranks, new_kyokus, ended_kyokus, done_indices = self._advance_once(
                    greedy=False, record=True, active_envs=active_envs, reset_completed=False,
                )
                transitions.extend(step)
                rewards.extend(new_rewards)
                ranks.extend(new_ranks)
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
                "rank_mean": float(np.mean(ranks)) if ranks else float("nan"),
                "rollout_s": elapsed,
                "transitions_per_s": float(len(transitions) / max(elapsed, 1e-9)),
                "model_decisions": float(self.model_decisions),
                "recorded_decisions": float(self.recorded_decisions),
                "sampled_seats_per_game": float(sum(bin(lineup.learner_mask).count("1") for lineup in self.lineups) / max(self.num_envs, 1)),
                "curriculum/progress": self.curriculum_snapshot.progress,
                "curriculum/stage": float(next(index for index, stage in enumerate(STAGES) if stage.name == self.curriculum_snapshot.stage.name)),
                "drain_kyokus": float(drain_kyokus),
                "drain_steps": float(drain_steps),
            }
            stats.update(self.profiler.delta({}, prefix="timing"))
            stats.update(self.efficiency.metrics())
            stats.update(self.public.metrics())
            for transition in transitions:
                self.semantic.record_transition_reward(transition)
            stats.update(self.semantic.summary())
            stats["train/opponents/history_pool_size"] = float(len(self.history))
            stats["train/opponents/history_cohort_size"] = float(len(self.history))
            return transitions, stats

else:
    RolloutWorker = None
