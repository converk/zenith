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

from .bridge import BatchedStateBridge, Decision, NUM_PLAYERS
from .profiling import StageProfiler
from .trajectory import Transition, finish_kyoku


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


def should_record_decision(decision: Decision, sampled_seats: list[tuple[int, int]]) -> bool:
    return decision.seat_id in sampled_seats[decision.env_index]


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
            self.bridge = BatchedStateBridge(self.state_machine, self.num_envs, self.profiler)
            self.sampled_seats = [sample_training_seats(self.rng) for _ in range(self.num_envs)]
            self.observations = list(self.envs.reset())
            self.bridge.sync(self.observations)
            self.start_scores = [[int(x) for x in scores] for scores in self.envs.scores()]
            self.pending: list[list[list[Transition]]] = [
                [[] for _ in range(NUM_PLAYERS)] for _ in range(self.num_envs)
            ]
            self.model_decisions = 0
            self.recorded_decisions = 0
            self.deferred_reset_indices: set[int] = set()

        def _model_actions(
            self, decisions: list[Decision], namespace: str, greedy: bool, record: bool,
        ) -> tuple[list[Any], list[Transition | None]]:
            with self.profiler.stage("rollout/model_state_prepare"):
                ids, attention, lengths, legal, history_lengths, history_generations = self.bridge.prepare(decisions)
            with self.profiler.stage("inference/rpc_wait"):
                result = ray.get(self.inference.infer.remote(
                    worker_id=self.worker_id,
                    namespace=namespace,
                    batch_indices=[decision.batch_index for decision in decisions],
                    ids=ids,
                    lengths=lengths,
                    legal_mask=legal,
                    history_lengths=history_lengths,
                    history_generations=history_generations,
                    greedy=greedy,
                ))
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
                        if not should_record_decision(decision, self.sampled_seats):
                            continue
                        length = int(lengths[row])
                        transitions[row] = Transition(
                            ids[row, :length].copy(), attention[row, :length].copy(), length, legal[row].copy(),
                            action_id, logprobs[row], values[row], history_length=int(history_lengths[row]),
                        )
                        self.recorded_decisions += 1
            return actions, transitions

        def _finish_games(self, done_indices: list[int]) -> list[int]:
            """Record hanchan-level results but leave native resets to the caller."""
            if not done_indices:
                return []
            ranks_by_env = self.envs.ranks()
            ranks: list[int] = []
            for env_index in done_indices:
                for seat in self.sampled_seats[env_index]:
                    ranks.append(int(ranks_by_env[env_index][seat]))
                self.sampled_seats[env_index] = sample_training_seats(self.rng)
                self.pending[env_index] = [[] for _ in range(NUM_PLAYERS)]
            return ranks

        def _reset_games(self, done_indices: list[int]) -> None:
            if not done_indices:
                return
            with self.profiler.stage("env/reset_completed_native"):
                self.observations = list(self.envs.reset_indices(done_indices))
            with self.profiler.stage("rollout/event_sync_after_reset"):
                self.bridge.sync(self.observations)
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
                model_actions, transitions = self._model_actions(decisions, "eval" if greedy else "rollout", greedy, record)
                for decision, action, transition in zip(decisions, model_actions, transitions):
                    actions_by_env[decision.env_index][decision.seat_id] = action
                    if transition is not None:
                        self.pending[decision.env_index][decision.seat_id].append(transition)

            with self.profiler.stage("env/step_batch_native"):
                self.observations = list(self.envs.step_batch(actions_by_env))
            with self.profiler.stage("rollout/event_sync_after_step"):
                end_kyoku, _end_game = self.bridge.sync(self.observations)
            scores_by_env = self.envs.scores()
            completed_kyokus = 0
            ended_kyoku_indices: list[int] = []
            for env_index in range(self.num_envs):
                if env_index in active and end_kyoku[env_index]:
                    completed_kyokus += 1
                    ended_kyoku_indices.append(env_index)
                    scores = [int(x) for x in scores_by_env[env_index]]
                    for seat in self.sampled_seats[env_index]:
                        reward = float(np.clip((scores[seat] - self.start_scores[env_index][seat]) / float(self.config["reward_scale"]), -float(self.config["reward_limit"]), float(self.config["reward_limit"])))
                        if record:
                            with self.profiler.stage("rollout/finish_kyoku_gae"):
                                completed.extend(finish_kyoku(self.pending[env_index][seat], reward, float(self.config["gamma"]), float(self.config["gae_lambda"])))
                        rewards.append(reward)
                        self.pending[env_index][seat] = []
                    self.start_scores[env_index] = scores
            done = self.envs.done()
            done_indices = [index for index, value in enumerate(done) if value and index in active]
            if done_indices:
                ranks.extend(self._finish_games(done_indices))
                if reset_completed:
                    self._reset_games(done_indices)
            return completed, rewards, ranks, completed_kyokus, ended_kyoku_indices, done_indices

        def collect(self) -> tuple[list[Transition], dict[str, float]]:
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
                "sampled_seats_per_game": 2.0,
                "drain_kyokus": float(drain_kyokus),
                "drain_steps": float(drain_steps),
            }
            stats.update(self.profiler.delta({}, prefix="timing"))
            return transitions, stats

        def evaluate(self, games: int) -> dict[str, float]:
            rewards: list[float] = []
            ranks: list[int] = []
            while len(ranks) < games * 2:
                _transitions, new_rewards, new_ranks, _kyokus, _ended, _done = self._advance_once(greedy=True, record=False)
                rewards.extend(new_rewards)
                ranks.extend(new_ranks)
            return {"eval_kyoku_reward": float(np.mean(rewards)) if rewards else 0.0, "eval_rank": float(np.mean(ranks[: games * 2]))}
else:
    RolloutWorker = None
