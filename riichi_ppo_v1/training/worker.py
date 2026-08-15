"""Ray rollout actors.  Workers own environments and Rust state-machine slots."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
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
    terminal_hanchan_rank_rewards,
    terminal_kyoku_reward,
)
from .trajectory import Transition, finish_kyoku_qboost


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


def history_namespace(update: int) -> str:
    """Inference namespace label for one frozen historical-policy checkpoint."""
    return f"history:u{int(update):03d}"


def parse_checkpoint_updates(checkpoint_dir: str | Path) -> list[int]:
    """Return sorted update numbers from atomic ``checkpoint_<5digit>.pt`` files."""
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        return []
    updates: list[int] = []
    pattern = re.compile(r"checkpoint_(\d{5})\.pt")
    for path in directory.glob("checkpoint_*.pt"):
        match = pattern.fullmatch(path.name)
        if match is not None:
            updates.append(int(match.group(1)))
    return sorted(updates)


def eligible_history_updates(
    updates: list[int],
    *,
    current_update: int,
    min_update: int,
    lag_updates: int,
) -> list[int]:
    """Keep checkpoints that are mature enough for the historical pool."""
    minimum = max(0, int(min_update))
    lag = max(0, int(lag_updates))
    return [
        update
        for update in sorted(int(value) for value in updates)
        if update >= minimum and int(current_update) - update >= lag
    ]


def normalize_opponent_fractions(
    current_frac: float,
    sft_frac: float,
    historical_frac: float,
    random_frac: float,
) -> tuple[float, float, float, float]:
    """Normalize opponent-mix fractions to sum to one."""
    total = (
        float(current_frac)
        + float(sft_frac)
        + float(historical_frac)
        + float(random_frac)
    )
    if total <= 0.0:
        raise ValueError("opponent_mix fractions must sum to a positive value")
    return (
        float(current_frac) / total,
        float(sft_frac) / total,
        float(historical_frac) / total,
        float(random_frac) / total,
    )


def resolve_opponent_fractions(
    *,
    current_frac: float,
    sft_frac: float,
    historical_frac: float,
    random_frac: float,
    history_pool: list[int],
) -> tuple[float, float, float, float]:
    """Drop the historical share and renormalize 70/30 when the pool is empty."""
    if float(historical_frac) > 0.0 and not history_pool:
        sft_frac = max(0.0, 1.0 - float(current_frac))
        historical_frac = 0.0
    return normalize_opponent_fractions(
        current_frac, sft_frac, historical_frac, random_frac,
    )


def should_record_transition(policy: str) -> bool:
    """Only the current self-play policy contributes learner transitions."""
    return policy == "current"


def rollout_lineup(
    rng: np.random.Generator,
    *,
    current_frac: float,
    sft_frac: float,
    historical_frac: float,
    random_frac: float,
    history_pool: list[int],
) -> tuple[str, str, str, str]:
    """Roll one table: all-current or one seat replaced by an opponent."""
    roll = float(rng.random())
    if roll < current_frac:
        return ("current",) * NUM_PLAYERS
    if roll < current_frac + sft_frac:
        opponent = "sft"
    elif roll < current_frac + sft_frac + historical_frac:
        if not history_pool:
            raise RuntimeError("historical opponent selected with an empty history pool")
        opponent = history_namespace(int(rng.choice(history_pool)))
    else:
        opponent = "random"
    seat = int(rng.integers(0, NUM_PLAYERS))
    return tuple(
        opponent if index == seat else "current"
        for index in range(NUM_PLAYERS)
    )


def build_rollout_lineups(
    *,
    num_envs: int,
    rng: np.random.Generator,
    current_frac: float,
    sft_frac: float,
    historical_frac: float,
    random_frac: float,
    history_pool: list[int],
) -> list[tuple[str, str, str, str]]:
    return [
        rollout_lineup(
            rng,
            current_frac=current_frac,
            sft_frac=sft_frac,
            historical_frac=historical_frac,
            random_frac=random_frac,
            history_pool=history_pool,
        )
        for _env_index in range(int(num_envs))
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
            self.walls = list(self.envs.walls())
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
                config.get("kyoku_reward_clip_points", 32_000)
            )
            if self.kyoku_reward_clip_points <= 0:
                raise ValueError("kyoku_reward_clip_points must be positive")
            self.deferred_reset_indices: set[int] = set()
            self.semantic = SemanticMetrics()
            self.lineups: list[tuple[str, str, str, str]] = [("current",) * NUM_PLAYERS for _ in range(self.num_envs)]
            self.history_pool: list[int] = []
            self._opponent_rng = np.random.default_rng(
                int(config["seed"]) * 1_000_003 + self.worker_id * 131
            )

        def set_rollout_context(
            self,
            update: int,
        ) -> None:
            """Install the config-driven rollout lineup.

            Default remains all-current self-play. With ``opponent_mix.enabled``
            (V14/E2), a fraction of envs replace exactly one seat with a frozen
            SFT policy (``sft``, greedy), a frozen historical PPO checkpoint
            (``history:uNNN``, greedy) and/or a uniform-random policy
            (``random``). Rewards/transitions are only recorded for ``current``
            seats.  When the historical pool is empty the historical share is
            renormalized onto current/SFT (70/30 relative).
            """
            mix = self.config.get("opponent_mix") or {}
            if not bool(mix.get("enabled", False)):
                self.lineups = [("current",) * NUM_PLAYERS for _ in range(self.num_envs)]
                return
            current_frac = float(mix.get("current_frac", 0.8))
            sft_frac = float(mix.get("sft_frac", 0.2))
            historical_frac = float(mix.get("historical_frac", 0.0))
            random_frac = float(mix.get("random_frac", 0.0))
            min_update = int(mix.get("historical_min_update", 0))
            lag_updates = int(mix.get("historical_lag_updates", 0))
            pool = eligible_history_updates(
                parse_checkpoint_updates(str(self.config["checkpoint_dir"])),
                current_update=int(update),
                min_update=min_update,
                lag_updates=lag_updates,
            )
            current_frac, sft_frac, historical_frac, random_frac = (
                resolve_opponent_fractions(
                    current_frac=current_frac,
                    sft_frac=sft_frac,
                    historical_frac=historical_frac,
                    random_frac=random_frac,
                    history_pool=pool,
                )
            )
            rng = np.random.default_rng(
                int(self.config["seed"]) * 1_000_003
                + int(update) * 131
                + self.worker_id * 7
            )
            self.lineups = build_rollout_lineups(
                num_envs=self.num_envs,
                rng=rng,
                current_frac=current_frac,
                sft_frac=sft_frac,
                historical_frac=historical_frac,
                random_frac=random_frac,
                history_pool=pool,
            )
            self.history_pool = pool

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
            ) = self.bridge.prepare(decisions, analysis_batch, walls=self.walls)
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
            q_taken = [float(value) for value in result["q_taken"]]
            expected_q = [float(value) for value in result["expected_q"]]
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
                            legal[row].copy(), action_id, logprobs[row], q_taken[row],
                            expected_q=expected_q[row],
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
            with self.profiler.stage("env/walls_refresh_after_reset"):
                self.walls = list(self.envs.walls())
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
                by_policy: dict[str, list[Any]] = {}
                for decision in decisions:
                    policy = self.lineups[decision.env_index][decision.seat_id]
                    by_policy.setdefault(policy, []).append(decision)
                pending_model_requests: list[
                    tuple[
                        str,
                        list[Decision],
                        DecisionAnalysisBatch,
                        Any,
                        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                    ]
                ] = []
                for policy, policy_decisions in by_policy.items():
                    if policy == "random":
                        actions = []
                        transitions = [None] * len(policy_decisions)
                        for decision in policy_decisions:
                            legal_actions = list(decision.observation.legal_actions())
                            if not legal_actions:
                                raise RuntimeError("random opponent seat has no legal action")
                            actions.append(legal_actions[
                                int(self._opponent_rng.integers(0, len(legal_actions)))
                            ])
                        for decision, action, transition in zip(
                            policy_decisions, actions, transitions, strict=True,
                        ):
                            actions_by_env[decision.env_index][decision.seat_id] = action
                            if transition is not None:
                                self.pending[decision.env_index][decision.seat_id].append(transition)
                        continue
                    with self.profiler.stage("rollout/reward_analysis"):
                        analysis_batch = DecisionAnalysisBatch.build(
                            policy_decisions, analyzer=self.efficiency, public=self.public,
                            profiler=self.profiler,
                        )
                    # ``policy`` is "rollout"/"sft" for the built-in models and
                    # "history:uNNN" for frozen historical checkpoints; the
                    # inference actor dispatches on the namespace itself.
                    namespace = "rollout" if policy == "current" else str(policy)
                    greedy = policy != "current"
                    request, prepared = self._submit_model_actions(
                        policy_decisions, namespace, greedy, analysis_batch,
                    )
                    pending_model_requests.append((
                        policy, policy_decisions, analysis_batch, request, prepared,
                    ))

                # Submit every policy namespace before blocking.  This lets the
                # inference actor overlap the first request with preparation of
                # the remaining namespaces and exposes more requests to its
                # cross-worker batcher at the same time.
                if pending_model_requests:
                    with self.profiler.stage("inference/rpc_wait"):
                        model_results = ray.get([
                            request
                            for _policy, _decisions, _analysis, request, _prepared
                            in pending_model_requests
                        ])
                else:
                    model_results = []
                for (
                    policy, policy_decisions, analysis_batch, _request, prepared,
                ), result in zip(pending_model_requests, model_results, strict=True):
                    record_policy = should_record_transition(policy)
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
            with self.profiler.stage("env/walls_refresh_after_step"):
                self.walls = list(self.envs.walls())
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
                    rank_rewards = (
                        terminal_hanchan_rank_rewards(scores)
                        if done[env_index] else (0.0, 0.0, 0.0, 0.0)
                    )
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
                        kyoku_reward = terminal_kyoku_reward(
                            scores[seat] - self.start_scores[env_index][seat],
                            self.kyoku_reward_clip_points,
                        )
                        rank_reward = float(rank_rewards[seat])
                        reward = kyoku_reward + rank_reward
                        pending = self.pending[env_index][seat]
                        if self.lineups[env_index][seat] == "current":
                            with self.profiler.stage("rollout/finish_kyoku_qboost"):
                                if pending:
                                    pending[-1].kyoku_reward = kyoku_reward
                                    pending[-1].hanchan_rank_reward = rank_reward
                                    pending[-1].reward += reward
                                completed.extend(finish_kyoku_qboost(
                                    pending, float(self.config["gamma"]),
                                    float(self.config["qboost_lambda"]),
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
            lineup_counts: Counter[str] = Counter()
            for lineup in self.lineups:
                lineup_counts.update(lineup)
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
                "opponent_mix/current_seats": float(lineup_counts["current"]),
                "opponent_mix/sft_seats": float(lineup_counts["sft"]),
                "opponent_mix/history_seats": float(
                    sum(
                        policy.startswith("history:")
                        for lineup in self.lineups
                        for policy in lineup
                    )
                ),
                "opponent_mix/random_seats": float(lineup_counts["random"]),
                "opponent_mix/history_pool_size": float(len(self.history_pool)),
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
