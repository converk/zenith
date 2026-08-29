"""Ray rollout actors:V18 编码装配与纯 GRP 边界奖励推理。"""

from __future__ import annotations

import json
import re
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import ray
except ImportError:  # imported lazily by the command line program
    ray = None

from ..model.bridge import NUM_PLAYERS, BatchedStateBridge, Decision
from ..model.grp import (
    GRP_HIDDEN,
    GRP_INPUT_SIZE,
    GRP_LAYERS,
    PREV_RESULT_RON,
    PREV_RESULT_RYUKYOKU,
    PREV_RESULT_TSUMO,
    GRPModel,
)
from .grp.prepare import (
    Boundary,
    KyokuResult,
    feature_row,
    game_type_from_mode,
    rank_among,
    result_increment,
)
from .grp.reward import RANK_UTILITY, rank_utility
from .metrics import SemanticMetrics
from .profiling import StageProfiler
from .rollout_buffer import RolloutBuffer
from .trajectory import Transition, finish_kyoku_gae


def _configure_rollout_torch_threads(config: dict[str, Any]) -> None:
    """尽早限制 rollout actor 内的 PyTorch CPU 线程池。"""
    value = config.get("rollout_worker_cpu_threads")
    if value is None:
        return
    threads = int(value)
    if threads <= 0:
        raise ValueError("rollout_worker_cpu_threads must be positive")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)


def _process_cpu_snapshot() -> dict[str, float]:
    """读取当前 actor 的线程数、上下文切换与常驻内存。"""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    try:
        threads = sum(1 for _entry in Path("/proc/self/task").iterdir())
    except OSError:
        threads = 0
    return {
        "threads": float(threads),
        "voluntary_context_switches": float(usage.ru_nvcsw),
        "involuntary_context_switches": float(usage.ru_nivcsw),
        "max_rss_kb": float(usage.ru_maxrss),
    }


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
    """Drop the historical share and renormalize when the pool is empty."""
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


class GrpRollout:
    """每个小局边界执行一次 GRP 的纯奖励装配(Mortal 方案,V18 契约)。

    每环境维护 1 条 21 维全局特征前缀序列与累计计数(各玩家和了/放铳/听牌流局
    次数);边界行由 ``feature_row`` 逐边界生成,与离线数据构造逐位一致;局风
    类型由 ``game_mode`` 经 ``game_type_from_mode`` 映射,每环境固定。每个非
    终局边界对整条序列执行 1 次 GRU 前向(输出 24 类 logits),经 calc_matrix
    得到 4 玩家期望 utility 并计算本小局 δ;终局(半庄结束)使用真实最终排名
    utility。动作数量不影响 GRP 调用次数。reward 为纯 GRP delta,无点差分量、
    无 σ 归一化。
    """

    def __init__(self, model: Any, game_type: int) -> None:
        self.model = model
        self.game_type = int(game_type)
        self.calls = 0
        # env -> 21 维特征前缀序列(行按边界追加)。
        self._sequences: dict[int, np.ndarray] = {}
        # env -> 累计计数(截至下一小局开始的各玩家和了/放铳/听牌流局次数)。
        self._wins: dict[int, list[int]] = {}
        self._dealins: dict[int, list[int]] = {}
        self._tenpai: dict[int, list[int]] = {}
        self._previous_v: dict[tuple[int, int], float] = {}

    def _expected_utility(self, env_index: int) -> torch.Tensor:
        """对某环境的完整 prefix 跑一次 GRU,返回 4 玩家期望 utility [4]。"""
        sequence = np.asarray(self._sequences[int(env_index)], dtype=np.float32)
        with torch.no_grad():
            logits = self.model(
                torch.as_tensor(sequence[None]),
                torch.as_tensor([len(sequence)]),
            )
        self.calls += 1
        matrix = self.model.calc_matrix(logits[0:1])[0]  # (4,4) 玩家→排名概率
        # 排名 utility 收敛为 grp/reward.RANK_UTILITY 单一来源(与离线一致)。
        return matrix @ logits.new_tensor(RANK_UTILITY)

    def start_match(self, env_index: int, boundary: Boundary) -> None:
        """登记首局边界并计算 V_0(1 次 GRP 调用/环境),累计计数归零。"""
        key = int(env_index)
        self._wins[key] = [0, 0, 0, 0]
        self._dealins[key] = [0, 0, 0, 0]
        self._tenpai[key] = [0, 0, 0, 0]
        self._sequences[key] = np.asarray(
            feature_row(
                boundary, self.game_type,
                (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0),
            ),
            dtype=np.float32,
        )[None, :]
        expected = self._expected_utility(key)
        for seat in range(NUM_PLAYERS):
            self._previous_v[(key, seat)] = float(expected[seat])

    def boundary_reward(
        self,
        env_index: int,
        boundary: Boundary,
        *,
        terminal_ranks: dict[int, int] | None = None,
    ) -> dict[int, float]:
        """小局边界结算:非终局跑 1 次 GRP(全 4 玩家),终局用真实排名 utility。

        返回 {seat: reward}:reward = V(boundary_{k+1}) - V(boundary_k);
        终局 = 真实最终排名 utility - V(终局前边界)。非终局时先以
        ``boundary.previous``(刚结束小局的结果)推进累计计数,再生成边界行。
        """
        if terminal_ranks is None:
            key = int(env_index)
            w, d, t = result_increment(boundary.previous)
            wins = self._wins[key]
            dealins = self._dealins[key]
            tenpai = self._tenpai[key]
            for seat in range(NUM_PLAYERS):
                wins[seat] += w[seat]
                dealins[seat] += d[seat]
                tenpai[seat] += t[seat]
            current_row = np.asarray(
                feature_row(
                    boundary, self.game_type,
                    tuple(wins), tuple(dealins), tuple(tenpai),
                ),
                dtype=np.float32,
            )
            self._sequences[key] = np.concatenate([
                self._sequences[key], current_row[None, :],
            ], axis=0)
            current_expected = self._expected_utility(key)
            rewards: dict[int, float] = {}
            for seat in range(NUM_PLAYERS):
                previous_key = (key, int(seat))
                previous_v = self._previous_v[previous_key]
                current_v = float(current_expected[seat])
                rewards[seat] = float(current_v - previous_v)
                self._previous_v[previous_key] = current_v
            return rewards
        # 终局:半庄结束,不再追加边界行,直接用真实排名 utility 计算 δ。
        rewards = {}
        for seat in range(NUM_PLAYERS):
            key = (int(env_index), int(seat))
            rewards[seat] = float(
                rank_utility(int(terminal_ranks[seat])) - self._previous_v[key]
            )
        return rewards


def _previous_result(
    events: list[list[str]],
    tenpai_flags: dict[int, bool] | None = None,
) -> KyokuResult | None:
    """从刚结束小局的事件流解析结果(无结果则为 None)。"""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for seat_events in events:
        for raw in seat_events:
            if raw in seen:
                continue
            seen.add(raw)
            try:
                rows.append(json.loads(raw))
            except (TypeError, ValueError):
                continue
    horas = [row for row in rows if row.get("type") == "hora"]
    if horas:
        first = horas[0]
        actor = int(first.get("actor", -1))
        target = int(first.get("target", -1))
        tsumo = bool(first.get("tsumo", False)) or actor == target
        deltas = [0, 0, 0, 0]
        for row in horas:
            values = row.get("deltas") or []
            for seat, value in enumerate(values[:4]):
                deltas[seat] += int(value)
        return KyokuResult(
            PREV_RESULT_TSUMO if tsumo else PREV_RESULT_RON,
            actor,
            None if tsumo else target,
            0,
            tuple(deltas),
        )
    draws = [row for row in rows if row.get("type") == "ryukyoku"]
    if draws:
        first = draws[0]
        deltas = [0, 0, 0, 0]
        for row in draws:
            values = row.get("deltas") or []
            for seat, value in enumerate(values[:4]):
                deltas[seat] += int(value)
        mask = 0
        if str(first.get("reason", "")) == "exhaustive_draw":
            for seat, flag in (tenpai_flags or {}).items():
                if flag:
                    mask |= 1 << int(seat)
        return KyokuResult(PREV_RESULT_RYUKYOKU, None, None, mask, tuple(deltas))
    return None


if ray is not None:
    @ray.remote
    class RolloutWorker:
        def __init__(
            self,
            worker_id: int,
            config: dict[str, Any],
            inference: Any,
        ) -> None:
            _configure_rollout_torch_threads(config)
            try:
                import riichi
                from riichienv import BatchedRiichiEnv, HandEvaluator
            except ImportError as exc:
                raise RuntimeError("install local riichi and RiichiEnv extensions before starting workers") from exc
            self.config = config
            self.worker_id = int(worker_id)
            self.inference = inference
            self.hand_evaluator = HandEvaluator
            # 环境 worker 刻意不 import CUDA、不持有策略模型;唯一推理 actor
            # 独占 GPU 执行。GRP 是约 13 万参数的 CPU 小模型,仅小局边界执行。
            self.num_envs = int(config["envs_per_worker"])
            self.envs = BatchedRiichiEnv(
                self.num_envs,
                seed=int(config["seed"]) + worker_id * 1_000,
                step_threads=int(config.get("env_step_threads", 4)),
                game_mode=config["game_mode"],
            )
            self.state_machine = riichi.MjaiKyokuStateMachineManager(self.num_envs)
            self.profiler = StageProfiler(enabled=bool(config.get("profile_enabled", True)))
            # V18 action query 只保留 Rust 紧凑事实与融合编码路径。
            self.bridge = BatchedStateBridge(
                self.state_machine,
                self.num_envs,
                self.profiler,
            )
            self.observations = list(self.envs.reset())
            self.walls = list(self.envs.walls())
            self.bridge.sync(self.observations)
            self.start_scores = [[int(x) for x in scores] for scores in self.envs.scores()]
            self.match_kyoku_counts = [0] * self.num_envs
            self.pending: list[list[list[Transition]]] = [
                [[] for _ in range(NUM_PLAYERS)] for _ in range(self.num_envs)
            ]
            self.model_decisions = 0
            self.recorded_decisions = 0
            self.semantic = SemanticMetrics()
            self.lineups: list[tuple[str, str, str, str]] = [
                ("current",) * NUM_PLAYERS for _ in range(self.num_envs)
            ]
            self.history_pool: list[int] = []
            self._opponent_rng = np.random.default_rng(
                int(config["seed"]) * 1_000_003 + self.worker_id * 131
            )
            self._pending_tenpai: dict[int, dict[int, bool]] = {}
            self.deferred_reset_indices: set[int] = set()
            # GRP:CPU 小模型,只在小局边界执行;PPO 阶段只读不更新。
            grp_payload = torch.load(
                config["grp_checkpoint"], map_location="cpu", weights_only=False,
            )
            # 结构按 checkpoint 的 model_config 构造(离线契约单一来源)。
            model_config = grp_payload.get("model_config") or {}
            grp_model = GRPModel(
                input_size=int(model_config.get("input_size", GRP_INPUT_SIZE)),
                hidden_size=int(model_config.get("hidden", GRP_HIDDEN)),
                num_layers=int(model_config.get("layers", GRP_LAYERS)),
            )
            grp_model.load_state_dict(grp_payload["model"], strict=True)
            grp_model.freeze()  # PPO 不更新 GRP
            self.grp = GrpRollout(
                grp_model, game_type_from_mode(str(config["game_mode"]))
            )
            for env_index in range(self.num_envs):
                self.grp.start_match(
                    env_index, self._boundary_from_observations(env_index, None)
                )

        def set_rollout_context(self, update: int) -> None:
            """Install the config-driven rollout lineup."""
            mix = self.config.get("opponent_mix") or {}
            if not bool(mix.get("enabled", False)):
                self.lineups = [("current",) * NUM_PLAYERS for _ in range(self.num_envs)]
                return
            current_frac = float(mix.get("current_frac", 0.8))
            sft_frac = float(mix.get("sft_frac", 0.2))
            historical_frac = float(mix.get("historical_frac", 0.0))
            random_frac = float(mix.get("random_frac", 0.0))
            pool = eligible_history_updates(
                parse_checkpoint_updates(str(self.config["checkpoint_dir"])),
                current_update=int(update),
                min_update=int(mix.get("historical_min_update", 0)),
                lag_updates=int(mix.get("historical_lag_updates", 0)),
            )
            fractions = resolve_opponent_fractions(
                current_frac=current_frac,
                sft_frac=sft_frac,
                historical_frac=historical_frac,
                random_frac=random_frac,
                history_pool=pool,
            )
            rng = np.random.default_rng(
                int(self.config["seed"]) * 1_000_003
                + int(update) * 131
                + self.worker_id * 7
            )
            self.lineups = build_rollout_lineups(
                num_envs=self.num_envs,
                rng=rng,
                current_frac=fractions[0],
                sft_frac=fractions[1],
                historical_frac=fractions[2],
                random_frac=fractions[3],
                history_pool=pool,
            )
            self.history_pool = pool

        def _boundary_from_observations(
            self, env_index: int, previous: KyokuResult | None,
        ) -> Boundary:
            observation = self.observations[env_index][0]
            return Boundary(
                round_wind=0 if int(observation.round_wind) == 0 else 1,
                kyoku_index=min(max(int(observation.kyoku_index), 0), 7),
                dealer=int(observation.oya),
                honba=int(observation.honba),
                sticks=int(observation.riichi_sticks),
                scores=tuple(int(value) for value in self.envs.scores()[env_index]),
                previous=previous,
            )

        def _final_tenpai_flags(
            self, env_index: int, actions_by_env: dict[int, Any],
        ) -> dict[int, bool]:
            """牌山耗尽前计算各座位听牌,供流局 exhaustive_draw 构造 tenpai 掩码。"""
            flags: dict[int, bool] = {}
            observations = self.observations[env_index]
            for seat in range(NUM_PLAYERS):
                obs = observations[seat]
                hands = getattr(obs, "hands", None)
                melds = getattr(obs, "melds", None)
                if hands is None or melds is None:
                    continue
                hand = list(hands[seat])
                meld_list = list(melds[seat])
                tile_count = len(hand) + 3 * len(meld_list)
                if tile_count == 13:
                    flags[seat] = self.hand_evaluator(hand, meld_list).is_tenpai()
                elif tile_count == 14:
                    action = actions_by_env.get(seat)
                    tile = getattr(action, "tile", None)
                    if tile is not None and int(tile) in hand:
                        remaining = list(hand)
                        remaining.remove(int(tile))
                        flags[seat] = self.hand_evaluator(remaining, meld_list).is_tenpai()
            return flags

        def _submit_model_actions(
            self,
            decisions: list[Decision],
            namespace: str,
            greedy: bool,
        ) -> tuple[Any, dict[str, np.ndarray]]:
            with self.profiler.stage("rollout/model_state_prepare"):
                batch = self.bridge.prepare(decisions, walls=self.walls)
            request = self.inference.infer.remote(
                worker_id=self.worker_id,
                namespace=namespace,
                batch_indices=np.asarray(
                    [decision.batch_index for decision in decisions], dtype=np.int64,
                ),
                actor_factors=batch.actor_factors,
                actor_numeric=batch.actor_numeric,
                actor_lengths=batch.actor_lengths,
                query_action_ids=batch.query_action_ids,
                query_pair_counts=batch.query_pair_counts,
                legal_mask=batch.legal_mask,
                critic_factors=batch.critic_factors,
                critic_lengths=batch.critic_lengths,
                greedy=greedy,
            )
            return request, {
                "actor_factors": batch.actor_factors,
                "actor_numeric": batch.actor_numeric,
                "actor_lengths": batch.actor_lengths,
                "query_action_ids": batch.query_action_ids,
                "query_pair_counts": batch.query_pair_counts,
                "legal_mask": batch.legal_mask,
                "critic_factors": batch.critic_factors,
                "critic_lengths": batch.critic_lengths,
            }

        def _model_actions(
            self,
            decisions: list[Decision],
            prepared: dict[str, np.ndarray],
            result: dict[str, Any],
            record: bool,
        ) -> tuple[list[Any], list[Transition | None]]:
            action_ids = [int(value) for value in result["action_ids"]]
            logprobs = [float(value) for value in result["logprobs"]]
            values = [float(value) for value in result["values"]]
            with self.profiler.stage("rollout/model_action_decode"):
                actions = self.bridge.decode(decisions, action_ids)
            self.model_decisions += len(decisions)
            transitions: list[Transition | None] = [None] * len(decisions)
            if record:
                with self.profiler.stage("rollout/transition_materialize"):
                    for row, decision in enumerate(decisions):
                        actor_length = int(prepared["actor_lengths"][row])
                        pair_count = int(prepared["query_pair_counts"][row])
                        critic_length = int(prepared["critic_lengths"][row])
                        transitions[row] = Transition(
                            prepared["actor_factors"][row, :actor_length].copy(),
                            prepared["actor_numeric"][row, :actor_length].copy(),
                            actor_length,
                            prepared["query_action_ids"][row, :pair_count].copy(),
                            pair_count,
                            prepared["legal_mask"][row].copy(),
                            action_ids[row],
                            logprobs[row],
                            values[row],
                            critic_factors=(
                                prepared["critic_factors"][row, :critic_length].copy()
                                if critic_length else None
                            ),
                            critic_length=critic_length,
                        )
                        self.semantic.record_decision(
                            action_ids[row], prepared["legal_mask"][row],
                        )
                        self.recorded_decisions += 1
            return actions, transitions

        def _finish_games(self, done_indices: list[int]) -> None:
            """Discard all four completed-seat trajectories before reset."""
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
            scores_by_env = self.envs.scores()
            for env_index in done_indices:
                self.grp.start_match(
                    env_index, self._boundary_from_observations(env_index, None)
                )
                self.start_scores[env_index] = [
                    int(value) for value in scores_by_env[env_index]
                ]
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
                by_policy: dict[str, list[Decision]] = {}
                for decision in decisions:
                    policy = self.lineups[decision.env_index][decision.seat_id]
                    by_policy.setdefault(policy, []).append(decision)
                pending_model_requests: list[
                    tuple[str, list[Decision], Any, dict[str, np.ndarray]]
                ] = []
                for policy, policy_decisions in by_policy.items():
                    if policy == "random":
                        for decision in policy_decisions:
                            legal_actions = list(decision.observation.legal_actions())
                            if not legal_actions:
                                raise RuntimeError("random opponent seat has no legal action")
                            actions_by_env[decision.env_index][decision.seat_id] = (
                                legal_actions[
                                    int(self._opponent_rng.integers(0, len(legal_actions)))
                                ]
                            )
                        continue
                    # ``policy`` 为 "current"/"sft"/"history:uNNN";推理 actor 按
                    # namespace 自行分发。
                    namespace = "rollout" if policy == "current" else str(policy)
                    greedy = policy != "current"
                    request, prepared = self._submit_model_actions(
                        policy_decisions, namespace, greedy,
                    )
                    pending_model_requests.append((policy, policy_decisions, request, prepared))

                # 先把全部 namespace 请求都提交,再阻塞等待;推理 actor 得以在
                # 其余请求编码期间并行服务首批请求。
                if pending_model_requests:
                    with self.profiler.stage("inference/rpc_wait"):
                        model_results = ray.get([
                            request
                            for _policy, _decisions, request, _prepared
                            in pending_model_requests
                        ])
                else:
                    model_results = []
                for (
                    policy, policy_decisions, _request, prepared,
                ), result in zip(pending_model_requests, model_results, strict=True):
                    record = should_record_transition(policy)
                    actions, transitions = self._model_actions(
                        policy_decisions, prepared, result, record,
                    )
                    for decision, action, transition in zip(
                        policy_decisions, actions, transitions, strict=True,
                    ):
                        actions_by_env[decision.env_index][decision.seat_id] = action
                        if transition is not None:
                            self.pending[decision.env_index][decision.seat_id].append(
                                transition
                            )

            # 牌山耗尽前先记录各座位听牌,供流局边界构造 tenpai 掩码。
            self._pending_tenpai.clear()
            for env_index in active:
                tiles_left = min(
                    int(getattr(obs, "tiles_left", 1))
                    for obs in self.observations[env_index].values()
                )
                if tiles_left <= 0:
                    self._pending_tenpai[env_index] = self._final_tenpai_flags(
                        env_index, actions_by_env[env_index]
                    )
            with self.profiler.stage("env/step_batch_native"):
                self.observations = list(self.envs.step_batch(actions_by_env))
            with self.profiler.stage("env/walls_refresh_after_step"):
                self.walls = list(self.envs.walls())
            with self.profiler.stage("rollout/event_sync_after_step"):
                end_kyoku, _end_game = self.bridge.sync(self.observations)
            scores_by_env = self.envs.scores()
            done = self.envs.done()
            done_indices = [
                index for index, value in enumerate(done) if value and index in active
            ]
            completed_kyokus = 0
            ended_kyoku_indices: list[int] = []
            for env_index in range(self.num_envs):
                if env_index not in active or not end_kyoku[env_index]:
                    continue
                completed_kyokus += 1
                ended_kyoku_indices.append(env_index)
                self.match_kyoku_counts[env_index] += 1
                previous = _previous_result(
                    self.bridge.last_events[env_index],
                    self._pending_tenpai.get(env_index),
                )
                boundary = self._boundary_from_observations(env_index, previous)
                terminal_ranks = None
                if done[env_index]:
                    scores = tuple(int(value) for value in scores_by_env[env_index])
                    terminal_ranks = {
                        seat: rank_among(seat, scores) for seat in range(NUM_PLAYERS)
                    }
                seat_rewards = self.grp.boundary_reward(
                    env_index, boundary, terminal_ranks=terminal_ranks,
                )
                current_seats = [
                    seat
                    for seat, policy in enumerate(self.lineups[env_index])
                    if policy == "current"
                ]
                self.semantic.record_kyoku(
                    current_seats,
                    [
                        int(scores_by_env[env_index][seat])
                        - self.start_scores[env_index][seat]
                        for seat in range(NUM_PLAYERS)
                    ],
                    self.bridge.last_events[env_index],
                    # 只传牌山耗尽前算好的座位听牌掩码;exhaustive_draw 由
                    # metrics 从终局事件 reason 自动判定,非荒牌流局不计入。
                    draw_tenpai=self._pending_tenpai.get(env_index),
                )
                for seat in range(NUM_PLAYERS):
                    reward = float(seat_rewards[seat])
                    pending = self.pending[env_index][seat]
                    if self.lineups[env_index][seat] == "current":
                        with self.profiler.stage("rollout/finish_kyoku_gae"):
                            if pending:
                                pending[-1].reward += reward
                                pending[-1].kyoku_reward = reward
                            completed.extend(finish_kyoku_gae(
                                pending,
                                float(self.config["gamma"]),
                                float(self.config.get("gae_lambda", 0.95)),
                            ))
                    rewards.append(reward)
                    self.pending[env_index][seat] = []
                self.start_scores[env_index] = [
                    int(value) for value in scores_by_env[env_index]
                ]
                if done[env_index]:
                    self.semantic.record_match_length(self.match_kyoku_counts[env_index])
            if done_indices:
                self._finish_games(done_indices)
            return completed, rewards, completed_kyokus, ended_kyoku_indices, done_indices

        def collect(
            self,
            update: int | None = None,
        ) -> tuple[RolloutBuffer, dict[str, float]]:
            if update is not None:
                self.set_rollout_context(int(update))
            # rollout 停止条件为「完整半庄数」:``games_per_update`` 是全局每
            # update 完整半庄目标,按 worker 数分摊;原生环境并行推进各桌,可因
            # 一波在途结算而小幅超额,这一有界超额优于丢弃半局。
            games_per_update = max(1, int(self.config["games_per_update"]))
            num_workers = max(1, int(self.config.get("num_workers", 1)))
            target = -(-games_per_update // num_workers)
            transitions: list[Transition] = []
            rewards: list[float] = []
            kyokus = 0
            games = 0
            drain_kyokus = 0
            drain_steps = 0
            draining = False
            active_envs = set(range(self.num_envs))
            self.profiler.reset()
            self.semantic = SemanticMetrics()
            for lineup in self.lineups:
                self.semantic.record_lineup(lineup)
            self.model_decisions = 0
            self.recorded_decisions = 0
            grp_calls_start = self.grp.calls
            if self.deferred_reset_indices:
                self._reset_games(sorted(self.deferred_reset_indices))
                self.deferred_reset_indices.clear()
            lineup_counts: Counter[str] = Counter()
            for lineup in self.lineups:
                lineup_counts.update(lineup)
            started = time.perf_counter()
            cpu_started = _process_cpu_snapshot()
            while active_envs:
                step, new_rewards, new_kyokus, ended_kyokus, done_indices = (
                    self._advance_once(active_envs=active_envs)
                )
                transitions.extend(step)
                rewards.extend(new_rewards)
                kyokus += new_kyokus
                games += len(done_indices)
                if not draining and games >= target:
                    draining = True
                if draining:
                    # 小局收口或整局结束的桌子立即冻结,不再多收一个小局。
                    frozen = set(ended_kyokus) | set(done_indices)
                    drain_kyokus += new_kyokus
                    drain_steps += 1
                    active_envs.difference_update(frozen)
                    self.deferred_reset_indices.update(done_indices)
                else:
                    self._reset_games(done_indices)
            elapsed = time.perf_counter() - started
            stats = {
                "games": float(games),
                "kyokus": float(kyokus),
                "sampled_rewards": float(len(rewards)),
                "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
                "rollout_s": elapsed,
                "transitions_per_s": float(len(transitions) / max(elapsed, 1e-9)),
                "model_decisions": float(self.model_decisions),
                "recorded_decisions": float(self.recorded_decisions),
                "grp_calls": float(self.grp.calls - grp_calls_start),
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
                "drain_kyokus": float(drain_kyokus),
                "drain_steps": float(drain_steps),
            }
            stats.update(self.profiler.delta({}, prefix="timing"))
            semantic_started = time.perf_counter()
            for transition in transitions:
                self.semantic.record_transition_reward(transition)
            stats.update(self.semantic.summary())
            stats["semantic_summary_s"] = time.perf_counter() - semantic_started
            payload_started = time.perf_counter()
            payload = RolloutBuffer(transitions)
            stats["worker_soa_pack_s"] = time.perf_counter() - payload_started
            array_count, array_bytes = payload.payload_stats()
            stats["return_array_count"] = float(array_count)
            stats["return_array_bytes"] = float(array_bytes)
            stats["return_payload_profile_s"] = time.perf_counter() - payload_started
            cpu_finished = _process_cpu_snapshot()
            stats.update({
                "worker_id": float(self.worker_id),
                "collect_total_s": time.perf_counter() - started,
                "system/threads_start": cpu_started["threads"],
                "system/threads_end": cpu_finished["threads"],
                "system/voluntary_context_switches": (
                    cpu_finished["voluntary_context_switches"]
                    - cpu_started["voluntary_context_switches"]
                ),
                "system/involuntary_context_switches": (
                    cpu_finished["involuntary_context_switches"]
                    - cpu_started["involuntary_context_switches"]
                ),
                "system/max_rss_kb": cpu_finished["max_rss_kb"],
                "system/torch_threads": float(torch.get_num_threads()),
                "system/torch_interop_threads": float(torch.get_num_interop_threads()),
            })
            return payload, stats

else:
    RolloutWorker = None
