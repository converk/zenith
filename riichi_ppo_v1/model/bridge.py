"""RiichiEnv、MJAI 与 Rust 状态机之间的严格转换边界。

V18 当前局面输入直接由 ``current_state.encode_batch`` 从原生 Observation 构造；
本模块保留动作解码/合法掩码/生命周期职责，不再装配事件历史或 Atomic Snapshot。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..training.profiling import StageProfiler
from .critic_features import (
    collect_visible_table_state,
    empty_critic_features,
    encode_critic_features,
    pad_critic_feature_rows,
)
from .current_state import EncodedStateBatch, encode_batch
from .schema import TID_COUNT

NUM_PLAYERS = 4
_DECISION_ACTION_TYPES = frozenset({"dahai", "reach", "ankan", "kakan", "ryukyoku"})


@dataclass(frozen=True)
class Decision:
    env_index: int
    seat_id: int
    observation: Any

    @property
    def batch_index(self) -> int:
        return self.env_index * NUM_PLAYERS + self.seat_id


@dataclass(frozen=True)
class PreparedBatch:
    """现行 V18 输入装配结果：完整 Actor 序列 + Query 元数据 + Critic 私有行。"""

    actor_factors: np.ndarray
    actor_numeric: np.ndarray
    actor_lengths: np.ndarray
    query_rows: np.ndarray
    query_action_ids: np.ndarray
    query_pair_counts: np.ndarray
    legal_mask: np.ndarray
    critic_factors: np.ndarray
    critic_lengths: np.ndarray


def tile_id_to_mjai(tile_id: int | None) -> str | None:
    """Convert RiichiEnv's physical 136-tile id to its MJAI spelling."""
    if tile_id is None:
        return None
    tile = int(tile_id)
    red = {16: "5mr", 52: "5pr", 88: "5sr"}
    if tile in red:
        return red[tile]
    if not 0 <= tile < TID_COUNT:
        raise ValueError(f"invalid RiichiEnv tile id {tile}")
    suit = tile // 36
    if suit < 3:
        return f"{tile % 36 // 4 + 1}{('m', 'p', 's')[suit]}"
    honors = ("E", "S", "W", "N", "P", "F", "C")
    return honors[(tile - 108) // 4]


@lru_cache(maxsize=1024)
def _normalized_action_json(raw_action: str, tsumogiri: bool) -> tuple[str, str]:
    """Return the canonical MJAI template and its type for one action shape."""
    value = json.loads(raw_action)
    action_type = str(value.get("type", ""))
    if action_type == "dahai":
        value["tsumogiri"] = bool(tsumogiri)
    expected_consumed = {"chi": 2, "pon": 2, "daiminkan": 3}.get(action_type)
    consumed = value.get("consumed")
    if expected_consumed is not None and isinstance(consumed, list) and len(consumed) == expected_consumed + 1:
        called = value.get("pai")
        try:
            consumed.remove(called)
        except ValueError as exc:
            raise ValueError(f"{action_type} replay action does not contain its called tile") from exc
        value["consumed"] = consumed
    return json.dumps(value, separators=(",", ":"), sort_keys=True), action_type


def _canonical_mjai(value: str) -> str:
    """把 MJAI 字符串重排为与 action_jsons 一致的规范 JSON。"""
    return json.dumps(json.loads(value), separators=(",", ":"), sort_keys=True)


def _action_objects_jsons_and_decision_flag(
    observation: Any,
) -> tuple[list[Any], list[str], int]:
    """一次遍历返回 Action 对象、规范模板与决策窗口标志。"""
    drawn = getattr(observation, "drawn_tile", None)
    objects: list[Any] = []
    result: list[str] = []
    has_decision_action = False
    for action in observation.legal_actions():
        objects.append(action)
        tsumogiri = action.tile is not None and drawn is not None and int(action.tile) == int(drawn)
        action_json, action_type = _normalized_action_json(action.to_mjai(), tsumogiri)
        result.append(action_json)
        has_decision_action |= action_type in _DECISION_ACTION_TYPES
    return objects, result, int(has_decision_action)


def action_jsons_and_decision_flag(observation: Any) -> tuple[list[str], int]:
    """返回规范动作模板与 snapshot 决策窗口标志。"""
    _objects, result, flag = _action_objects_jsons_and_decision_flag(observation)
    return result, flag


def action_jsons(observation: Any) -> list[str]:
    """Create exact action templates, including the physical tsumogiri distinction."""
    return action_jsons_and_decision_flag(observation)[0]


class BatchedStateBridge:
    """One strict, vectorized boundary for all tables owned by a rollout worker."""

    def __init__(
        self,
        state_machine: Any,
        num_envs: int,
        profiler: StageProfiler | None = None,
        *,
        critic_include_public_state: bool = False,
    ) -> None:
        self.state_machine = state_machine
        self.num_envs = int(num_envs)
        self.profiler = profiler or StageProfiler(enabled=False)
        self.critic_include_public_state = bool(critic_include_public_state)
        self.last_rust_stats: dict[str, int] = {}
        self.last_events: list[list[list[str]]] = [[[] for _ in range(NUM_PLAYERS)] for _ in range(num_envs)]
        self.observations_by_env: list[dict[int, Any]] | None = None
        # batch_index → {action_id: Action 对象};prepare 时直接登记状态机给
        # 出的 action_id→源下标映射,decode 免掉状态机模板往返与逐动作匹配。
        self._action_by_id: dict[int, dict[int, Any]] = {}

    def sync(self, observations_by_env: list[dict[int, Any]]) -> tuple[np.ndarray, np.ndarray]:
        if len(observations_by_env) != self.num_envs:
            raise RuntimeError(f"received {len(observations_by_env)} environments, expected {self.num_envs}")
        events_by_env: list[list[list[str]]] = []
        with self.profiler.stage("state/event_extract"):
            for env_index, observations in enumerate(observations_by_env):
                if set(observations) != set(range(NUM_PLAYERS)):
                    raise RuntimeError(f"env={env_index} must expose all four player observations")
                events_by_env.append([list(observations[seat].new_events()) for seat in range(NUM_PLAYERS)])
        self.observations_by_env = observations_by_env
        self.last_events = events_by_env
        with self.profiler.stage("state/rust_event_apply"):
            end_kyoku, end_game = self.state_machine.apply_events_batch(list(range(self.num_envs)), events_by_env)
        with self.profiler.stage("state/boundary_array_convert"):
            return np.asarray(end_kyoku, dtype=np.bool_), np.asarray(end_game, dtype=np.bool_)

    def decode(self, decisions: list[Decision], action_ids: list[int]) -> list[Any]:
        if len(decisions) != len(action_ids):
            raise ValueError("decisions and action_ids must have the same length")
        # 本步 prepare 已登记 action_id→Action 对象映射(与状态机
        # action_ids_with_source_indices 同一来源),直接本地查表,免去
        # decode_actions + select_action_from_mjai 的逐决策往返。
        final: list[Any] = []
        unhandled: list[int] = []
        for index, (decision, action_id) in enumerate(zip(decisions, action_ids, strict=True)):
            action = self._action_by_id.get(decision.batch_index, {}).get(int(action_id))
            if action is None:
                unhandled.append(index)
                final.append(None)
            else:
                final.append(action)
        if unhandled:
            # 兜底:未走 prepare(或映射缺失)的回退到状态机模板 + env 匹配路径。
            with self.profiler.stage("state/rust_decode_actions"):
                mjai_actions = self.state_machine.decode_actions(
                    [decisions[index].batch_index for index in unhandled],
                    [int(action_ids[index]) for index in unhandled],
                )
            with self.profiler.stage("state/env_select_action_from_mjai"):
                for index, mjai in zip(unhandled, mjai_actions, strict=True):
                    decision = decisions[index]
                    action = decision.observation.select_action_from_mjai(mjai)
                    if action is None:
                        raise RuntimeError(
                            f"MJAI action was rejected: env={decision.env_index} "
                            f"seat={decision.seat_id} action_id={action_ids[index]} mjai={mjai}"
                        )
                    final[index] = action
        return final

    def prepare(
        self,
        decisions: list[Decision],
        walls: list[list[int]] | None = None,
    ) -> PreparedBatch:
        """装配 V18 当前局面输入：Shared+Analysis+Query 序列；Critic 私有行单独返回。

        动作解码与合法掩码仍由状态机负责（生命周期/动作执行），不再生成事件历史
        或 Atomic Snapshot 输入行。
        """
        if not decisions:
            raise ValueError("cannot prepare an empty decision batch")
        batch_indices = [decision.batch_index for decision in decisions]
        with self.profiler.stage("state/legal_action_json"):
            action_rows = [
                _action_objects_jsons_and_decision_flag(decision.observation)
                for decision in decisions
            ]
            legal_objects = [objects for objects, _actions, _flag in action_rows]
            legal_actions = [actions for _objects, actions, _flag in action_rows]
        with self.profiler.stage("state/rust_prepare_decisions"):
            # 只取合法掩码;V18 输入由 current_state.encode_batch 独立装配。
            prepared = self.state_machine.prepare_decisions(
                batch_indices, legal_actions,
            )
        with self.profiler.stage("state/rust_action_index_map"):
            mask = np.asarray(prepared, dtype=np.bool_)
            index_rows = self.state_machine.action_ids_with_source_indices(batch_indices)
            per_row_actions: list[list[tuple[Any, int]]] = []
            for row, mappings in enumerate(index_rows):
                actions_by_id: list[tuple[Any, int]] = []
                for action_id, source_index in mappings:
                    action_id = int(action_id)
                    source_index = int(source_index)
                    if not 0 <= source_index < len(legal_objects[row]):
                        raise RuntimeError(
                            f"state machine returned invalid legal action index {source_index}"
                        )
                    actions_by_id.append((legal_objects[row][source_index], action_id))
                expected = np.flatnonzero(mask[row]).tolist()
                if [action_id for _action, action_id in actions_by_id] != expected:
                    raise RuntimeError("state machine action-index mapping disagrees with legal mask")
                per_row_actions.append(actions_by_id)
        with self.profiler.stage("state/action_id_to_object_map"):
            for decision, actions_by_id in zip(decisions, per_row_actions, strict=True):
                self._action_by_id[decision.batch_index] = {
                    int(action_id): action for action, action_id in actions_by_id
                }
        with self.profiler.stage("state/current_state_assembly"):
            state_batch: EncodedStateBatch = encode_batch(
                [
                    (decision.observation, actions)
                    for decision, actions in zip(decisions, per_row_actions, strict=True)
                ]
            )
        with self.profiler.stage("state/critic_feature_encode"):
            if self.observations_by_env is None:
                critic_features = [empty_critic_features() for _decision in decisions]
            else:
                table_cache = {
                    env_index: collect_visible_table_state(
                        self.observations_by_env[env_index],
                        include_public_state=False,
                    )
                    for env_index in {decision.env_index for decision in decisions}
                }
                critic_features = [
                    encode_critic_features(
                        table_cache[decision.env_index],
                        decision.seat_id,
                        future_wall_tiles=(
                            walls[decision.env_index][:5]
                            if walls is not None
                            else ()
                        ),
                    )
                    for decision in decisions
                ]
            critic_factors, critic_lengths = pad_critic_feature_rows(critic_features)
        return PreparedBatch(
            actor_factors=np.asarray(state_batch.actor_factors, dtype=np.int32),
            actor_numeric=np.asarray(state_batch.actor_numeric, dtype=np.float32),
            actor_lengths=np.asarray(state_batch.actor_lengths, dtype=np.int64),
            query_rows=np.asarray(state_batch.query_rows, dtype=np.int32),
            query_action_ids=np.asarray(state_batch.action_ids, dtype=np.int32),
            query_pair_counts=np.asarray(state_batch.query_pair_counts, dtype=np.int64),
            legal_mask=np.asarray(state_batch.legal_mask, dtype=np.bool_),
            critic_factors=np.asarray(critic_factors, dtype=np.uint8),
            critic_lengths=np.asarray(critic_lengths, dtype=np.int64),
        )
