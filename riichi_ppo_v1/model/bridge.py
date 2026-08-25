"""RiichiEnv、MJAI 与 Rust 状态机之间的严格转换边界。"""

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
from .schema import NUM_ACTIONS, TID_COUNT
from .snapshot import encode_snapshot_batch
from .native_encoding import encode_action_queries_batch_native

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
    """现行 V18 输入编码的批量化装配结果。

    目标序列 = Objective Facts(history)+ Compact Snapshot + 每动作一对
    Offense/Defense Query;Critic 特权输入(三家对手手牌 + 后 5 牌山)单独保存,
    不含公开牌河/副露重复表示。
    """

    history_factors: np.ndarray
    history_numeric: np.ndarray
    history_lengths: np.ndarray
    snapshot_factors: np.ndarray
    snapshot_numeric: np.ndarray
    snapshot_lengths: np.ndarray
    query_rows: np.ndarray
    query_action_ids: np.ndarray
    query_pair_counts: np.ndarray
    legal_mask: np.ndarray
    history_generations: np.ndarray
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
    """Return the canonical MJAI template and its type for one action shape.

    RiichiEnv repeatedly exposes the same small action vocabulary across tables.
    Caching this pure JSON normalization avoids both a repeated parse/dump and a
    second parse later when constructing ``decision_flags``.
    """
    value = json.loads(raw_action)
    action_type = str(value.get("type", ""))
    if action_type == "dahai":
        # Action.to_mjai intentionally does not carry this environment-only bit.
        value["tsumogiri"] = bool(tsumogiri)
    expected_consumed = {"chi": 2, "pon": 2, "daiminkan": 3}.get(action_type)
    consumed = value.get("consumed")
    if expected_consumed is not None and isinstance(consumed, list) and len(consumed) == expected_consumed + 1:
        # Offline replay actions contain the called tile in consume_tiles,
        # whereas canonical MJAI keeps it only in ``pai``.
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


def snapshot_json(observation: Any, decision_flags: int = 0) -> str:
    pid = int(observation.player_id)
    hands = getattr(observation, "hands", None)
    if hands is None:
        raise RuntimeError("Observation must expose all hands for the state bridge")
    data = {
        "player_id": pid,
        "oya": int(observation.oya),
        "round_wind": int(observation.round_wind),
        "kyoku_index": int(observation.kyoku_index),
        "honba": int(observation.honba),
        "riichi_sticks": int(observation.riichi_sticks),
        "scores": [int(x) for x in observation.scores],
        "dora_indicators": [tile_id_to_mjai(x) for x in observation.dora_indicators],
        "hand": [tile_id_to_mjai(x) for x in hands[pid]],
        "drawn_tile": tile_id_to_mjai(getattr(observation, "drawn_tile", None)),
        "riichi_declared": [bool(x) for x in observation.riichi_declared],
        "decision_flags": int(decision_flags),
    }
    return json.dumps(data, separators=(",", ":"))


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
        with self.profiler.stage("state/rust_decode_actions"):
            mjai_actions = self.state_machine.decode_actions(
                [decision.batch_index for decision in decisions], [int(action_id) for action_id in action_ids]
            )
        result: list[Any] = []
        with self.profiler.stage("state/env_select_action_from_mjai"):
            for decision, action_id, mjai in zip(decisions, action_ids, mjai_actions):
                action = decision.observation.select_action_from_mjai(mjai)
                if action is None:
                    raise RuntimeError(f"MJAI action was rejected: env={decision.env_index} seat={decision.seat_id} action_id={action_id} mjai={mjai}")
                result.append(action)
        return result

    def prepare(
        self,
        decisions: list[Decision],
        walls: list[list[int]] | None = None,
    ) -> PreparedBatch:
        """装配 V18 输入:Objective Facts + Atomic Snapshot + 每动作一对 Query。

        每个唯一 action id(合法掩码内)生成一对 query,重复物理动作按同一
        action id 取代表动作;Critic 只保留三家对手手牌与后 5 张牌山。
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
        with self.profiler.stage("state/snapshot_json"):
            snapshots = [
                snapshot_json(decision.observation, decision_flag)
                for decision, (_objects, _actions, decision_flag) in zip(
                    decisions, action_rows, strict=True,
                )
            ]
        with self.profiler.stage("state/rust_prepare_decisions"):
            prepared = self.state_machine.prepare_decisions(
                batch_indices, legal_actions, snapshots,
            )
        with self.profiler.stage("state/rust_action_index_map"):
            mask = np.asarray(prepared[3], dtype=np.bool_)
            index_rows = self.state_machine.action_ids_with_source_indices(batch_indices)
            ids_by_row: list[list[int]] = []
            per_row_actions: list[dict[int, Any]] = []
            for row, mappings in enumerate(index_rows):
                ids: list[int] = []
                actions_by_id: dict[int, Any] = {}
                for action_id, source_index in mappings:
                    action_id = int(action_id)
                    source_index = int(source_index)
                    if not 0 <= source_index < len(legal_objects[row]):
                        raise RuntimeError(
                            f"state machine returned invalid legal action index {source_index}"
                        )
                    ids.append(action_id)
                    actions_by_id[action_id] = legal_objects[row][source_index]
                expected = np.flatnonzero(mask[row]).tolist()
                if ids != expected:
                    raise RuntimeError("state machine action-index mapping disagrees with legal mask")
                ids_by_row.append(ids)
                per_row_actions.append(actions_by_id)
        with self.profiler.stage("state/query_assembly"):
            query_rows: list[np.ndarray] = []
            action_id_rows: list[np.ndarray] = []
            # 生产路径只保留 Rust 融合编码,不再携带 Python 逐动作或 batch 回退。
            triples: list[tuple[Any, Any, int]] = []
            row_offsets: list[int] = [0]
            for row, decision in enumerate(decisions):
                for action_id in ids_by_row[row]:
                    triples.append(
                        (decision.observation, per_row_actions[row][action_id], int(action_id))
                    )
                row_offsets.append(len(triples))
            encoded = encode_action_queries_batch_native(triples)
            self.last_rust_stats = {
                "actions": len(triples),
                "unique_offense_rows": encoded.unique_offense_rows,
                "unique_shanten_rows": encoded.unique_shanten_rows,
            }
            for row, _decision in enumerate(decisions):
                start, end = row_offsets[row], row_offsets[row + 1]
                value = encoded.query_rows[start:end]
                query_rows.append(value.reshape(-1, value.shape[-1]))
                action_id_rows.append(np.asarray(ids_by_row[row], dtype=np.int32))
            snapshot_factors, snapshot_numeric, snapshot_lengths = encode_snapshot_batch(
                [decision.observation for decision in decisions]
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
        with self.profiler.stage("state/numpy_array_convert"):
            history_factors = np.asarray(prepared[0], dtype=np.uint8)
            history_numeric = np.asarray(prepared[1], dtype=np.float32)
            history_lengths = np.asarray(prepared[2], dtype=np.int64)
            history_generations = np.asarray(prepared[4], dtype=np.int64)
            mask_a = np.asarray(mask, dtype=np.bool_)
            if mask_a.shape != (len(decisions), NUM_ACTIONS) or not np.all(mask_a.any(axis=1)):
                raise RuntimeError("state machine returned an empty or malformed decision mask")
        with self.profiler.stage("state/input_padding"):
            batch = len(decisions)
            history_width = int(history_lengths.max())
            action_capacity = max(len(ids) for ids in action_id_rows)
            if action_capacity < 1:
                raise RuntimeError("V18 batch requires at least one legal action per row")
            query_array = np.zeros((batch, 2 * action_capacity, 15), dtype=np.int32)
            action_ids_array = np.zeros((batch, action_capacity), dtype=np.int32)
            pair_counts = np.zeros(batch, dtype=np.int64)
            for row, (rows_value, ids_value) in enumerate(zip(query_rows, action_id_rows, strict=True)):
                count = len(ids_value)
                if rows_value.shape[0] != 2 * count:
                    raise RuntimeError("query rows must be one offense/defense pair per action")
                query_array[row, : rows_value.shape[0]] = rows_value
                action_ids_array[row, :count] = ids_value
                pair_counts[row] = count
            if history_factors.shape[1] < history_width:
                raise RuntimeError("prepared history factors are shorter than declared lengths")
            history_factors = history_factors[:, :history_width]
            history_numeric = history_numeric[:, :history_width]
        return PreparedBatch(
            history_factors=history_factors,
            history_numeric=history_numeric,
            history_lengths=history_lengths,
            snapshot_factors=snapshot_factors,
            snapshot_numeric=snapshot_numeric,
            snapshot_lengths=snapshot_lengths,
            query_rows=query_array,
            query_action_ids=action_ids_array,
            query_pair_counts=pair_counts,
            legal_mask=mask_a,
            history_generations=history_generations,
            critic_factors=np.asarray(critic_factors, dtype=np.uint8),
            critic_lengths=np.asarray(critic_lengths, dtype=np.int64),
        )
