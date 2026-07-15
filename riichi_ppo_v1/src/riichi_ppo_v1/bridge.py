"""Strict conversion boundary between RiichiEnv, MJAI and the Rust state machine."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from .profiling import StageProfiler

NUM_PLAYERS = 4
NUM_ACTIONS = 241


@dataclass(frozen=True)
class Decision:
    env_index: int
    seat_id: int
    observation: Any

    @property
    def batch_index(self) -> int:
        return self.env_index * NUM_PLAYERS + self.seat_id


def tile_id_to_mjai(tile_id: int | None) -> str | None:
    """Convert RiichiEnv's physical 136-tile id to its MJAI spelling."""
    if tile_id is None:
        return None
    tile = int(tile_id)
    red = {16: "5mr", 52: "5pr", 88: "5sr"}
    if tile in red:
        return red[tile]
    if not 0 <= tile < 136:
        raise ValueError(f"invalid RiichiEnv tile id {tile}")
    suit = tile // 36
    if suit < 3:
        return f"{tile % 36 // 4 + 1}{('m', 'p', 's')[suit]}"
    honors = ("E", "S", "W", "N", "P", "F", "C")
    return honors[(tile - 108) // 4]


def action_jsons(observation: Any) -> list[str]:
    """Create exact action templates, including the physical tsumogiri distinction."""
    drawn = getattr(observation, "drawn_tile", None)
    result: list[str] = []
    for action in observation.legal_actions():
        value = json.loads(action.to_mjai())
        if value.get("type") == "dahai":
            # Action.to_mjai intentionally does not carry this environment-only bit.
            value["tsumogiri"] = action.tile is not None and drawn is not None and int(action.tile) == int(drawn)
        result.append(json.dumps(value, separators=(",", ":"), sort_keys=True))
    return result


def snapshot_json(observation: Any) -> str:
    pid = int(observation.player_id)
    hands = getattr(observation, "hands", None)
    if hands is None:
        raise RuntimeError("Observation must expose all hands for the V3 snapshot bridge")
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
        "last_discard": tile_id_to_mjai(getattr(observation, "last_discard", None)),
        "last_tedashis": [tile_id_to_mjai(x) for x in observation.last_tedashis],
    }
    return json.dumps(data, separators=(",", ":"))


class BatchedStateBridge:
    """One strict, vectorized boundary for all tables owned by a rollout worker."""

    def __init__(self, state_machine: Any, num_envs: int, profiler: StageProfiler | None = None) -> None:
        self.state_machine = state_machine
        self.num_envs = int(num_envs)
        self.profiler = profiler or StageProfiler(enabled=False)
        self.last_events: list[list[list[str]]] = [[[] for _ in range(NUM_PLAYERS)] for _ in range(num_envs)]

    def sync(self, observations_by_env: list[dict[int, Any]]) -> tuple[np.ndarray, np.ndarray]:
        if len(observations_by_env) != self.num_envs:
            raise RuntimeError(f"received {len(observations_by_env)} environments, expected {self.num_envs}")
        events_by_env: list[list[list[str]]] = []
        with self.profiler.stage("state/event_extract"):
            for env_index, observations in enumerate(observations_by_env):
                if set(observations) != set(range(NUM_PLAYERS)):
                    raise RuntimeError(f"env={env_index} must expose all four player observations")
                events_by_env.append([list(observations[seat].new_events()) for seat in range(NUM_PLAYERS)])
        self.last_events = events_by_env
        with self.profiler.stage("state/rust_event_apply"):
            end_kyoku, end_game = self.state_machine.apply_events_batch(list(range(self.num_envs)), events_by_env)
        with self.profiler.stage("state/boundary_array_convert"):
            return np.asarray(end_kyoku, dtype=np.bool_), np.asarray(end_game, dtype=np.bool_)

    def prepare(self, decisions: list[Decision]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not decisions:
            raise ValueError("cannot prepare an empty decision batch")
        batch_indices = [decision.batch_index for decision in decisions]
        with self.profiler.stage("state/legal_action_json"):
            legal_actions = [action_jsons(decision.observation) for decision in decisions]
        with self.profiler.stage("state/snapshot_json"):
            snapshots = [snapshot_json(decision.observation) for decision in decisions]
        with self.profiler.stage("state/rust_prepare_decisions"):
            ids, attention, lengths, mask, history_lengths, history_generations = self.state_machine.prepare_decisions(batch_indices, legal_actions, snapshots)
        with self.profiler.stage("state/numpy_array_convert"):
            ids_a = np.asarray(ids, dtype=np.int64)
            attention_a = np.asarray(attention, dtype=np.bool_)
            lengths_a = np.asarray(lengths, dtype=np.int64)
            mask_a = np.asarray(mask, dtype=np.bool_)
            history_lengths_a = np.asarray(history_lengths, dtype=np.int64)
            history_generations_a = np.asarray(history_generations, dtype=np.int64)
        if ids_a.ndim != 3 or ids_a.shape[0] != len(decisions) or ids_a.shape[2] != 8:
            raise RuntimeError(f"invalid batched state input shape {ids_a.shape}")
        if mask_a.shape != (len(decisions), NUM_ACTIONS) or not np.all(mask_a.any(axis=1)):
            raise RuntimeError("state machine returned an empty or malformed decision mask")
        if history_lengths_a.shape != (len(decisions),) or history_generations_a.shape != (len(decisions),):
            raise RuntimeError("state machine returned malformed cache metadata")
        if np.any(history_lengths_a < 0) or np.any(history_lengths_a >= lengths_a):
            raise RuntimeError("history length must be a strict prefix of the decision sequence")
        return ids_a, attention_a, lengths_a, mask_a, history_lengths_a, history_generations_a

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
