"""Strict conversion boundary between RiichiEnv, MJAI and the Rust state machine."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from typing import Any

import numpy as np

from .critic_features import (
    collect_visible_table_state,
    empty_critic_features,
    encode_critic_features,
    encode_public_summary,
    pad_critic_feature_rows,
)
from ..training.profiling import StageProfiler

NUM_PLAYERS = 4
NUM_ACTIONS = 241
_DECISION_ACTION_TYPES = frozenset({"dahai", "reach", "ankan", "kakan", "ryukyoku"})


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


def _action_jsons_and_decision_flag(observation: Any) -> tuple[list[str], int]:
    """Create exact templates and the snapshot's action-window flag together."""
    drawn = getattr(observation, "drawn_tile", None)
    result: list[str] = []
    has_decision_action = False
    for action in observation.legal_actions():
        tsumogiri = action.tile is not None and drawn is not None and int(action.tile) == int(drawn)
        action_json, action_type = _normalized_action_json(action.to_mjai(), tsumogiri)
        result.append(action_json)
        has_decision_action |= action_type in _DECISION_ACTION_TYPES
    return result, int(has_decision_action)


def action_jsons(observation: Any) -> list[str]:
    """Create exact action templates, including the physical tsumogiri distinction."""
    return _action_jsons_and_decision_flag(observation)[0]


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

    def prepare(self, decisions: list[Decision], analysis: Any | None = None) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        if not decisions:
            raise ValueError("cannot prepare an empty decision batch")
        batch_indices = [decision.batch_index for decision in decisions]
        with self.profiler.stage("state/legal_action_json"):
            action_rows = [_action_jsons_and_decision_flag(decision.observation) for decision in decisions]
            legal_actions = [actions for actions, _flag in action_rows]
        with self.profiler.stage("state/snapshot_json"):
            snapshots = [
                snapshot_json(decision.observation, decision_flag)
                for decision, (_actions, decision_flag) in zip(decisions, action_rows, strict=True)
            ]
        with self.profiler.stage("state/rust_prepare_decisions"):
            factors, numeric, token_lengths, mask, history_generations = self.state_machine.prepare_decisions(batch_indices, legal_actions, snapshots)
        with self.profiler.stage("state/critic_feature_encode"):
            if self.observations_by_env is None:
                critic_features = [empty_critic_features() for _decision in decisions]
                public_features = [empty_critic_features() for _decision in decisions]
            else:
                table_cache = {
                    env_index: collect_visible_table_state(
                        self.observations_by_env[env_index],
                        include_public_state=True,
                    )
                    for env_index in {decision.env_index for decision in decisions}
                }
                critic_features = [
                    encode_critic_features(
                        table_cache[decision.env_index],
                        decision.seat_id,
                    )
                    for decision in decisions
                ]
                public_features = [
                    encode_public_summary(table_cache[decision.env_index], decision.seat_id)
                    for decision in decisions
                ]
            critic_factors, critic_lengths = pad_critic_feature_rows(critic_features)
        with self.profiler.stage("state/numpy_array_convert"):
            factors_a = np.asarray(factors, dtype=np.uint8)
            numeric_a = np.asarray(numeric, dtype=np.float32)
            token_lengths_a = np.asarray(token_lengths, dtype=np.int64)
            mask_a = np.asarray(mask, dtype=np.bool_)
            history_generations_a = np.asarray(history_generations, dtype=np.int64)
            critic_factors_a = np.asarray(critic_factors, dtype=np.uint8)
            critic_lengths_a = np.asarray(critic_lengths, dtype=np.int64)
        if analysis is not None:
            with self.profiler.stage("state/candidate_feature_encode"):
                candidate_factors, candidate_numeric = analysis.candidate_tokens(decisions, mask_a)
                new_lengths = token_lengths_a + np.asarray(
                    [len(row) for row in candidate_factors], dtype=np.int64,
                )
                # The model appends one learned query token.
                if np.any(new_lengths + 1 > 4096):
                    raise RuntimeError(
                        f"candidate tokens overflow context: max={int(new_lengths.max()) + 1} limit=4096"
                    )
                width = int(new_lengths.max())
                extended_factors = np.zeros((len(decisions), width, 10), dtype=np.uint8)
                extended_numeric = np.zeros((len(decisions), width, 8), dtype=np.float32)
                for row, (extra_factors, extra_numeric) in enumerate(
                    zip(candidate_factors, candidate_numeric, strict=True)
                ):
                    base = int(token_lengths_a[row])
                    extended_factors[row, :base] = factors_a[row, :base]
                    extended_numeric[row, :base] = numeric_a[row, :base]
                    extended_factors[row, base : base + len(extra_factors)] = extra_factors
                    extended_numeric[row, base : base + len(extra_numeric)] = extra_numeric
                factors_a = extended_factors
                numeric_a = extended_numeric
                token_lengths_a = new_lengths
        with self.profiler.stage("state/public_summary_append"):
            new_lengths = token_lengths_a + np.asarray(
                [feature.length for feature in public_features], dtype=np.int64,
            )
            if np.any(new_lengths + 1 > 4096):
                raise RuntimeError(
                    f"public-summary tokens overflow context: max={int(new_lengths.max()) + 1} limit=4096"
                )
            width = int(new_lengths.max())
            extended_factors = np.zeros((len(decisions), width, 10), dtype=np.uint8)
            extended_numeric = np.zeros((len(decisions), width, 8), dtype=np.float32)
            for row, feature in enumerate(public_features):
                base = int(token_lengths_a[row])
                extended_factors[row, :base] = factors_a[row, :base]
                extended_numeric[row, :base] = numeric_a[row, :base]
                if feature.length:
                    extended_factors[row, base : base + feature.length] = feature.factors
            factors_a = extended_factors
            numeric_a = extended_numeric
            token_lengths_a = new_lengths
        if factors_a.ndim != 3 or factors_a.shape[0] != len(decisions) or factors_a.shape[2] != 10:
            raise RuntimeError(f"invalid token factor shape {factors_a.shape}")
        if numeric_a.shape != (*factors_a.shape[:2], 8):
            raise RuntimeError(f"invalid token numeric shape {numeric_a.shape}")
        if mask_a.shape != (len(decisions), NUM_ACTIONS) or not np.all(mask_a.any(axis=1)):
            raise RuntimeError("state machine returned an empty or malformed decision mask")
        if token_lengths_a.shape != (len(decisions),) or history_generations_a.shape != (len(decisions),):
            raise RuntimeError("state machine returned malformed cache metadata")
        if np.any(token_lengths_a < 0) or np.any(token_lengths_a > factors_a.shape[1]):
            raise RuntimeError("invalid token length")
        return (
            factors_a,
            numeric_a,
            token_lengths_a,
            mask_a,
            history_generations_a,
            critic_factors_a,
            critic_lengths_a,
        )

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
