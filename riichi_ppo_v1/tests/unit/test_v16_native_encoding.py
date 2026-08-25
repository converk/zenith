"""V16 Rust 融合 Action Query 的真实环境回归。"""

from __future__ import annotations

import os

import numpy as np
import pytest

# 测试不需要 GPU,只走 CPU 原生环境。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import riichi
from riichienv import BatchedRiichiEnv, RiichiEnv
from riichienv.action import Action, ActionType

from riichi_ppo_v1.model.action_query import analyze_action_queries, encode_query_row
from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision
from riichi_ppo_v1.model.encoding_protocol import QUERY_ROW_ACTION_TYPE


@pytest.fixture()
def rollout_state():
    num_envs = 2
    envs = BatchedRiichiEnv(
        num_envs, seed=7, step_threads=1, game_mode="4p-red-half",
    )
    bridge = BatchedStateBridge(
        riichi.MjaiKyokuStateMachineManager(num_envs), num_envs,
    )
    observations = list(envs.reset())
    bridge.sync(observations)
    return envs, bridge, observations, list(envs.walls())


def _decisions(observations) -> list[Decision]:
    return [
        Decision(env_index, seat, observation)
        for env_index, table in enumerate(observations)
        for seat, observation in table.items()
        if observation.legal_actions()
    ]


def _step(envs, bridge, observations, tick: int):
    actions = [{} for _ in observations]
    for env_index, table in enumerate(observations):
        for seat, observation in table.items():
            legal = list(observation.legal_actions())
            if legal:
                actions[env_index][seat] = legal[(tick + env_index + seat) % len(legal)]
    observations = list(envs.step_batch(actions))
    bridge.sync(observations)
    return observations, list(envs.walls())


def test_native_query_rows_match_compatibility_api(rollout_state) -> None:
    envs, bridge, observations, walls = rollout_state
    checked = 0
    deduplicated = 0
    for tick in range(30):
        decisions = _decisions(observations)
        if decisions:
            prepared = bridge.prepare_v16(decisions, walls=walls)
            mappings = bridge.state_machine.action_ids_with_source_indices(
                [decision.batch_index for decision in decisions],
            )
            for row, (decision, action_mappings) in enumerate(
                zip(decisions, mappings, strict=True),
            ):
                legal = list(decision.observation.legal_actions())
                expected: list[np.ndarray] = []
                expected_ids: list[int] = []
                for action_id, source_index in action_mappings:
                    offense, defense = analyze_action_queries(
                        decision.observation,
                        legal[int(source_index)],
                        int(action_id),
                    )
                    expected.extend((encode_query_row(offense), encode_query_row(defense)))
                    expected_ids.append(int(action_id))
                count = len(expected_ids)
                np.testing.assert_array_equal(
                    prepared.query_rows[row, : 2 * count],
                    np.asarray(expected, dtype=np.int32),
                )
                np.testing.assert_array_equal(
                    prepared.query_action_ids[row, :count],
                    np.asarray(expected_ids, dtype=np.int32),
                )
                checked += count
            stats = bridge.last_v16_rust_stats
            deduplicated += max(0, stats["actions"] - stats["unique_offense_rows"])
        observations, walls = _step(envs, bridge, observations, tick)
    assert checked > 0
    assert deduplicated > 0


def test_native_bridge_has_no_runtime_fallback_switches() -> None:
    bridge = BatchedStateBridge(riichi.MjaiKyokuStateMachineManager(1), 1)
    assert not hasattr(bridge, "batch_query")
    assert not hasattr(bridge, "rust_encoding")


def test_kyushu_kyuhai_compatibility_preserves_ryukyoku_code() -> None:
    env = RiichiEnv(seed=0, game_mode="4p-red-half")
    env.reset()
    observation = env.get_observation(0)
    offense, defense = analyze_action_queries(
        observation, Action(ActionType.KYUSHU_KYUHAI), 240,
    )
    rows = np.stack((encode_query_row(offense), encode_query_row(defense)))
    assert offense.action_type == defense.action_type == "ryukyoku"
    np.testing.assert_array_equal(rows[:, QUERY_ROW_ACTION_TYPE], [10, 10])
