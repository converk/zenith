"""V16 action-query 批量生成测试:与逐动作 oracle 逐元素一致。"""

from __future__ import annotations

import os

import numpy as np

# 测试不需要 GPU,只走 CPU 原生环境。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

import riichi
from riichienv import BatchedRiichiEnv

from riichi_ppo_v1.model.action_query import (
    analyze_action_queries,
    analyze_action_queries_batch,
)
from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision


def _small_config() -> dict:
    return {"game_mode": "4p-red-half"}


@pytest.fixture()
def rollout_state():
    config = _small_config()
    num = 2
    envs = BatchedRiichiEnv(num, seed=7, step_threads=1, game_mode=config["game_mode"])
    sm = riichi.MjaiKyokuStateMachineManager(num)
    bridge_old = BatchedStateBridge(sm, num, batch_query=False)
    bridge_new = BatchedStateBridge(sm, num, batch_query=True)
    obs = list(envs.reset())
    walls = list(envs.walls())
    bridge_old.sync(obs)
    bridge_new.sync(obs)
    return environment(envs, bridge_old, bridge_new, obs, walls, num)


class environment:
    def __init__(self, envs, bridge_old, bridge_new, obs, walls, num):
        self.envs = envs
        self.bridge_old = bridge_old
        self.bridge_new = bridge_new
        self.obs = obs
        self.walls = walls
        self.num = num

    def step(self):
        obs = self.obs
        actions = [{} for _ in range(self.num)]
        for ei, ob in enumerate(obs):
            for seat, o in ob.items():
                legal = list(o.legal_actions())
                if legal:
                    actions[ei][seat] = legal[0]
        self.obs = list(self.envs.step_batch(actions))
        self.walls = list(self.envs.walls())
        self.bridge_old.sync(self.obs)
        self.bridge_new.sync(self.obs)

    def decision_batch(self):
        decisions = []
        for ei, ob in enumerate(self.obs):
            for seat, o in ob.items():
                if o.legal_actions():
                    decisions.append(Decision(ei, seat, o))
        return decisions


def test_batch_query_matches_oracle_in_prepare(rollout_state) -> None:
    env = rollout_state
    mismatches = 0
    checked = 0
    for _ in range(15):
        decisions = env.decision_batch()
        if decisions:
            old = env.bridge_old.prepare_v16(decisions, walls=env.walls)
            new = env.bridge_new.prepare_v16(decisions, walls=env.walls)
            for key in (
                "history_factors", "history_numeric", "history_lengths",
                "snapshot_kinds", "snapshot_cat", "snapshot_num", "snapshot_lengths",
                "query_rows", "query_action_ids", "query_pair_counts", "legal_mask",
                "critic_factors", "critic_lengths",
            ):
                a = np.asarray(getattr(old, key))
                b = np.asarray(getattr(new, key))
                checked += 1
                if a.shape != b.shape or not np.array_equal(a, b):
                    mismatches += 1
        env.step()
    assert mismatches == 0
    assert checked > 0


def test_analyze_action_queries_batch_matches_per_action(rollout_state) -> None:
    env = rollout_state
    triples: list[tuple[object, object, int]] = []
    for _ in range(15):
        decisions = env.decision_batch()
        if decisions:
            prepared = env.bridge_old.prepare_v16(decisions, walls=env.walls)
            mask = np.asarray(prepared.legal_mask)
            for row, decision in enumerate(decisions):
                for action_id in np.flatnonzero(mask[row]):
                    legal = list(decision.observation.legal_actions())
                    triples.append((decision.observation, legal[0], int(action_id)))
        env.step()
    assert triples, "no legal actions collected"
    oracle = [analyze_action_queries(o, a, i) for o, a, i in triples]
    batched = analyze_action_queries_batch(triples)
    assert len(batched) == len(oracle)
    for left, right in zip(oracle, batched, strict=True):
        assert left[0] == right[0]
        assert left[1] == right[1]
