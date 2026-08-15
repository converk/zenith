"""V16 输入编码装配(bridge)的集成测试。

覆盖:每唯一 action id 恰一对 query、全部 answer 落在声明基数内、Critic 特权
输入只含对手手牌与后 5 牌山(不含公开汇总)、装配结果可直接进入 v16 前向。
"""

from __future__ import annotations

import unittest

try:
    import riichi
    from riichienv import BatchedRiichiEnv
except ImportError:  # pragma: no cover - 扩展未安装时跳过
    riichi = None
    BatchedRiichiEnv = None

import numpy as np
import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision
from riichi_ppo_v1.model.critic_features import (
    SEGMENT_CRITIC_FUTURE_WALL,
    SEGMENT_CRITIC_PRIVATE,
)
from riichi_ppo_v1.model.encoding_protocol import (
    DEFENSE_SLOT_ORDER,
    OFFENSE_SLOT_ORDER,
    QUERY_ROW_ACTION_ID,
    QUERY_ROW_ANSWER_START,
    QUERY_ROW_QUERY_TYPE,
    QUERY_DEFENSE,
    QUERY_OFFENSE,
    SLOT_CARDINALITIES,
)


@unittest.skipUnless(riichi is not None and BatchedRiichiEnv is not None, "local RiichiEnv extensions are not installed")
class V16EncodingBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = BatchedRiichiEnv(1, seed=11, step_threads=1)
        self.manager = riichi.MjaiKyokuStateMachineManager(1)
        self.bridge = BatchedStateBridge(self.manager, 1)

    def _decisions(self) -> list[Decision]:
        observations = list(self.env.reset())
        self.bridge.sync(observations)
        return [
            Decision(0, seat, observation)
            for seat, observation in observations[0].items()
            if observation.legal_actions()
        ]

    def test_prepare_v16_encodes_one_query_pair_per_unique_action(self) -> None:
        decisions = self._decisions()
        batch = self.bridge.prepare_v16(decisions, walls=list(self.env.walls()))
        self.assertEqual(batch.legal_mask.shape, (len(decisions), 241))
        for row in range(len(decisions)):
            ids = np.flatnonzero(batch.legal_mask[row]).tolist()
            self.assertEqual(list(batch.query_action_ids[row, : len(ids)]), ids)
            self.assertEqual(int(batch.query_pair_counts[row]), len(ids))
            rows = batch.query_rows[row, : 2 * len(ids)]
            self.assertTrue(np.all(rows[0::2, QUERY_ROW_QUERY_TYPE] == QUERY_OFFENSE))
            self.assertTrue(np.all(rows[1::2, QUERY_ROW_QUERY_TYPE] == QUERY_DEFENSE))
            self.assertTrue(np.all(rows[0::2, QUERY_ROW_ACTION_ID] == ids))
            self.assertTrue(np.all(rows[1::2, QUERY_ROW_ACTION_ID] == ids))

    def test_query_answers_stay_within_declared_cardinality(self) -> None:
        decisions = self._decisions()
        batch = self.bridge.prepare_v16(decisions, walls=None)
        for row in range(len(decisions)):
            pairs = int(batch.query_pair_counts[row])
            rows = batch.query_rows[row, : 2 * pairs]
            offense = rows[0::2, QUERY_ROW_ANSWER_START:]
            defense = rows[1::2, QUERY_ROW_ANSWER_START:]
            for index, slot in enumerate(OFFENSE_SLOT_ORDER):
                self.assertTrue(np.all(offense[:, index] < SLOT_CARDINALITIES[slot]))
            for index, slot in enumerate(DEFENSE_SLOT_ORDER):
                self.assertTrue(np.all(defense[:, index] < SLOT_CARDINALITIES[slot]))

    def test_critic_private_contains_only_hands_and_future_wall(self) -> None:
        decisions = self._decisions()
        batch = self.bridge.prepare_v16(decisions, walls=list(self.env.walls()))
        for row in range(len(decisions)):
            length = int(batch.critic_lengths[row])
            segments = set(np.unique(batch.critic_factors[row, :length, 0]).tolist())
            self.assertLessEqual(
                segments, {SEGMENT_CRITIC_PRIVATE, SEGMENT_CRITIC_FUTURE_WALL},
            )

    def test_bridge_batch_drives_v16_forward(self) -> None:
        decisions = self._decisions()
        batch = self.bridge.prepare_v16(decisions, walls=list(self.env.walls()))
        model = KyokuTransformerActorCritic(ModelConfig.preset("v16")).eval()
        with torch.no_grad():
            output = model.forward_v16(
                torch.as_tensor(batch.history_factors),
                torch.as_tensor(batch.history_numeric),
                torch.as_tensor(batch.history_lengths),
                torch.as_tensor(batch.snapshot_kinds),
                torch.as_tensor(batch.snapshot_cat),
                torch.as_tensor(batch.snapshot_num),
                torch.as_tensor(batch.snapshot_lengths),
                torch.as_tensor(batch.query_rows),
                torch.as_tensor(batch.query_action_ids),
                torch.as_tensor(batch.query_pair_counts),
                torch.as_tensor(batch.legal_mask),
                critic_factors=torch.as_tensor(batch.critic_factors),
                critic_lengths=torch.as_tensor(batch.critic_lengths),
            )
        legal = torch.as_tensor(batch.legal_mask)
        self.assertTrue(torch.isfinite(output["policy_logits"][legal]).all())
        self.assertTrue(torch.isneginf(output["policy_logits"][~legal]).all())
        self.assertEqual(output["value"].shape, (len(decisions),))
