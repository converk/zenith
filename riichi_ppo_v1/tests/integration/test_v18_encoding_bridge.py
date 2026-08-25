"""真实 RiichiEnv 到 V18 bridge/model 的集成测试。"""

import numpy as np
import torch
import riichi
from riichienv import BatchedRiichiEnv

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision
from riichi_ppo_v1.model.encoding_protocol import SNAPSHOT_FIELD_COUNT
from riichi_ppo_v1.model.semantic_validation import (
    assert_actor_input_semantics,
    assert_critic_token_semantics,
)


def test_live_bridge_shapes_semantics_and_full_forward() -> None:
    env = BatchedRiichiEnv(1, seed=11, step_threads=1)
    observations = list(env.reset())
    bridge = BatchedStateBridge(riichi.MjaiKyokuStateMachineManager(1), 1)
    bridge.sync(observations)
    decisions = [
        Decision(0, seat, observation)
        for seat, observation in observations[0].items()
        if observation.legal_actions()
    ]
    batch = bridge.prepare(decisions, walls=list(env.walls()))
    assert_actor_input_semantics(
        batch.history_factors, batch.history_numeric, batch.history_lengths,
        batch.snapshot_factors, batch.snapshot_numeric, batch.snapshot_lengths,
        batch.query_rows, batch.query_action_ids, batch.query_pair_counts,
        batch.legal_mask,
    )
    assert_critic_token_semantics(batch.critic_factors, batch.critic_lengths)
    with torch.no_grad():
        output = KyokuTransformerActorCritic().eval()(
            torch.as_tensor(batch.history_factors),
            torch.as_tensor(batch.history_numeric),
            torch.as_tensor(batch.history_lengths),
            torch.as_tensor(batch.snapshot_factors),
            torch.as_tensor(batch.snapshot_numeric),
            torch.as_tensor(batch.snapshot_lengths),
            torch.as_tensor(batch.query_rows),
            torch.as_tensor(batch.query_action_ids),
            torch.as_tensor(batch.query_pair_counts),
            torch.as_tensor(batch.legal_mask),
            critic_factors=torch.as_tensor(batch.critic_factors),
            critic_lengths=torch.as_tensor(batch.critic_lengths),
        )
    legal = torch.as_tensor(batch.legal_mask)
    assert torch.isfinite(output["policy_logits"][legal]).all()
    assert torch.isneginf(output["policy_logits"][~legal]).all()
    assert SNAPSHOT_FIELD_COUNT == 54
    assert np.all(batch.snapshot_lengths == SNAPSHOT_FIELD_COUNT)
