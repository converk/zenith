from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import riichi
from riichienv import BatchedRiichiEnv

from riichi_ppo_v1.model.bridge import BatchedStateBridge
from riichi_ppo_v1.sft.policy_adapter import load_policy_adapter
from riichi_ppo_v1.training.rewards import (
    DecisionAnalysisBatch, EfficiencyAnalyzer, PublicStateTracker,
)
from riichi_ppo_v1.training.worker import active_decisions


ROOT = Path(__file__).parents[3]
V11_CHECKPOINT = ROOT / "checkpoints/train_riichi_v11_sft_40pct/best.pt"


@pytest.mark.skipif(not V11_CHECKPOINT.is_file(), reason="v11 compatibility checkpoint unavailable")
def test_v11_frozen_feature_logits_action_and_environment_acceptance() -> None:
    envs = BatchedRiichiEnv(1, seed=20260802, step_threads=1, game_mode="4p-red-half")
    bridge = BatchedStateBridge(riichi.MjaiKyokuStateMachineManager(1), 1)
    observations = list(envs.reset())
    bridge.sync(observations)
    public = PublicStateTracker(1)
    public.update(bridge.last_events)
    decisions = active_decisions(observations, {0})
    analysis = DecisionAnalysisBatch.build(
        decisions, analyzer=EfficiencyAnalyzer(), public=public,
    )
    adapter = load_policy_adapter(V11_CHECKPOINT, device="cpu")
    prepared = adapter.prepare(bridge, decisions, analysis)
    logits = adapter.masked_logits(prepared)

    assert prepared.lengths.tolist() == [43]
    assert prepared.legal.sum(axis=1).tolist() == [14]
    np.testing.assert_array_equal(prepared.factors[0, :8], np.asarray([
        [1, 1, 2, 0, 0, 0, 0, 0, 1, 1],
        [2, 2, 1, 1, 0, 0, 0, 0, 0, 0],
        [2, 2, 1, 2, 0, 0, 0, 0, 0, 0],
        [2, 2, 1, 3, 0, 0, 0, 0, 0, 0],
        [2, 2, 1, 4, 0, 0, 0, 0, 0, 0],
        [2, 3, 1, 0, 0, 0, 0, 0, 0, 0],
        [2, 3, 2, 0, 0, 0, 0, 0, 0, 0],
        [2, 3, 3, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8))
    top = torch.topk(logits[0], 5)
    assert top.indices.tolist() == [58, 59, 33, 65, 67]
    torch.testing.assert_close(
        top.values,
        torch.tensor([6.3799534, 4.956987, 1.000553, 0.4798334, 0.03239475]),
        rtol=1e-5, atol=1e-5,
    )
    action_ids = logits.argmax(-1).tolist()
    assert action_ids == [58]
    actions = bridge.decode(decisions, action_ids)
    # A rejected decode raises before this point; stepping is the environment oracle.
    next_observations = envs.step_batch([{decisions[0].seat_id: actions[0]}])
    assert len(next_observations) == 1
