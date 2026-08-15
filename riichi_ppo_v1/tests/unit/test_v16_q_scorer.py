"""Top-3 Q-boosting 候选集与动作表示 detach 的契约测试(US4)。"""

from __future__ import annotations

import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
from riichi_ppo_v1.training.learner import (
    candidate_q_loss,
    select_top3_candidates,
)


def _v16_config() -> ModelConfig:
    return ModelConfig(
        layers=5, shared_layers=4, critic_layers=2, d_model=32,
        query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64,
        context_tokens=256, policy_head_type="symmetric_action_query",
    )


def test_training_candidates_are_top3_union_behavior_within_four() -> None:
    logits = torch.tensor([[0.1, 0.9, 0.8, 0.7, 0.6, 0.5]])
    legal = torch.tensor([[True, True, True, True, True, False]])
    behavior = torch.tensor([4])
    boost, training = select_top3_candidates(logits, legal, behavior)
    assert boost.tolist() == [[1, 2, 3]]
    assert sorted(training[training >= 0].tolist()) == [1, 2, 3, 4]
    # 行为动作已在 Top-3 内时去重,仍不超过 4 个候选。
    behavior = torch.tensor([2])
    boost, training = select_top3_candidates(logits, legal, behavior)
    assert sorted(training[training >= 0].tolist()) == [1, 2, 3]
    assert int((training >= 0).sum()) <= 4


def test_q_scorer_detaches_action_hidden_from_actor() -> None:
    model = KyokuTransformerActorCritic(_v16_config())
    critic_hidden = torch.randn(2, 32, requires_grad=True)
    action_hiddens = torch.randn(2, 3, 32)
    action_ids = torch.tensor([[0, 5, 9], [1, 6, 10]])
    pair_counts = torch.tensor([3, 3])
    candidates = torch.tensor([[0, 9], [1, 10]])
    scores = model.q_scores_v16(
        critic_hidden, action_hiddens, action_ids, pair_counts, candidates,
    )
    assert scores.shape == (2, 2)
    assert torch.isfinite(scores).all()
    loss = candidate_q_loss(scores, torch.zeros_like(scores), torch.ones_like(scores, dtype=torch.bool))
    loss.backward()
    # Q loss 不得经动作表示直接更新 Actor 分支。
    assert model.action_fusion[0].weight.grad is None
    assert model.policy_mlp[1].weight.grad is None
    # 但可以更新 critic 侧的 z_critic。
    assert critic_hidden.grad is not None
    assert model.q_scorer[0].weight.grad is not None


def test_q_scorer_shape_is_512_to_256_to_one() -> None:
    model = KyokuTransformerActorCritic(ModelConfig.preset("v16"))
    first = model.q_scorer[0]
    assert isinstance(first, torch.nn.Linear)
    assert first.in_features == 2 * model.config.d_model
    assert first.out_features == model.config.d_model
    assert model.q_scorer[-1].out_features == 1
