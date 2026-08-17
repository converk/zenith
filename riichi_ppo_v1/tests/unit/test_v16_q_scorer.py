"""Top-3 Q-boosting 候选集与动作表示 detach 的契约测试(US4)。"""

from __future__ import annotations

import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
from riichi_ppo_v1.model.architecture import dueling_candidate_q
from riichi_ppo_v1.training.learner import (
    boosted_top3_probabilities,
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


def test_q_scorer_action_to_pair_index_ignores_padded_action_ids() -> None:
    model = KyokuTransformerActorCritic(_v16_config())
    critic_hidden = torch.zeros(1, 32)
    # 有效 pair 是位置 0(action 0)与位置 1(action 5);位置 2 是编码 padding,
    # action_id 以 0 补齐。修复前 scatter 会把 padding 行写进 action 0 的列,
    # 使候选 0 错误地指向 padding 位置;高级索引赋值只写有效 pair。
    action_ids = torch.tensor([[0, 5, 0]])
    pair_counts = torch.tensor([2])
    candidates = torch.tensor([[0, 5]])

    def score_with_padding(padding_value: float) -> float:
        action_hiddens = torch.stack((
            torch.full((32,), 1.0),
            torch.full((32,), 2.0),
            torch.full((32,), padding_value),
        ))[None]
        scores = model.q_scores_v16(
            critic_hidden, action_hiddens, action_ids, pair_counts, candidates,
        )
        assert torch.isfinite(scores).all()
        return float(scores[0, 0])

    # 候选 0 必须始终映射到 pair 0;padding 行取值变化不得影响其得分。
    assert score_with_padding(-100.0) == score_with_padding(100.0)


def test_dueling_q_preserves_probability_weighted_value_identity() -> None:
    """Σ p_i·Q_i = V(s) 必须由构造恒成立。"""
    raw = torch.tensor([[1.0, 3.0, 2.0], [4.0, -1.0, 2.0]])
    probs = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
    value = torch.tensor([7.0, -3.0])
    advantages, q_values = dueling_candidate_q(raw, probs, value)
    normalized = probs / probs.sum(dim=-1, keepdim=True)
    expected_advantages = raw - (normalized * raw).sum(dim=-1, keepdim=True)
    torch.testing.assert_close(advantages, expected_advantages)
    torch.testing.assert_close(q_values, value[:, None] + advantages)
    torch.testing.assert_close(
        (normalized * q_values).sum(dim=-1), value,
    )
    torch.testing.assert_close(
        (normalized * advantages).sum(dim=-1), torch.zeros(2),
    )


def test_dueling_q_excludes_invalid_candidates_from_baseline() -> None:
    """无效候选(-inf)不参与重归一化,其 Q 保持 -inf。"""
    raw = torch.tensor([[1.0, float("-inf"), 3.0]])
    probs = torch.tensor([[0.4, 0.1, 0.5]])
    value = torch.tensor([2.0])
    _advantages, q_values = dueling_candidate_q(raw, probs, value)
    valid_probs = torch.tensor([[0.4, 0.5]]) / 0.9
    valid = q_values[:, 0] * valid_probs[0, 0] + q_values[:, 2] * valid_probs[0, 1]
    torch.testing.assert_close(valid, value)
    assert torch.isneginf(q_values[:, 1]).all()


def test_dueling_q_detaches_value_for_q_loss_path() -> None:
    """Q loss 不得向 value_head 回传梯度:Value 由独立 return loss 训练。"""
    raw = torch.randn(2, 3, requires_grad=True)
    probs = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    value = torch.randn(2, requires_grad=True)
    _advantages, q_values = dueling_candidate_q(
        raw, probs, value, detach_value=True,
    )
    q_values.sum().backward()
    assert value.grad is None
    assert raw.grad is not None


def test_boosted_top3_probabilities_reweights_within_fixed_mass() -> None:
    """p_boost 保持 Top-3 总质量不变,且向优势更大的候选倾斜。"""
    probs = torch.tensor([[0.1, 0.2, 0.3]])
    advantages = torch.tensor([[0.0, 1.0, -1.0]])
    boosted = boosted_top3_probabilities(
        probs, advantages, lambda_q=1.0, temperature=1.0,
    )
    torch.testing.assert_close(boosted.sum(dim=-1), probs.sum(dim=-1))
    # 相对质量应向优势更高的候选倾斜,低优势候选的相对质量下降。
    assert float(boosted[0, 1]) > float(boosted[0, 0])
    assert float(boosted[0, 1]) > float(boosted[0, 2])
    assert (
        float(boosted[0, 1] / boosted[0, 0])
        > float(probs[0, 1] / probs[0, 0])
    )
    assert (
        float(boosted[0, 2] / boosted[0, 0])
        < float(probs[0, 2] / probs[0, 0])
    )


def test_boosted_top3_probabilities_masks_invalid_slots() -> None:
    """padding/无效位置输出 0,不参与质量守恒。"""
    probs = torch.tensor([[0.5, 0.5, 0.0]])
    advantages = torch.tensor([[0.5, -0.5, float("-inf")]])
    boosted = boosted_top3_probabilities(
        probs, advantages, lambda_q=1.0, temperature=1.0,
    )
    torch.testing.assert_close(boosted[0, 2], torch.zeros(()))
    torch.testing.assert_close(boosted[0, :2].sum(), torch.tensor(1.0))
