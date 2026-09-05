"""V19 Actor/Critic 信息边界测试。"""

from __future__ import annotations

import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.model.semantic_validation import assert_critic_token_semantics
from riichi_ppo_v1.tests.v19_fixtures import actor_inputs, critic_inputs


def _forward(model, inputs, critic=None):
    kwargs = {
        "actor_factors": inputs["actor_factors"],
        "actor_numeric": inputs["actor_numeric"],
        "actor_lengths": inputs["actor_lengths"],
        "query_action_ids": inputs["action_ids"],
        "query_pair_counts": inputs["query_pair_counts"],
        "legal_mask": inputs["legal_mask"],
    }
    if critic is not None:
        kwargs["critic_factors"] = critic["critic_factors"]
        kwargs["critic_lengths"] = critic["critic_lengths"]
    return model(**kwargs, policy_only=(critic is None))


def test_actor_ignores_critic_private_input() -> None:
    model = KyokuTransformerActorCritic()
    inputs = actor_inputs(batch=2, action_ids=(1, 7, 12))
    without = _forward(model, inputs)
    # 仅添加 critic 输入（同时复用同一 actor 输入）不会改变 actor logits。
    critic = critic_inputs(batch=2)
    with_private = _forward(model, inputs, critic)
    assert torch.allclose(
        without["raw_policy_logits"], with_private["raw_policy_logits"], atol=1e-6, rtol=1e-6
    )


def test_critic_value_changes_with_private_input() -> None:
    model = KyokuTransformerActorCritic()
    inputs = actor_inputs(batch=2, action_ids=(1, 7))
    critic = critic_inputs(batch=2)
    output = _forward(model, inputs, critic)
    # 修改三家闭手的 count。
    changed = critic["critic_factors"].clone()
    changed[0, 1, 5] = 2
    output_changed = _forward(model, inputs, {"critic_factors": changed, "critic_lengths": critic["critic_lengths"]})
    assert output["value"].shape == (2,)
    assert not torch.allclose(output["value"], output_changed["value"])


def test_critic_rows_exclude_analysis_and_action() -> None:
    critic = critic_inputs(batch=2)
    assert_critic_token_semantics(critic["critic_factors"], critic["critic_lengths"])
    segments = critic["critic_factors"][:, :, 0].unique().tolist()
    assert set(segments) <= {4}
