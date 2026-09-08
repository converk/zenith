"""V19 P1 信念诊断单测：梯度余弦助手与反事实 detach 覆写。

覆盖 2026-09-08「信念注入残差化与 token 精简」实施方案新增的两个 P1：
- ``flatten_grads_cosine``：两组逐参数梯度（允许 None）的余弦/范数纯测量；
- ``is_belief_private_parameter``：信念私有参数判定（不含 token_matrix）；
- ``belief_summary_detach=False`` 反事实前向：策略梯度可达信念私有头
  （默认 True 时不可达，隔离契约由 test_v19_belief_gradient_isolation 锁定）。
"""

from __future__ import annotations

import pytest
import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.tests.v19_fixtures import actor_inputs
from riichi_ppo_v1.training.belief import flatten_grads_cosine, is_belief_private_parameter


def test_is_belief_private_parameter() -> None:
    """四头/backbone/query 属私有，token_matrix 与主干/接口不属于。"""
    assert is_belief_private_parameter("belief_network.hand_head.weight")
    assert is_belief_private_parameter("belief_backbone.layers.0.weight")
    assert is_belief_private_parameter("belief_query")
    assert not is_belief_private_parameter("belief_network.token_matrix.weight")
    assert not is_belief_private_parameter("public_backbone.layers.0.weight")
    assert not is_belief_private_parameter("actor_backbone.layers.0.weight")
    assert not is_belief_private_parameter("belief_readout.proj.weight")


def test_flatten_grads_cosine_known_directions() -> None:
    """同向 → +1，反向 → -1，正交分量 → 已知余弦；范数为各自 L2。"""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 4.0, 6.0])
    c = torch.tensor([-1.0, -2.0, -3.0])
    d = torch.tensor([1.0, 0.0, 0.0])

    cosine, norm_a, norm_b = flatten_grads_cosine([a], [b])
    assert cosine == pytest.approx(1.0)
    assert norm_a == pytest.approx(float(a.norm()))
    assert norm_b == pytest.approx(float(b.norm()))

    cosine, _, _ = flatten_grads_cosine([a], [c])
    assert cosine == pytest.approx(-1.0)

    cosine, _, _ = flatten_grads_cosine([a], [d])
    assert cosine == pytest.approx(float(torch.dot(a, d) / (a.norm() * d.norm())))


def test_flatten_grads_cosine_none_and_zero() -> None:
    """None（无图连接）与零梯度都应返回 None（余弦无定义）。"""
    a = torch.tensor([1.0, 2.0])
    assert flatten_grads_cosine([a], [None]) is None
    assert flatten_grads_cosine([None], [a]) is None
    assert flatten_grads_cosine([torch.zeros(2)], [a]) is None
    assert flatten_grads_cosine([], []) is None


def test_summary_detach_override_enables_counterfactual_path() -> None:
    """反事实覆写：解除 summary detach 后策略梯度可达信念私有头。

    默认 ``belief_summary_detach=True`` 下策略损失到四头/backbone 无图路径
    （隔离契约）；显式置 False 构造「开闸②」假想图，策略梯度必须可达四头
    与 belief_query（这正是 learner/SFT 反事实余弦诊断所依赖的管路）。
    """
    torch.manual_seed(2026)
    model = KyokuTransformerActorCritic()
    inputs = actor_inputs(batch=2, action_ids=(1, 7, 12))

    def policy_backward(belief_summary_detach: bool) -> None:
        model.zero_grad(set_to_none=True)
        output = model(
            actor_factors=inputs["actor_factors"],
            actor_numeric=inputs["actor_numeric"],
            actor_lengths=inputs["actor_lengths"],
            query_action_ids=inputs["action_ids"],
            query_pair_counts=inputs["query_pair_counts"],
            legal_mask=inputs["legal_mask"],
            policy_only=True,
            belief_public_grad_scale=0.0,
            belief_readout_enabled=True,
            belief_readout_detach=True,
            belief_summary_detach=belief_summary_detach,
            validate_structure=False,
        )
        log_probs = torch.log_softmax(output["policy_logits"], dim=-1)
        actions = inputs["action_ids"][:, 0]
        (-log_probs.gather(1, actions[:, None]).mean()).backward()

    policy_backward(belief_summary_detach=True)
    assert model.belief_network.hand_head.weight.grad is None
    assert model.belief_query.grad is None

    policy_backward(belief_summary_detach=False)
    assert model.belief_network.hand_head.weight.grad is not None
    assert model.belief_query.grad is not None
    assert model.belief_network.token_matrix.weight.grad is not None
