"""V18 拓扑、隔离注意力与动作重排不变量。"""

import torch

from riichi_ppo_v1.model.architecture import (
    KyokuTransformerActorCritic,
    ModelConfig,
    _isolated_action_layout,
)
from riichi_ppo_v1.tests.v18_fixtures import actor_inputs


def test_fixed_topology_and_isolated_mask() -> None:
    config = ModelConfig.preset("v18")
    assert (config.d_model, config.query_heads, config.kv_heads, config.head_dim) == (256, 16, 4, 16)
    assert (config.ffn_dim, config.layers, config.shared_layers, config.critic_layers) == (704, 4, 3, 2)
    mask, valid, positions = _isolated_action_layout(torch.tensor([2]), torch.tensor([2]), 6)
    assert valid.all()
    assert mask[0, 0, 2, 0] and mask[0, 0, 2, 3]
    assert not mask[0, 0, 2, 4] and not mask[0, 0, 4, 2]
    assert positions.tolist() == [[0, 1, 2, 3, 2, 3]]


def test_pair_permutation_preserves_action_aligned_logits() -> None:
    torch.manual_seed(7)
    model = KyokuTransformerActorCritic().eval()
    inputs = actor_inputs(batch=2)
    with torch.no_grad():
        baseline = model.forward_actor(**inputs)["raw_policy_logits"]
        permutation = torch.tensor([2, 0, 1])
        altered = dict(inputs)
        altered["query_action_ids"] = inputs["query_action_ids"][:, permutation]
        altered["query_rows"] = inputs["query_rows"].view(2, 3, 2, 15)[:, permutation].reshape(2, 6, 15)
        permuted = model.forward_actor(**altered)["raw_policy_logits"]
    torch.testing.assert_close(baseline, permuted, atol=1e-5, rtol=1e-5)


def test_query_metadata_changes_embedding() -> None:
    model = KyokuTransformerActorCritic().eval()
    rows = actor_inputs(batch=1)["query_rows"]
    baseline = model.query_embedding(rows)
    for column in (2, 3, 4):
        changed = rows.clone(); changed[:, :, column] = 2
        assert not torch.equal(baseline, model.query_embedding(changed))
