"""V18 Actor/Critic 信息边界测试。"""

import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.tests.v18_fixtures import actor_inputs


def test_actor_api_has_no_private_input_and_is_deterministic() -> None:
    model = KyokuTransformerActorCritic().eval()
    inputs = actor_inputs(batch=1)
    with torch.no_grad():
        first = model.forward_actor(**inputs)["raw_policy_logits"]
        second = model.forward_actor(**inputs)["raw_policy_logits"]
    torch.testing.assert_close(first, second)
    assert "critic_factors" not in model.forward_actor.__annotations__


def _critic_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.tensor([[
        [4, 4, 2, 2, 1, 1, 0, 4, 0, 1],
        [4, 4, 2, 3, 2, 2, 0, 4, 0, 1],
        [4, 4, 2, 4, 3, 3, 0, 4, 0, 1],
        [5, 6, 3, 1, 1, 1, 0, 1, 0, 1],
        [5, 6, 3, 2, 1, 2, 0, 1, 0, 1],
        [5, 6, 3, 3, 1, 3, 0, 1, 0, 1],
        [5, 6, 3, 4, 1, 4, 0, 1, 0, 1],
        [5, 6, 3, 5, 1, 5, 0, 1, 0, 1],
    ]], dtype=torch.long)
    return rows, torch.tensor([8])


def test_private_mutation_changes_critic_but_not_actor_logits() -> None:
    torch.manual_seed(19)
    model = KyokuTransformerActorCritic().eval()
    inputs = actor_inputs(batch=1)
    critic, lengths = _critic_inputs()
    changed = critic.clone()
    changed[0, 0, 5] = 9
    changed[0, 3, 5] = 7
    with torch.no_grad():
        first = model(**inputs, critic_factors=critic, critic_lengths=lengths)
        second = model(**inputs, critic_factors=changed, critic_lengths=lengths)
    torch.testing.assert_close(first["raw_policy_logits"], second["raw_policy_logits"])
    assert not torch.equal(first["value"], second["value"])
