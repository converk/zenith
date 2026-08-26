"""V18 生产校验入口的集成门槛。"""

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.model.parameter_count import assert_v18_parameter_contract


def test_parameter_contract_under_6m() -> None:
    report = assert_v18_parameter_contract(KyokuTransformerActorCritic())
    assert report["total"] <= 6_000_000
