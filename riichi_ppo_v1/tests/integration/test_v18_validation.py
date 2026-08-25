"""V18 生产校验入口的集成门槛。"""

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.model.parameter_count import assert_v18_parameter_contract


def test_unified_v18_validation_contract() -> None:
    report = assert_v18_parameter_contract(KyokuTransformerActorCritic())
    assert 4_900_000 <= report["total"] <= 5_100_000
