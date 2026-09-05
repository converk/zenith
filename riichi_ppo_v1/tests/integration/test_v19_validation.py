"""V19 生产校验入口的集成门槛。"""

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.model.parameter_count import assert_v19_parameter_contract


def test_parameter_contract_under_7_2m() -> None:
    report = assert_v19_parameter_contract(KyokuTransformerActorCritic())
    assert 7_000_000 <= report["total"] <= 7_200_000
