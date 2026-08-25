"""V18 参数量与 state-key 门槛。"""

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.model.parameter_count import assert_v18_parameter_contract


def test_parameter_and_state_key_contract() -> None:
    report = assert_v18_parameter_contract(KyokuTransformerActorCritic())
    assert report["total"] == 4_940_802
    assert not report["forbidden_q_keys"]
