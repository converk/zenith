"""V18 参数量、分项与 state-key 门槛。"""

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.model.parameter_count import assert_v18_parameter_contract, parameter_report


def test_parameter_contract() -> None:
    model = KyokuTransformerActorCritic()
    report = assert_v18_parameter_contract(model)
    assert report["total"] <= 6_000_000
    assert report["state_key_count"] > 0
    assert set(report["by_group"]) == {"embedding", "shared", "actor", "critic", "head"}
    assert report["by_group"]["embedding"] > 0
    assert report["by_group"]["shared"] > 0
    assert report["by_group"]["actor"] > 0
    assert report["by_group"]["critic"] > 0
    assert report["by_group"]["head"] > 0
    assert not report["forbidden_q_keys"]


def test_no_mha_or_legacy_keys() -> None:
    model = KyokuTransformerActorCritic()
    keys = set(model.state_dict())
    for forbidden in ("history", "snapshot", "query_embedding", "snapshot_embeddings",
                      "candidate_q", "q_scorer", "dueling_q"):
        assert not any(forbidden in key for key in keys), forbidden
