"""V18 Query metadata、动作 ID 集合与 supplier 域测试。"""

import numpy as np
import pytest

from riichi_ppo_v1.model.encoding_protocol import (
    QUERY_ROW_ACTION_ID,
    QUERY_ROW_PRIMARY_TILE,
    QUERY_ROW_SOURCE_SEAT,
)
from riichi_ppo_v1.model.semantic_validation import assert_actor_input_semantics
from riichi_ppo_v1.tests.v18_fixtures import actor_inputs


def _numpy_inputs() -> dict[str, np.ndarray]:
    return {name: value.numpy() for name, value in actor_inputs(batch=1).items()}


def test_action_metadata_and_unordered_legal_set_are_accepted() -> None:
    inputs = _numpy_inputs()
    rows = inputs["query_rows"]
    rows[:, :, 2] = 4
    rows[:, :, QUERY_ROW_PRIMARY_TILE] = 34
    rows[:, :, QUERY_ROW_SOURCE_SEAT] = 3
    assert_actor_input_semantics(**inputs)
    assert set(inputs["query_action_ids"][0]) == set(np.flatnonzero(inputs["legal_mask"][0]))


def test_pair_metadata_disagreement_and_duplicate_ids_are_rejected() -> None:
    inputs = _numpy_inputs()
    inputs["query_rows"][0, 1, QUERY_ROW_PRIMARY_TILE] = 1
    with pytest.raises(AssertionError):
        assert_actor_input_semantics(**inputs)


def test_supplier_applicability_is_fail_closed() -> None:
    inputs = _numpy_inputs()
    inputs["query_rows"][0, 0:2, 2] = 10
    with pytest.raises(AssertionError, match="lacks source"):
        assert_actor_input_semantics(**inputs)

    inputs = _numpy_inputs()
    inputs["query_rows"][0, 0:2, QUERY_ROW_SOURCE_SEAT] = 1
    with pytest.raises(AssertionError, match="non-supplier"):
        assert_actor_input_semantics(**inputs)


def test_native_supplier_metadata_and_non_supplier_na() -> None:
    from RiichiEnv.tests.env.helper import helper_setup_env
    from riichienv import Action, ActionType
    import riichi

    from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision

    env = helper_setup_env(
        hands=[
            [0, 4, 8, 12, 16, 20, 24, 36, 40, 44, 48, 52, 56],
            [1, 2, 3, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42], [], [],
        ],
        current_player=0,
        active_players=[0],
        drawn_tile=108,
        wall=list(range(136)),
    )
    observation = env.step({0: Action(ActionType.DISCARD, tile=0)})[1]
    bridge = BatchedStateBridge(riichi.MjaiKyokuStateMachineManager(1), 1)
    bridge.sync([{seat: env.get_observation(seat) for seat in range(4)}])
    prepared = bridge.prepare([Decision(0, 1, observation)])
    rows = prepared.query_rows[0, :2 * int(prepared.query_pair_counts[0])]
    action_types = rows[0::2, 2]
    source_codes = rows[0::2, QUERY_ROW_SOURCE_SEAT]
    assert np.all(source_codes[action_types == 1] == 0)
    # 观察者 1 响应观察者 0 的供牌,相对座次为上家,协议编码为 3。
    assert np.all(source_codes[np.isin(action_types, (5, 6))] == 3)
    inputs = _numpy_inputs()
    inputs["query_action_ids"][0, 1] = inputs["query_action_ids"][0, 0]
    inputs["query_rows"][0, 2:4, QUERY_ROW_ACTION_ID] = inputs["query_action_ids"][0, 0]
    with pytest.raises(AssertionError, match="legal-mask set"):
        assert_actor_input_semantics(**inputs)
