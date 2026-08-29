from __future__ import annotations


def test_bot_reuses_training_bridge_helpers() -> None:
    import riichi_ppo_v1.model.bridge as training_bridge

    import riichi_lab_bot.bridge as bot_bridge

    assert bot_bridge.BatchedStateBridge is training_bridge.BatchedStateBridge
    assert (
        bot_bridge.action_jsons_and_decision_flag
        is training_bridge.action_jsons_and_decision_flag
    )
