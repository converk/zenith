from __future__ import annotations


def test_bot_reuses_training_encode_path() -> None:
    import riichi_ppo_v1.model.bridge as training_bridge
    import riichi_ppo_v1.model.current_state as training_current_state

    import riichi_lab_bot.bridge as bot_bridge

    # 在线桥接与训练/评测共用同一条 current_state.encode_batch 编码路径。
    assert bot_bridge.encode_batch is training_current_state.encode_batch
    assert (
        bot_bridge.action_jsons_and_decision_flag
        is training_bridge.action_jsons_and_decision_flag
    )


def test_v19_protocol_constants_are_active() -> None:
    import riichi_ppo_v1.model.encoding_protocol as protocol

    # V19 定版:删 RIVER_SUMMARY/critic-future,新增 RIICHI_CARD=14、BELIEF=15、
    # SEGMENT_BELIEF=5,context_tokens=320。
    assert protocol.KIND_RIICHI_CARD == 14
    assert protocol.KIND_BELIEF == 15
    assert protocol.SEGMENT_BELIEF == 5
    assert protocol.CONTEXT_TOKENS == 320
    assert not hasattr(protocol, "KIND_CRITIC_FUTURE")
    assert not hasattr(protocol, "KIND_RIVER_SUMMARY")
    assert not hasattr(protocol, "SEGMENT_CRITIC_FUTURE")
