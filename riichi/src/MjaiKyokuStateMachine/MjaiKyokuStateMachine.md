# MjaiKyokuStateMachine V4

状态机是 RiichiEnv MJAI 广播与 PPO V4 输入之间的严格边界。每张桌有四个相对视角状态机；
它们保存完整原始事件生命周期、可缓存的公开 event block 历史以及可从当前 Observation
补全的临时 board state。

```text
new_events -> apply_events_batch -> prepare_decisions
legal_actions -> action mask / exact MJAI action template
```

`prepare_decisions(batch_indices, legal_action_jsons, snapshot_jsons)` 返回：

```text
block_kinds       [B, L]       uint8
turn_fields       [B, L, 4, 4] uint8
meld_fields       [B, L, 8]    uint8
board_state       [B, 12, 160] uint8
block_lengths     [B]          int64
action_mask       [B, 241]     bool
history_generation[B]          int64
```

模型顺序是 `[confirmed event blocks][12 board tokens][decision token]`。只有 confirmed
blocks 可进入 KV cache；board state 由当前决策 snapshot 生成，不会修改缓存。

普通弃牌、立直、立直成立和翻宝牌存入 `TURN_BLOCK`；吃碰杠存入 `MELD_BLOCK`。`tsumo` 只更新
活牌山和当前局面，终局事件只表达边界和结算。详细字段和隐私规则见
`riichi_ppo_v1/KyokuEventTupleProtocol.md`。
