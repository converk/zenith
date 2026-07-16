# MjaiKyokuStateMachine V5

状态机将 RiichiEnv 的逐观察者 MJAI 增量转成 Token V5 的公开语义输入，并保留固定
241 动作空间与原始 MJAI action template 的精确映射。

```text
Observation.new_events -> apply_events_batch -> prepare_decisions
Observation.legal_actions -> bool[241] -> decode_actions -> select_action_from_mjai
```

`prepare_decisions(batch_indices, legal_action_jsons, snapshot_jsons)` 返回：

```text
token_factors       [B, L, 10] uint8
token_numeric       [B, L, 8]  float32
token_lengths       [B]        int64
action_mask         [B, 241]   bool
history_generation  [B]        int64
```

每个 table 有四个相对观察者状态机。它们维护同局 append-only 的公开语义事件 history；
当前局面 suffix 从当前 snapshot 生成。历史包括开局、弃牌、吃碰杠、翻宝和立直事件，摸牌只更新
活牌山，终局只表达 reward/lifecycle 边界。所有 token 规则、隐私约束与 factor 字段见
`riichi_ppo_v1/KyokuEventTupleProtocol.md`。
