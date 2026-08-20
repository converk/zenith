# Data Model: RiichiLab Bot V16 输入适配

## PreparedDecision

- `observation`: 当前请求的 observation 或带 override 的 `ObservationView`
- `seat`: bot 所属绝对座位
- `history_factors/history_numeric/history_lengths`: V16 objective facts
- `snapshot_kinds/snapshot_cat/snapshot_num/snapshot_lengths`: V16 compact snapshot
- `query_rows/query_action_ids/query_pair_counts`: 每合法动作一对 query
- `legal_mask`: 241 维合法动作掩码
- `legal_jsons`: 当前 observation 的 canonical MJAI legal templates
- `event_context`: 用于安全响应补 target/pai 的最近响应窗口事实

## OnlineStateBridge

- 持有单席 `riichi.MjaiKyokuStateMachineManager(1)`
- 持有 `BatchedStateBridge` 适配器并复用训练侧 V16 prepare/decode
- 持有在线字段 tracker,从 accepted MJAI events 重建缺失 snapshot 字段

## Online Snapshot Tracker

- `riichi_declared/accepted/declaration_indices/sutehais`
- `tsumogiri_flags/last_tedashis/discard_counts`
- `tiles_left/tsumo_count`
- `missed_agari_doujun/missed_agari_riichi`
- `last_discard/current_claim_window` 辅助字段

## PolicyEngine

- `checkpoint`: 只读模型路径
- `model`: strict loaded `KyokuTransformerActorCritic`
- `metadata`: checkpoint format-like diagnostic fields,不作为兼容闸门
- `infer(prepared)`: V16 semantic validation + `forward_v16(policy_only=True)` + argmax

## Validation Rules

- `query_pair_counts == legal_mask.sum()`。
- `query_rows[0::2]` 必须是 offense,`query_rows[1::2]` 必须是 defense。
- query action ids 必须与 legal ids 顺序一致。
- snapshot lengths、history lengths、query lengths 必须在数组容量内,总 context 不超过模型配置。
- V16 categorical slot 必须在 `encoding_protocol.py` 声明基数内。
