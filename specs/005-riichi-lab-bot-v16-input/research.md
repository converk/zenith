# Research: RiichiLab Bot V16 输入适配

## Decision: 复用训练侧 `prepare_v16`

**Rationale**: 训练侧已经负责 history/objective facts、compact snapshot、query pair、action id 与 legal mask 的权威装配。bot 单独实现会引入线上/训练漂移。

**Alternatives considered**: 在 bot 内调用 `build_snapshot_facts()` 和 `analyze_action_queries()` 手工拼接。拒绝原因是还需重复 decoded action representative 映射与 padding 规则。

## Decision: checkpoint 按模型配置 strict load

**Rationale**: V16 SFT 与 V17 PPO payload 都含 `model_config` 和 `model`;旧 `sft_contract_version` 仍可能是历史字符串,不应作为 bot 的兼容闸门。

**Alternatives considered**: 复用 `evaluation.load_policy_adapter()`。拒绝原因是 bot 还需要自定义 warmup、metadata 与 online `PreparedDecision`。

## Decision: 语义校验拆为 V16 专用断言

**Rationale**: V16 snapshot/query 不再是旧 10 列 token suffix;旧 `assert_actor_token_semantics()` 不能覆盖 V16 三段结构。

**Alternatives considered**: 仅依赖模型前向 shape error。拒绝原因是模型不会检查业务语义,也不能解释 query/action mask 不一致。

## Decision: 缺字段由在线 tracker 重建,完整字段用于一致性测试

**Rationale**: RiichiLab payload 可能缺少 V16 query 依赖字段;本地完整 RiichiEnv observation 是最可靠 oracle。

**Alternatives considered**: 缺字段统一默认 False/0。拒绝原因是会污染 tiles-left、对手摘要和 missed-agari/furiten 语义。
