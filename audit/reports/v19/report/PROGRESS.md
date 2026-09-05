# V19 实施进度记录

> 本文件按阶段记录 V18→V19 架构升级：改动文件、关键决策与理由、测试结果。
> 设计依据：`audit/reports/v19/design/` 三册 D1–D32 与 `AGENTS.md`。

## 阶段 0：契约与常量（进行中/已完成部分）

改动文件：
- `RiichiEnv/riichienv-state-machine/src/lib.rs`：`ENCODING_PROTOCOL_VERSION` 18→19。
- `riichi_ppo_v1/model/encoding_protocol.py`：V19 schema 全量变更——删除
  `KIND_CRITIC_FUTURE(14)/SEGMENT_CRITIC_FUTURE(5)/KIND_RIVER_SUMMARY(6)`；
  新增 `KIND_RIICHI_CARD(14)/KIND_BELIEF(15)/SEGMENT_BELIEF(5)`；
  `CONTEXT_TOKENS=256→320`；PLAYER/RIVER_DISCARD/MELD/TILE_STATE/
  OPPONENT_ANALYSIS 字段按 D9 收敛；MELD +meld_turn/called_tsumogiri；
  RIICHI_CARD schema 按 §5。
- `riichi_ppo_v1/model/architecture.py`：ModelConfig `layers=5/critic_layers=1`、
  `preset("v19")`、`_segment_map`/`_assert_structure` 更新（V19 段表 + 立直卡/
  信念 kind；critic 删除 future 校验）。
- `riichi_ppo_v1/model/critic_features.py`：删除 future wall 全部代码；
  改为优先从 Observation.privileged_hands 取四家真手（在线/回放同一数据源）。
- `riichi_ppo_v1/model/bridge.py`：prepare 删除 walls 参数与 future 传参。
- `RiichiEnv/riichienv-core/src/observation/mod.rs` + `state/mod.rs`：
  Observation 新增 `temp_furiten` / `permanent_furiten`（全状态标记）并
  `privileged_hands` 在线填充（仅训练/Rust 侧使用，不进 Actor 输入）。
- `RiichiEnv/riichienv-python/src/current_state_encoding.rs`：河区重构
  （删 SUMMARY/被鸣锚行/relative_seat/supplied）、RIICHI_CARD 恒发射、
  MELD 新字段、TILE_STATE/OPPONENT_ANALYSIS 收敛；新增信念五头标签批量导出
  `prepare_belief_labels_batch`（D26：上帝视角、反事实、无未来信息）。
- `riichi_ppo_v1/model/belief_labels.py`：新增 Python 标签边界（新文件）。
- `riichi_ppo_v1/sft/contract.py`：V19 契约版本与文案。
- `riichi_ppo_v1/model/parameter_count.py`、`tools/validate.py`：
  V19 参数审计入口（阈值 7.2M）。
- `riichi_ppo_v1/configs/v19_ppo.yaml`、`v19_sft.yaml`：自包含 V19 配置
  （PPO 超参沿用 v18_ppo.yaml；新增信念键）。
- `riichi_ppo_v1/docs/v19_input_protocol.md`：新写 V19 输入协议文档。
- `riichi_ppo_v1/tests/v18_fixtures.py`：迁移为 V19 合成张量夹具
  （供测试使用；文件名仅历史遗留，后续阶段统一清理为 v19 命名）。

关键决策与理由：
- 信念 token 注入位置定稿为 SEP_ACTIONS 之后、第一对 Query 之前：满足
  “最后一个信念 token 距第一对 query 恒距 1” 的输入分册 §6 不变式。
- 在线 Observation 全量填充 privileged_hands/temp_furiten 是训练侧特权
  数据源；Actor 编码路径不消费，语义验收将反向断言无泄漏。
- 标签 Loss 返回原始点数，训练侧按 /24000 clip 归一化（训练分册 §9 既定口径）。

测试结果（本阶段已做）：
- `cargo check`/`cargo test -p riichienv-python` 通过（10 tests）。
- 编码器冒烟：真实 fixture `encode_kyoku` 产出 V19 序列（含 RIICHI_CARD，
  无 RIVER_SUMMARY）；belief 标签批量导出形状 [N,102]/[N,3]/[N,105]/[N,102]/[N,102] 通过。

待办：阶段 0 剩余为 v19 单测迁移与契约哈希单测；随阶段 2/3/4 子代理完成后由主会话集成。
