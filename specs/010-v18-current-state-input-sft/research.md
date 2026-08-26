# Research: V18 当前局面输入与 Actor 决策架构重构

## 1. 总体结论

- 现行 V18 = 事件历史(Objective Facts) + 54 行 Atomic Snapshot + 每动作 Query(局部 position ID)。
  本次改为**决策时刻状态快照**：Shared 公共前缀(桌况/自身手牌/self 分析/四家 player/三家完整牌河+
  两个六张摘要/当前副露/34 tile-state) + Actor-only 尾部(三个 Opponent Analysis + 按 action ID
  升序的 Offense/Defense Query)。Critic 私有输入不变(三家真实闭手 + 未来五张)。
- 编码器放在 `riichienv-python`(可直接访问 `riichienv_core::observation::Observation` 全部字段)，
  复用 `riichi` crate 的 `shanten`/`analysis` 内核；`riichi` crate 提供查询编码与状态机(仅用于
  合法掩码/动作解码/生命周期)。模型输入不再包含任何事件历史 token。

## 2. 关键决策

### 2.1 编码位置与 Rust/PyO3 分工
- **Decision**: 新增 `riichienv-python/src/current_state_encoding.rs`，PyO3 函数
  `riichienv.prepare_current_state_batch(observations) -> CurrentStateBatch`
  （扁平 token rows `[total, 32] i32`、numeric `[total, 8] f32`、offsets `[B+1] i64`）。
- Rationale: Observation 的当前字段(hands/discards/tsumogiri_flags/melds/scores/riichi*/dora/
  合法动作)都在 `riichienv-core`；该 crate 已在 Rust 层依赖 `riichi`，可直接调用
  `riichi::shanten::calculate*` 与 `riichi::analysis` 内核，避免跨进程数组搬运与语义漂移。
- Alternatives: 放在 `riichi` 状态机 crate → 无权访问 Observation；放在 Python → 大量逐 decision
  循环、无法复用 Rust 向听/防守内核，性能差且不一致。

### 2.2 查询编码复用
- **Decision**: O0–O9/D0–D9 继续由现有 `riichi.encode_query_batch` +
  `riichienv.analyze_encoding_yaku_batch` 计算（行宽 15 布局不变），新编码器只产共享/分析 token；
  Python 侧按 action ID 升序把 query 行拼进 actor 序列（fields[2..16]）。
- Rationale: 该内核已被 25 个基线测试覆盖，语义无需重写；只换装配与模型消费方式。
- Alternatives: 在新编码器内重算 query 行 → 重复实现并有漂移风险。

### 2.3 行布局单一来源
- **Decision**: `riichi_ppo_v1/model/encoding_protocol.py` 作为字段/基数/segment/kind/separator
  的**唯一 Python 单一来源**（类似现行 QUERY_ROW 布局），contract SHA256 由其生成；Rust 编码器
  镜像同一常量（模块内注释说明镜像关系），Python 语义校验 + Rust/PyO3 集成测试做交叉验证。
- Rationale: 现行项目的 snapshot 走 Rust schema 导出、query 行走 Python 单源；新协议字段太多，
  全表双写风险大。以 Python 单源保证 manifest/嵌入/校验一致，以集成测试保证 Rust 编码一致。
- Alternatives: Rust schema 导出 + Python 消费（更“单源”但工作量翻倍且仍要维护字段索引）。

### 2.4 嵌入拓扑（保持 d_model=256，密集槽位融合）
- **Decision**: 密集类别（TABLE/SELF_STATE_ANALYSIS/PLAYER/RIVER_SUMMARY/MELD/
  OPPONENT_ANALYSIS/ACTION_QUERY）使用「每离散槽位独立 embedding 表(dense_slot_dim=32) →
  按规范顺序 concat → 每类别专属 Linear 投影到 dense_fusion_dim=512 → 共享
  RMSNorm+gated/SiLU MLP → 投影回 256」；简单类别（BOS/SELF_HAND/RIVER_DISCARD/TILE_STATE/
  CRITIC_*）使用轻量 concat+Linear(simple_slot_dim=16)。
- Rationale: 每类别独立大 MLP 会显著超 6.0M 参数；共享融合层 + 类别专属输入投影把总参数压在
  ~4.5M 附近，同时保留槽位间非线性交互；这与提示词“先减少重复的类别专属融合层或共享合理组件”
  一致。
- Alternatives: 每类别 512 融合全独立 → 估算 >7M；扩大到 d_model=384/512 → 被提示词明确禁止。

### 2.5 GQA 与 MHA
- **Decision**: 生产只保留 16Q/4KV GQA（kv_heads=4），无 MHA 双分支、无实验 flag。
- Rationale: 现行 4,940,802 参数把 kv_heads 4→16 会增加 589,824 并提升 4 倍 K/V 存储，但输入
  表示碰撞来自无类型求和而非 KV 共享；参考 Ainslie et al. GQA (EMNLP 2023) 与 RoFormer
  (arXiv:2104.09864)。RoPE 与 GQA 无冲突。

### 2.6 RoPE/位置/注意力
- **Decision**: 所有有效 token（含 BOS、separator、摘要、Analysis、Query、Critic、Value Query）
  使用连续唯一单调递增 position ID；Shared `0..P-1`，Actor 尾部从 P 续接，Critic 分支自己从 P
  续接。Shared 公共 backbone 用**双向 GQA**；Actor-only 层用结构化 mask；Critic 用全双向 +
  Value Query 最后读取全部。动作对不再共享局部 position ID。
- Rationale: 状态快照不是自回归序列，双向建模能正确交互桌况/手牌/牌河/副露/tile-state；
  结构化 mask 保证公共 token 不读 Actor 尾部、动作对隔离、padding 不可见。

### 2.7 上下文上界（context_tokens=256）
- **Decision**: 保持 256，给出严格上界并合成极端 fixture 验证。
- 上界核算（每个 player 最多 18 次舍牌、自身暗手最多 14 种非零牌、四家最多 16 个副露、
  单决策最多 30 个合法动作对）：
  - Shared：BOS 1 + 桌况 1 + SEP_SELF_HAND 1 + 自身手牌 14 + self 分析 1 + SEP_PLAYERS 1 +
    4 player + SEP_RIVERS 1 + 3×(SEP 1 + 摘要 2 + 牌河 18) = 63 + SEP_MELDS 1 + 副露 16 +
    SEP_TILE_STATE 1 + 34 tile-state = **139**；
  - Actor 尾部：SEP_OPPONENT_ANALYSIS 1 + 3 + SEP_ACTIONS 1 + 2×30 = **65**；
  - Actor 总上界 ≈ **204 ≤ 256**（实测极端应落在 185–215 区间）；
  - Critic 总上界 = Shared 139 + SEP_CRITIC 1 + 3×13 闭手 + 5 未来 + Value 1 = **185 ≤ 256**。
- Rationale: 不截断任何段；256 充分余量。
- Alternatives: 若实现后严格上界有更高组合（如 24 张牌河 ×3 = 72 + 更多动作），取 288/320 并
  记录，仍不截断。

### 2.8 数据格式 / manifest
- **Decision**: encoded shard 改存 `actor_offsets/actor_factors/actor_numeric/query_offsets/
  action_offsets/query_rows/action_ids/legal(packbits)/actions/身份字段`；
  `EncodedSample` 改为 `actor_factors[T,32]/actor_numeric[T,8]/query_rows[2Q,15]/action_ids/
  legal_mask/action/身份`。manifest 仍用 `riichi-sft-encoded-v18`，`encoding_protocol_version=18`，
  但 `encoding_contract_sha256` 因字段 schema/顺序变化而更新（旧 shard/checkpoint 一律不兼容）。
- Rationale: 同一 actor 序列字节级贯穿 precompute→shard→collator→模型；无 history/snapshot 双轨。

### 2.9 自己牌河
- **Decision**: 不生成自身逐张 river token；信息无损转移：34 tile-state 的 self_discard_count/
  self_ever_discarded、SELF_PLAYER.river_length + 立直宣言信息、SELF_STATE_ANALYSIS 精确振听
  （永久/同巡/立直）由 same-river/kind 直接导出。
- Rationale: 提示词要求省约 10–15 token，且不得破坏振听语义（振听由“自己舍过的牌种”与
  missed_agari 标志导出，无需逐张）。

### 2.10 决策模式（桌况）
- **Decision**: mode 0=主动舍牌（drawn_tile 非 None）；mode 1=响应舍牌；mode 2=响应加杠。
  后两者从 `observation.new_events()` 最后一条公开事件的 type（dahai vs kakan）判定（仅用于
  编码分类，不进入模型输入）。
- Rationale: Observation 无“最后供牌者/供牌”公共字段；事件只用于同步与模式判定，符合提示词
  “MJAI new_events() 可继续用于同步/生命周期/奖励/动作执行”。

### 2.11 信息隔离与 Critic
- **Decision**: Actor 序列不含 critic rows；Critic rows 仍由 Python `critic_features.py` 构造
  （segment 4=闭手、segment 5=未来五张、SEP_CRITIC），但改用新行宽并带 1-based 相对座次与
  position 字段。模型 Critic 分支只接收 shared hidden + critic rows + value query。
- Rationale: 闭手/未来五张在在线环境不是 Observation 字段（仅在 replay privileged_hands 与
  env.walls），保持 Python 装配 + 测试隔离，代价低且已被现有测试覆盖。

## 3. 参数估算

- 类别专属输入投影(Lin(F_cat→512))合计约 2.4M；共享 gated MLP(512→1024→256)约 0.65M；
  简单类别 concat 投影约 0.18M；token-kind/segment 嵌入约 0.03M；6 层 GQA 骨干
  (384·256+256·256=163,840/层)约 0.98M；policy head(512→256→1)约 0.20M；
  value head + value query 约 0.001M。合计约 **4.5M**，<6.0M 有充分余量；实现后必须用统一
  parameter report 验证并按 `embedding/shared/actor/critic/head` 分项报告。

## 4. 风险与缓解

- Rust/PyO3 编码器大而新 → 用真实 replay fixture + 合成边界测试逐字段断言；先写 Python 侧
  schema 与语义校验，再实现 Rust 编码器，最后由集成测试对齐。
- 旧 PPO 调用方（worker/inference/learner/rollout_buffer/evaluation）引用旧 `PreparedBatch`
  字段 → 本阶段不改、不兼容、不测试，只在 audit 记为待迁移；保持 `PreparedBatch` 类名与
  `bridge.prepare` 入口存在（字段更新），因此 import 不碎、运行需迁移。
- 上下文若超 256 → 按 2.7 提升固定值并记录，不截断。
- 旧 encoded shard/checkpoint → manifest/hash/模型配置 strict fail closed，不迁移、不覆盖、
  不删除物质资产。
