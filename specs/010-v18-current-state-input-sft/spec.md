# Feature Specification: V18 当前局面输入与 Actor 决策架构重构

**Feature Branch**: `010-v18-current-state-input-sft`

**Created**: 2026-08-27

**Status**: Draft

**Input**: `audit/reports/v18/design/V18当前局面输入与Actor决策架构重构提示词.md`（完整需求），
叠加用户约束：不引入 PPO 阶段考虑、不兼容旧代码/输入格式（直接删除）、不实现 PPO/GRP 训练。

## 背景与范围

现行 V18 使用「追加式完整历史事件 + 当前状态 suffix + 固定 54 行 Atomic Snapshot + 每动作一对
Query（局部 position ID）」。本次重构彻底移除 Actor 的完整事件历史，把公共输入改为**决策时刻的
局面快照**：完整三家对手牌河（逐张）、当前四家副露、自身手牌、34 种牌确定性状态、三家 Opponent
Analysis、每合法动作一对 Offense/Defense Query。Critic 私有输入保持「三家真实闭手 + 未来五张牌」。
PPO 代码、配置、训练、评测和运行链不在本阶段范围；旧 V18 输入契约不保留兼容分支、不提供
state-dict 迁移（旧 checkpoint/encoded shard 视为不兼容）。本阶段不生成完整 60% 编码数据集，
不启动正式长时 SFT、GRP 或 PPO。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 当前局面公共快照编码 (Priority: P1)

作为训练数据生产者，我需要每个决策时刻只获得「此刻桌面上仍存在什么」的完整公开结构快照，
不再需要「这一局依次发生了什么」的 MJAI 事件序列，以便模型直接对当前局面建模。

**Why this priority**: 公共快照是新输入协议的地基，所有下游（Opponent Analysis、Query、SFT）
都依赖它。

**Independent Test**: 用真实 replay fixture 与合成边界状态逐决策编码，断言：三家对手牌河逐张
顺序（按相对座次：下家/对面/上家，每家内部由旧到新）、每家恰好一个 FIRST_SIX_SUMMARY 与
RECENT_SIX_SUMMARY、当前副露只表达当前形态（kakan 不重复历史 pon）、恰好 34 个 tile-state
token、无 tiles_left/独立当前供牌公共 token/MJAI 历史事件 token；自己牌河不生成逐张 token，
但自己舍牌计数与振听语义完全正确。

**Acceptance Scenarios**:

1. **Given** 任意合法决策状态，**When** 按新协议编码，**Then** 共享公共前缀严格按
   `[BOS] → 桌况 → [SEP_SELF_HAND] → 自身手牌 → SELF_STATE_ANALYSIS → [SEP_PLAYERS] →
   SELF_PLAYER → SHIMOCHA_PLAYER → TOIMEN_PLAYER → KAMICHA_PLAYER → [SEP_RIVERS] →
   [SEP_SHIMOCHA_RIVER] → summary/discards/summary → [SEP_TOIMEN_RIVER] → … →
   [SEP_KAMICHA_RIVER] → … → [SEP_MELDS] → 四家副露 → [SEP_TILE_STATE] → 34 tile-state`
   排列，且任何段缺失、重复、错序均 fail closed。
2. **Given** 牌河不足六张，**When** 编码摘要，**Then** 内部 6 槽带有效长度、padding 为严格零
   贡献，且不与合法牌混淆。
3. **Given** 观察者自身牌河，**When** 编码，**Then** 不产生自身逐张 river token，但 34
   tile-state 中自己舍牌张数/是否舍过、SELF_PLAYER 牌河长度、SELF_STATE_ANALYSIS 精确振听
   类型（永久/同巡/立直）与事实一致。
4. **Given** 任意状态，**When** 检查输入，**Then** 不存在 `tiles_left`、独立当前供牌公共字段、
   MJAI 历史事件 token 或 54 行 Atomic Snapshot 独立输入块。

---

### User Story 2 - Actor-only Opponent Analysis 与 Action Query (Priority: P1)

作为策略消费者，我需要每个合法动作恰好一对 Offense/Defense Query，与三个只由公开信息派生的
Opponent Analysis token，且它们都位于 Actor 分支，参与 action logits。

**Why this priority**: Actor 决策质量取决于这些确定性槽位聚合信息；Query 排序与隔离必须规范。

**Independent Test**: 打乱环境合法动作输入顺序，经 action ID 升序规范排序编码后，序列与按
action ID 对齐的 raw logits 不变；任一 Opponent Analysis 有效字段改变都会改变其 embedding，
并在适当 fixture 中能影响对应 action logits。

**Acceptance Scenarios**:

1. **Given** 一个合法动作集合，**When** 编码 Query，**Then** 每个动作恰好两个 token，按 action
   ID 升序先 Offense 后 Defense；环境返回顺序不影响编码结果。
2. **Given** 任一动作 Query，**When** 检查 O0–O9/D0–D9，**Then** 保留现行语义（O0 动作后向听、
   O1 有效牌种数、O2 有效牌剩余数、O3 等待牌种数、O4 等待役覆盖、O5 基础番、O6 振听、
   O7 门清、O8 可立直、O9 保留宝牌/赤牌数；D0–D2 对三家现物、D3–D5 筋关系、D6–D8 动作后
   手中对三家现物库存、D9 候选牌公开出现数），chi/pon/daiminkan/ron 的 supplier 为 1..3，
   其余为 N/A。
3. **Given** 每个动作对，**When** 应用注意力，**Then** 该动作对可读取全部 Shared 公共前缀、
   三个 Opponent Analysis 与自己的 Offense/Defense 两 token；不同动作对互不可见；公共 token
   不读取 Actor 尾部；Opponent Analysis 不读取 Action Query。
4. **Given** 三个 Opponent Analysis token，**When** 检查位置，**Then** 位于 Actor 分支、
   Action Query 之前，由确定槽位聚合（非空白 learned query、非概率预测器）。

---

### User Story 3 - RoPE、分隔符、注意力布局与信息边界 (Priority: P1)

作为训练框架维护者，我需要所有有效 token（含 BOS、分隔符、摘要、Opponent Analysis、
Action Query、Critic 私有 token 与 Value Query）应用 RoPE，公共快照用双向 GQA，Actor/Critic
严格信息隔离。

**Why this priority**: 位置语义与信息边界是模型正确性与公平性的基础，必须可自动验证。

**Independent Test**: 固定全部 Actor 输入只改变三家闭手或未来牌山，逐 action ID 的 Actor raw
logits 完全不变；合法改变 Critic 私有输入时 value 必须变化；Critic 输入长度与 value 不受
Analysis/Action token 拼接影响。

**Acceptance Scenarios**:

1. **Given** 每条分支，**When** 计算位置，**Then** 所有非 padding token 使用连续、唯一、单调
   递增 position ID；分隔符占位、不在类别边界重置；Shared 前缀为 `0..P-1`，Actor 尾部从 P
   续接，Critic 尾部在自己的分支中从 P 续接；动作对不再复用局部 position ID。
2. **Given** 公共前缀，**When** 应用注意力，**Then** 使用有效公共 token 之间双向 GQA，而不是
   沿类别排列做 causal attention。
3. **Given** Actor 输入，**When** 应用 mask，**Then** Shared 只读 Shared；三个 Analysis 读全部
   Shared 与三个 Analysis；每个 Action Query 读全部 Shared、三个 Analysis 与自己动作对；
   不同动作对互不可见；padding 永不被有效 token 读取。
4. **Given** Critic 输入，**Then** 只读 Shared 公共表示、三家真实闭手、未来五张牌与 Value
   Query；Value Query 能读全部有效 Critic 输入；Actor-only Analysis/Action token 不进入
   Critic shape/length/embedding/mask。
5. **Given** 分隔符，**When** 校验，**Then** 使用类别专属 learned separator（不共用无类型
   `[SEP]`），集合单点定义，顺序严格校验，缺失/重复/错序 fail closed。

---

### User Story 4 - 密集 token 槽位融合与 GQA 生产契约 (Priority: P1)

作为模型工程师，我需要密集类别（summary、SELF_STATE_ANALYSIS、player、Opponent Analysis、
Query、meld）使用槽位感知非线性融合，而保持 `d_model=256`、16Q/4KV GQA、3+1+2 层生产拓扑，
总参数控制在 6.0M 以内。

**Why this priority**: 输入可区分性首先由槽位 embedding 与融合函数决定，不由 KV head 是否共享
决定；本 Story 锁定生产契约并给出参数/上下文统计。

**Independent Test**: 每个有效槽位单独改变都改变最终 token embedding；交换前六/最近六内部两个
位置改变 embedding；padding 与合法零值不混淆；梯度到达所有槽位 embedding 与融合层；激活槽位
数量变化不因简单求和导致未归一化幅度爆炸；大批随机合法组合不存在因编码器错误造成的完全相同
embedding；metadata 消融影响对应 Action/Opponent Analysis embedding 与最终 logits。

**Acceptance Scenarios**:

1. **Given** 生产模型，**When** 检查配置，**Then** `d_model=256`、`query_heads=16`、
   `kv_heads=4`、`head_dim=16`、`ffn_dim=704`、3 Shared + 1 Actor-only + 2 Critic 层、
   RMSNorm/RoPE/gated FFN、GQA（无 MHA 双分支、无实验 flag）。
2. **Given** 密集类别，**When** 嵌入，**Then** 每个离散槽位使用独立字段/位置 embedding 表
   （不共用 ID），`dense_slot_dim=32`，固定槽位按规范顺序 concat（非无类型求和），numeric 先
   稳定归一化再专属投影，RMSNorm + gated/SiLU MLP + 投影回 `d_model=256`，
   `dense_fusion_dim=512`；padding 子槽严格零贡献。
3. **Given** 最终模型，**When** 统计参数，**Then** 提供统一参数统计（embedding/shared/actor/
   critic/head 分项），总参数 ≤ 6.0M；state dict 不含 MHA 双分支、旧 Snapshot/history adapter、
   Q scorer/Q boost。
4. **Given** 上下文目标，**When** 计算，**Then** `context_tokens=256`（若严格上界超过 256 用
   下一个有充分余量的固定值并解释），不静默截断任何段；提供理论/实测上界与既定 60% selection
   上的代表性统计（public/Actor/Critic 的 mean/p50/p95/p99/max 与各 segment 贡献）。

---

### User Story 5 - Actor-only SFT 全链路生命周期 (Priority: P1)

作为 SFT 工程师，我需要从 replay/precompute/shard/collator 到模型入口编码完全一致，并完成
Actor-only SFT 的前向、BC loss、反向、optimizer step、保存与 strict 加载。

**Why this priority**: 数据一致性决定训练正确性；本 Story 提供可复现的小规模端到端证明。

**Independent Test**: 小型 replay → precompute → encoded shard → collator → Actor-only SFT
（含保存/加载）完整生命周期集成测试通过；同一局面从 Observation/replay fixture、PyO3 批编码、
SFT precompute、encoded shard 解码到 collator/模型入口生成张量级一致编码；dataset
manifest/hash/schema 错误 fail closed。

**Acceptance Scenarios**:

1. **Given** 同一局面，**When** 走 Observation/replay fixture、PyO3 批编码、precompute、
   shard 解码、collator 与模型入口，**Then** 公共、Opponent Analysis 与 Action Query 编码
   张量级一致；不存在「precompute 新协议、SFT 训练仍读旧历史 token」双轨。
2. **Given** Actor-only SFT，**When** 训练，**Then** Critic/value 冻结且无梯度；保存文件只含
   Actor 范围权重、精确 `model_config` 与 contract 元数据；加载 strict 校验。
3. **Given** encoded manifest/shard，**When** 校验，**Then** format/contract
   SHA256/字段 schema/segment 顺序/分隔符/RoPE 语义/模型配置全部 strict 校验，任何旧 V18
   shard/checkpoint 视为不兼容（不迁移、不覆盖、不删除物质资产）。
4. **Given** 训练入口，**When** 运行，**Then** 使用自包含配置（含 `d_model`、16Q/4KV、
   `dense_slot_dim=32`、`dense_fusion_dim=512`、层数、FFN、RoPE base、最终 context 上限），
   无 overlay/隐式继承；SFT 节奏键仍由 `sft/contract.py` 单点定义。

---

### User Story 6 - 清理、文档与审计收敛 (Priority: P2)

作为仓库维护者，我需要本阶段范围内的旧 V18 活跃实现清理干净、文档与代码一致、PPO 待迁移
引用单独盘点、无临时产物残留。

**Why this priority**: 保证「全仓只保留一个活跃契约」与可追溯性。

**Independent Test**: 全仓 `rg` 旧契约引用（history_factors/Atomic Snapshot/54 行/
`_isolated_action_layout`/旧局部 position 等），活跃路径零命中；PPO 专用引用分类为后续待迁移项；
审计记录含 schema hash、参数量、token 统计、性能、测试命令与结果。

**Acceptance Scenarios**:

1. **Given** 删除任何一个旧模块/文件前，**When** 执行全仓 `rg` 引用检查，**Then** 零引用且
   测试通过后才删除；模块若仍承担动作解码/生命周期等独立职责则拆分职责而非误删。
2. **Given** 文档，**When** 同步，**Then** `riichi_ppo_v1/docs/v18_input_protocol.md` 重写、
   `KyokuEventTupleProtocol.md` 明确事件只用于同步不再作为模型输入、模型/SFT/encoded
   manifest 文档与配置更新、根 README/`riichi_ppo_v1` README/`AGENTS.md`/目录职责更新；
   PPO 专用文档仅加「待迁移、当前不适用新 V18 输入」标记。
3. **Given** 验证结果，**When** 记录，**Then** `audit/reports/v18/report/PROGRESS.md` 包括
   改动范围、schema hash、参数量、token 统计、性能、测试命令与结果；冒烟测试临时日志与结果
   清理干净。
4. **Given** PPO 路径，**When** 盘点，**Then** 对旧输入契约的依赖只做引用盘点并记录为后续
   待迁移项，本阶段不清理、不兼容、不测试。

## Functional Requirements

- **FR-01** 移除 Actor 完整 MJAI 事件历史输入：`history_factors/history_numeric/
  history_lengths/history_generations` 活跃模型契约、append-only event token cache 及其在
  数据预处理/模型/SFT 中的依赖全部移除；`new_events()` 仅用于同步/生命周期/奖励/动作执行。
- **FR-02** 移除固定 54 行 Atomic Snapshot 作为独立输入块，以及与新类别重复的旧 opponent
  summary、旧 causal-history position/mask 假设、旧 Query pair 共享局部 position ID 协议。
- **FR-03** 共享公共前缀按提示词 §四 的规范序列排列；分隔符集合单点定义并 fail closed。
- **FR-04** 桌况 token 保留：场风、局数、本场数、立直棒数、庄家/观察者自风、四家点数、观察者
  名次、相对三家的点差、所有公开宝牌指示牌（保留重复倍率）、决策模式（主动/响应舍牌/响应加杠）、
  自己当前摸牌、自己立直状态。
- **FR-05** 桌况明确不输入：`tiles_left`、精确/估计剩余活牌数、独立「当前供牌者」「当前供出牌」
  公共 token、一发/流局满贯/双立直/海底/河底/岭上/抢杠等独立低频状态、完整历史事件与
  generation/cache 标识。
- **FR-06** 自身手牌按 34 牌序以当前非零牌种 token 表达：牌种、张数、是否含赤五、是否当前摸牌、
  当前立直下是否锁定；SELF_STATE_ANALYSIS 密集 token 含门清/暗牌数/副露数/overall·standard·
  chiitoitsu·kokushi shanten/进张牌种类数与剩余实体数/等待牌种类数与剩余实体数/永久·同巡·立直振听/
  当前确定基础番与自身手牌副露中宝牌赤牌数。
- **FR-07** 四家 player token 每家一个：相对座次、自风、是否庄家、点数、名次、相对观察者点差、
  暗牌数、副露数、杠数、是否门清、牌河长度、立直状态/巡目/宣言牌/立直后舍牌数、副露中确定役牌番
  与可见宝牌/赤牌数；不含预测性/人工威胁信息。
- **FR-08** 自己牌河不生成逐张 token；其信息无损转移到 34 tile-state 的自己舍牌张数/是否舍过、
  SELF_PLAYER 牌河长度与立直宣言信息、SELF_STATE_ANALYSIS 精确振听类型。
- **FR-09** 三家对手牌河逐张保留（按相对座次、内部由旧到新），每张 token 含：对手相对座次、
  本地 river index、牌种与赤牌、手切/摸切、立直前/宣言/立直后三态、是否作为吃碰杠供牌、
  相对该玩家最新舍牌的年龄桶。
- **FR-10** 每家对手固定 FIRST_SIX_SUMMARY 与 RECENT_SIX_SUMMARY：内部 6 槽位顺序保留
  （含牌种/赤牌、手切/摸切、立直阶段），槽位用固定 slot-id embedding 区分 1..6，padding
  不混淆、带有效长度。
- **FR-11** 每个当前副露一个 token：拥有者、chi/pon/daiminkan/ankan/kakan、完整构成牌与赤牌、
  被鸣牌、供牌者、开放/暗置、当前副露序号、确定役牌番与可见宝牌/赤牌数；只表达当前形态，
  加杠不重复历史碰。
- **FR-12** 固定 34 个 tile-state token，按 34 牌序：自己暗手张数、自己舍牌张数/是否舍过、
  总公开张数、含自身手牌后总已知张数、未知实体牌数、是否四张全见、是否宝牌及倍率、是否
  场风/自风/赤五对应牌种、是否当前进张、是否当前和牌、对三家分别是否现物、对三家筋类别
  （非筋/单侧筋/双侧筋/不适用）、壁类别（无壁/one-chance/no-chance）、是否宝牌邻张；
  不含人工危险分数或隐藏手牌监督回填概率。
- **FR-13** 三个 Actor-only Opponent Analysis token：每名对手相对座次、立直状态/巡目/宣言牌、
  门清/开放、暗牌数/副露数/杠数、副露确定役牌番与可见宝牌/赤牌数、立直后手切数/摸切数、
  最近六张手切数/摸切数、自己手中对该家现物牌种数/实体牌数、对手牌河长度；位于 Actor 分支、
  Action Query 之前，参与 action logits；由预计算确定槽位形成（保持单 Actor 决策层，不得靠
  token 排列假装层内顺序）。
- **FR-14** 每个合法动作两个 Query token，按 action ID 升序先 Offense 后 Defense；共享
  metadata 含 action ID、动作类型、主牌、完整 consume 组合、供牌相对座次、是否打出当前摸牌；
  需要 supplier 的 chi/pon/daiminkan/ron 为 1..3，其余 N/A；不新增独立公共供牌 token。
- **FR-15** 所有 token（含 BOS、分隔符、摘要、Analysis、Query、Critic 私有与 Value Query）
  应用 RoPE，position ID 连续唯一；分隔符占位、分支处续接（Actor/Critic 各自从 P 继续）；
  动作对不再共享局部 position ID；每个 token 仍有 token-type/segment/category/relative-seat
  内容 embedding。
- **FR-16** 公共快照使用有效公共 token 之间双向 GQA；Actor 结构化 mask 按 §6.4；Critic mask
  按 §6.5（Critic 只读 Shared 表示、三家真实闭手、未来五张与 Value Query）。
- **FR-17** 密集类别使用槽位感知非线性融合：每离散槽位独立 embedding 表、`dense_slot_dim=32`、
  按规范顺序 concat、numeric 稳定归一化后专属投影、`dense_fusion_dim=512`、
  RMSNorm+gated/SiLU MLP+投影回 `d_model=256`；简单单事实 token 可继续轻量 factor embedding。
- **FR-18** 生产拓扑 `d_model=256`、16Q/4KV GQA、`head_dim=16`、`ffn_dim=704`、3 Shared +
  1 Actor + 2 Critic、RMSNorm/RoPE/gated FFN；不保留 GQA/MHA 双分支或实验 flag；总参数 ≤6.0M，
  超限先减少重复融合层或共享合理组件（不得降 d_model/层数/删输入信息）。
- **FR-19** Actor/Critic 信息边界：Actor 可读自身手牌与私有振听、公开桌况/三家完整牌河/四家
  副露/公开宝牌指示、仅由合法可见信息计算的 tile-state 与 Opponent Analysis、合法动作
  Offense/Defense Query；Actor 不得读对手闭手/真实摸牌/真实牌山/里宝/事后标签/Critic token；
  Critic 私有输入严格为三家真实闭手（固定相对座次顺序）与未来五张牌（固定摸牌顺序）。
- **FR-20** 编码一致性：同一局面从 Observation/replay fixture、PyO3 批编码、SFT precompute、
  encoded shard 解码到 collator/模型入口产生字节/张量级一致的公共、Opponent Analysis 与
  Action Query 编码；编码直接以 Observation 当前字段（hands、discards、tsumogiri_flags、
  melds、scores、riichi 状态、dora indicators、合法动作等）构造快照。
- **FR-21** 排序与上下文：动作对按 action ID 升序；三家对手恒按下家/对面/上家；牌河内部按本地
  river index 升序；34 tile-state 按固定 34 牌序；所有 separator/BOS/有效 token 计入 length；
  不允许静默截断；目标 `context_tokens=256`（超界用下一个固定值并解释）。
- **FR-22** encoded manifest 继续使用 V18 版本标识，contract SHA256、字段 schema、segment
  顺序、分隔符、RoPE/position 语义、模型配置与数据格式全部更新并 strict 校验；旧 V18 encoded
  shard/checkpoint 不提供迁移。
- **FR-23** SFT 全链路：Actor-only SFT 前向、BC loss、反向、optimizer step、保存与 strict 加载；
  Critic/value 冻结且无梯度；小规模 replay→precompute→shard→collator→训练生命周期集成测试；
  dataset manifest/hash/schema 错误 fail closed。
- **FR-24** 本阶段只生成最小 fixture/抽样 shard/临时预计算用于测试并清理；不生成完整
  `datasets/tenhou_sft_2024_2025_encoded_60pct_v18`；不修改/迁移 PPO 的 worker、rollout
  buffer、learner、推理/采样、训练配置、checkpoint/resume、1v3 评测与性能基线；不恢复
  Q scorer/Q boosting；不保留旧协议 adapter。

## Success Criteria

- **SC-01** 现行 V18 输入不含事件历史、`tiles_left`、独立当前供牌 token 或 54 行 Snapshot
  （全仓活跃路径 `rg` 零命中）。
- **SC-02** 三家完整牌河、每家两个六张摘要、当前副露、34 tile-state 编码正确（真实 replay +
  合成边界 fixture 测试通过）。
- **SC-03** 所有类别分隔符与所有有效 token 应用 RoPE；公共双向 mask、Actor 结构化 mask、
  Critic 隔离 mask 测试通过；打乱环境合法动作顺序经规范排序后张量与 logits 不变。
- **SC-04** 三个 Opponent Analysis 位于 Actor-only 尾部、Action Query 之前并影响 action
  logits；Critic 不接收 Analysis/Action，私有输入严格为三家闭手 + 未来五张（信息隔离测试通过）。
- **SC-05** Offense/Defense O0–O9/D0–D9 保留；动作排序与 pair 隔离正确；raw logits 与 action
  ID 映射唯一，非法动作严格 `-inf`。
- **SC-06** 生产模型 `d_model=256`、16Q/4KV GQA、密集槽位非线性融合、无 MHA 双分支；统一参数
  统计报告 embedding/shared/actor/critic/head 分项，总参数 ≤ 6.0M。
- **SC-07** `context_tokens=256` 理论/实测不截断；在既定 60% selection 上给出 public/Actor/
  Critic 的 mean/p50/p95/p99/max 与 segment 贡献统计（不落完整数据集）。
- **SC-08** Observation/replay、PyO3、SFT precompute、shard、collator 与模型入口编码一致
  （张量级一致性测试通过）。
- **SC-09** Actor-only SFT、Actor/Critic 隔离、RoPE/mask/schema/回放/集成/性能测试通过；
  总参数、dense 槽位敏感性、padding/梯度/尺度测试通过。
- **SC-10** 全仓一致性复核确认活跃路径无旧契约引用、PPO 待迁移引用已分类记录、无未清理临时
  产物；`PROGRESS.md` 含 schema hash/参数量/token 统计/性能/测试命令与结果。
- **SC-11** 未生成完整 V18 数据集、未启动正式长时 SFT/GRP/PPO、未删除 V16/V17 归档资产。

## Key Entities

- `CurrentStateBatch`：一次 PyO3 批编码的产物（共享+Analysis 序列 token rows/numeric/lengths、
  可选的 critic rows/numeric/lengths）。
- `EncodedProtocolRow`：单 token 的固定宽度行布局（segment、token_kind、按类别字段、numeric）。
- `EncodedSample`：SFT 样本（actor 序列、action_ids、legal_mask、监督动作、身份字段）。
- `DenseTokenSchema`：密集类别字段/基数/数值槽位的单一来源（Python 消费，Rust 编码器镜像）。
- `StateSnapshotModel`：新 `KyokuTransformerActorCritic` 前向契约（actor 序列 + query 元数据 +
  critic 私有 + legal mask）。

## Edge Cases

- 牌河长度为 0、1、5、6、7、25（上限）；摘要不足六张与恰好六张、最近六张与首六张重叠。
- 副露为 0/1/2/3/4 个；kakan 与历史 pon 并存时只表达 kakan；暗杠 opened=false；连风役牌番。
- 34 种牌四种全见/未知数为 0；one-chance/no-chance 边界；宝牌重复指示倍率；赤五在自身手牌/
  牌河/副露/未来五张中的保留。
- 立直未宣言/宣言等待受理/已受理；宣言牌在牌河中；立直后手切/摸切计数；双立直不单独编码。
- 决策模式：主动舍牌（有摸牌）、响应舍牌（无摸牌）、响应加杠；`drawn_tile=None` 与 `last_discard`
  的区分。
- supplier 动作相对座次 1..3 且必须与最后供牌者一致；非 supplier 动作必须 N/A；自摸/荣和编码区分。
- 动作集合为空（不应出现）与非规范顺序输入；重复 action ID；legal mask 与 query 集合不一致。
- 上下文严格上界：最晚巡、三家最长牌河、最多副露、最多合法动作、Critic 私有段；若组合超
  256 用下一个固定值并解释，不截断。
- 旧 checkpoint/encoded shard 与新契约不兼容：加载必须 fail closed，不迁移、不覆盖、不删除
  物质资产。
- 信息隔离：三家闭手/未来牌山改变不影响 Actor；Analysis/Action token 不进入 Critic shape。
- 性能：CPU/Rust/PyO3 编码吞吐与分段耗时、GPU Actor-only SFT 前向/反向/显存/tokens/s 测量；
  冒烟测试后清理日志与结果文件。

## Assumptions

- 版本仍为 V18（不新增 V19）；V16/V17 资产只读归档。
- PPO 阶段（worker、rollout buffer、learner、推理/采样、训练配置、checkpoint/resume、1v3
  评测、性能基线、GRP 训练）不在本阶段范围；旧 PPO 调用方对新输入暂时不兼容，仅记录为后续
  待迁移项，允许为保持包级 import/通用 schema 编译正常作必要的机械性修正。
- `context_tokens=256` 为提示词钦定目标；若严格上界确实超过则改用下一个有充分余量的固定值。
- 不修改宪法（Actor 公开信息边界与 Critic 私有输入保持提示词定义）；若调查发现需求与宪法冲突，
  停止冲突部分并报告具体条款。
- SFT 验证/checkpoint 每 3000 steps、最终 96 半庄节奏不变；PPO 1v3 机制常量不动。
- 编码数据路径 `datasets/tenhou_sft_2024_2025_encoded_60pct_v18` 保持为后续正式路径，但本阶段
  不生成完整数据。
- 注释与新增文档使用中文；删除前全仓 `rg`；冒烟测试临时产物清理。
