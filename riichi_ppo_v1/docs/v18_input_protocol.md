# V18 输入协议（encoding protocol 18，当前局面快照）

V18 是唯一活跃输入契约。Actor 输入为**决策时刻的状态快照序列**：Shared 公共前缀
（桌况 / 自身手牌 / SELF_STATE_ANALYSIS / 四家 PLAYER / 三家完整牌河与两个六张摘要 /
当前副露 / 34 tile-state）+ Actor-only 尾部（三个 Opponent Analysis + 按 action ID
升序的 Offense/Defense Query）。Critic 在共享公共表示之后单独读取三家真实闭手与未来
五张牌。V16/V17 文档与产物仅作冷存储；PPO/rollout 与 `riichi_lab_bot` 的旧输入引用已
盘点为 V18 后续待迁移项。

## 1. 张量契约

| 张量 | 形状 | 语义 |
| --- | --- | --- |
| `actor_factors` | `[B,T,32] int64` | 完整 Actor 序列行：`[segment, kind, fields...]` |
| `actor_numeric` | `[B,T,8] float32` | 数值槽位（桌况分数/点差、player 点数/点差），归一化到 [-1,1] |
| `actor_lengths` | `[B] int64` | 每行有效 token 数（含 BOS 与分隔符），≤256 |
| `query_rows` | `[B,2Q,15] int64` | 每动作 Offense/Defense 连续两行（同旧语义） |
| `query_action_ids` | `[B,Q] int64` | 升序唯一，与 legal_mask 集合相等 |
| `query_pair_counts` | `[B] int64` | 每行合法动作数 |
| `legal_mask` | `[B,241] bool` | 动作 ID 集合 |
| `critic_factors` | `[B,C,32] int64` | SEP_CRITIC + 三家闭手 + 未来五张 |
| `critic_lengths` | `[B] int64` | Critic 私有行长度 |

segment/kind/sepparator/字段 schema 只在 `model/encoding_protocol.py` 定义一次；
Rust 编码器 `RiichiEnv/riichienv-python/src/current_state_encoding.rs` 镜像同一常量。
契约 SHA256 由 schema + query 槽位 + 动作空间生成，任何变化都使旧 encoded
shard/checkpoint fail closed。

## 2. 规范序列

```text
[BOS]
[TABLE]
[SEP_SELF_HAND]
SELF_HAND × K            # 34 牌序、非零牌种
[SELF_STATE_ANALYSIS]
[SEP_PLAYERS]
SELF_PLAYER / SHIMOCHA_PLAYER / TOIMEN_PLAYER / KAMICHA_PLAYER
[SEP_RIVERS]
[SEP_SHIMOCHA_RIVER]  FIRST_SIX_SUMMARY  DISCARDS…  RECENT_SIX_SUMMARY
[SEP_TOIMEN_RIVER]    …（同上）
[SEP_KAMICHA_RIVER]   …（同上）
[SEP_MELDS]
MELD × M               # 拥有者 SELF→SHIMO→TOI→KAMI，同家按 meld_index 升序
[SEP_TILE_STATE]
TILE_STATE × 34        # 34 牌序
[SEP_OPPONENT_ANALYSIS]
OPPONENT_ANALYSIS × 3   # SHIMOCHA, TOIMEN, KAMICHA（Actor-only）
[SEP_ACTIONS]
ACTION_OFFENSE_QUERY / ACTION_DEFENSE_QUERY ×(2 per action)   # action_id 升序
```

相对座次：0=自身，1=下家，2=对面，3=上家。没有**自身逐张牌河 token**、没有
`tiles_left`、没有独立当前供牌公共字段、没有 MJAI 历史事件 token，也没有 54 行
Atomic Snapshot；自己舍牌计数、立直宣言信息与振听语义由 tile-state / SELF_PLAYER /
SELF_STATE_ANALYSIS 无损承载。

## 3. 关键类别字段（摘要）

- **TABLE**：场风/局数/本场/立直棒/庄家/自座/决策模式（主动舍牌、响应舍牌、响应加杠）/
  当前摸牌（type/red/is_current）/ 自己立直状态 / 5 个宝牌指示牌槽（保留重复倍率）/
  自己名次 + 四家分数与三家点差（numeric）。
- **SELF_STATE_ANALYSIS**：门清、暗牌数、副露数、overall/standard/chiitoi/kokushi shanten、
  进张牌种数与剩余实体数、等待牌种数与剩余实体数、永久/同巡/立直振听、自身手牌副露中的
  宝牌/赤牌数与确定基础番。
- **PLAYER**：相对/绝对座次、自风、是否庄家、名次、点数/点差（numeric）、暗牌数、副露数、
  杠数、是否门清、牌河长度、立直状态/巡目/宣言牌/立直后舍牌数、副露确定役牌番与可见
  宝牌赤牌数。
- **RIVER_DISCARD**：相对座次、本地 river index、牌种/赤牌、手切/摸切、立直前/宣言/立直后
  三态、是否作为吃碰杠供牌、相对最新舍牌的年龄桶。每对手前后各一个六槽摘要
  （FIRST_SIX / RECENT_SIX，槽位顺序保留、有效长度显式、padding 严格零贡献）。
- **MELD**：拥有者、chi/pon/daiminkan/ankan/kakan、完整构成牌与赤牌、被鸣牌、供牌相对座次、
  开放/暗置、当前副露序号、确定役牌番与可见宝牌赤牌数（kakan 不重复历史 pon）。
- **TILE_STATE**（34 个）：自己暗手/舍牌张数、总公开/已知/未知/四张全见、宝牌倍率、
  场风/自风/赤五对应、当前进张/和牌、对三家现物、对三家筋类别（非筋/单侧/双侧/不适用）、
  壁类别（无壁/one-chance/no-chance）、宝牌邻张。
- **OPPONENT_ANALYSIS**：立直状态/巡目/宣言牌、门清、暗牌数、副露数、杠数、确定役牌番与
  可见宝牌赤牌数、立直后手切/摸切数、最近六张手切/摸切数、自己手中对该家现物牌种/实体数、
  牌河长度。三个 Analysis 位于 Actor 分支、Action Query 之前，参与 action logits。
- **ACTION_QUERY**：O0–O9 / D0–D9 语义不变；chi/pon/daiminkan/ron 的 supplier 相对座次
  为 1..3，其余为 N/A。

## 4. RoPE、分隔符与结构化注意力

- 所有有效 token（含 BOS、分隔符、摘要、Analysis、Query、Critic 私有与 Value Query）
  使用连续唯一单调递增 position ID；Shared 前缀 `0..P-1`，Actor 尾部从 P 续接，
  Critic 分支从 P 续接；分隔符占位，不在类别边界重置。动作对不再复用局部 position ID。
- 分隔符为类别专属 learned separator，集合单点定义，顺序严格校验 fail closed。
- Shared 公共 backbone 使用**双向 GQA**（不作 causal）；
  Actor-only 层：Shared 只读 Shared；三个 Analysis 读 Shared ∪ Analysis；每个 Action
  Query 读 Shared ∪ Analysis ∪ 自己动作对；不同动作对互不可见；padding 不可见。
- Critic 分支读 Shared 表示 + 三家闭手 + 未来五张 + Value Query（Value 可读全部）；
  Analysis/Action token 不进入 Critic shape/length/embedding/mask。

## 5. 模型与持久化

固定拓扑：`d_model=256`、16 Q heads / 4 KV heads（GQA）、`head_dim=16`、`ffn_dim=704`、
3 Shared + 1 Actor + 2 Critic 层，`dense_slot_dim=32`、`dense_fusion_dim=512`，
`context_tokens=256`；RMSNorm/RoPE/gated FFN。密集类别使用槽位独立 embedding 表 +
共享输入投影（512）+ 共享 gated MLP；总参数 ≤6.0M（当前约 5.76M）。无 MHA 双分支、
无 Q scorer/Q boost、无 history/54 行 Snapshot adapter。checkpoint 只接受 V18
`current_state_snapshot` 配置与精确 state keys；Actor-only SFT artifact 仅保存 Actor
范围参数并 strict load，不提供旧版本兼容。

encoded manifest format 为 `riichi-sft-encoded-v18`，含 `state_protocol=
riichi-current-state-v18-1`；协议 hash、token 行宽、字段 schema、RoPE/mask 语义与信息
隔离均由生产校验器与测试共同验证。
