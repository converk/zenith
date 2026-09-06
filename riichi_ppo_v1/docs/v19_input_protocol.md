# V19 输入协议（encoding protocol 19，当前局面快照 + 信念注入）

V19 是唯一活跃输入契约。Actor 输入为**决策时刻的状态快照序列**：Shared 公共前缀
（桌况 / 自身手牌 / SELF_STATE_ANALYSIS / 四家 PLAYER / 三家纯打牌序列河（被鸣牌已移除）
与每家恒发射 RIICHI_CARD / 当前副露 / 34 tile-state）+ Actor-only 尾部（三个
Opponent Analysis + 模型内部生成的 30 个信念 token + 按 action ID 升序的
Offense/Defense Query）。Critic 在共享公共表示之后单独读取三家真实闭手（V18 未来
五张牌已删除）。V18 文档与产物仅作冷存储；活跃代码不提供任何旧协议分支。

## 1. 张量契约

| 张量 | 形状 | 语义 |
| --- | --- | --- |
| `actor_factors` | `[B,T,32] int64` | 完整 Actor 序列行：`[segment, kind, fields...]`，不含信念 token |
| `actor_numeric` | `[B,T,8] float32` | 数值槽位（桌况分数/点差、player 点数/点差），归一化到 [-1,1] |
| `actor_lengths` | `[B] int64` | 每行有效 token 数（不含 30 个信念 token；模型前向内部 +30） |
| `query_rows` | `[B,2Q,15] int64` | 每动作 Offense/Defense 连续两行（旧语义） |
| `query_action_ids` | `[B,Q] int64` | 升序唯一，与 legal_mask 集合相等 |
| `query_pair_counts` | `[B] int64` | 每行合法动作数 |
| `legal_mask` | `[B,241] bool` | 动作 ID 集合 |
| `critic_factors` | `[B,C,32] int64` | SEP_CRITIC + 三家闭手（无 future 行） |
| `critic_lengths` | `[B] int64` | Critic 私有行长度 |

segment/kind/separator/字段 schema 只在 `model/encoding_protocol.py` 定义一次；
Rust 编码器 `RiichiEnv/riichienv-python/src/current_state_encoding.rs` 镜像同一常量。
契约 SHA256 由 schema + query 槽位 + 动作空间生成，任何变化都使旧 shard/checkpoint
fail closed。

## 2. 规范序列（Rust 编码器产物 + 模型内注入）

```text
[BOS]
[TABLE]
[SEP_SELF_HAND]
SELF_HAND × K            # 34 牌序、非零牌种
[SELF_STATE_ANALYSIS]
[SEP_PLAYERS]
SELF_PLAYER / SHIMOCHA_PLAYER / TOIMEN_PLAYER / KAMICHA_PLAYER
[SEP_RIVERS]
[SEP_SHIMOCHA_RIVER]  RIVER_DISCARD × m  RIICHI_CARD
[SEP_TOIMEN_RIVER]    RIVER_DISCARD × m  RIICHI_CARD
[SEP_KAMICHA_RIVER]   RIVER_DISCARD × m  RIICHI_CARD
[SEP_MELDS]
MELD × M               # 拥有者 SELF→SHIMO→TOI→KAMI，同家按 meld_index 升序
[SEP_TILE_STATE]
TILE_STATE × 34        # 34 牌序
[SEP_OPPONENT_ANALYSIS]
OPPONENT_ANALYSIS × 3   # SHIMOCHA, TOIMEN, KAMICHA（Actor-only）
[SEP_ACTIONS]
--- 模型内部插入(不经 Rust) ---
BELIEF × 30             # 三家 ×10,SEGMENT_BELIEF=5,紧跟 SEP_ACTIONS 之后
--- 编码器尾部 ---
ACTION_OFFENSE_QUERY / ACTION_DEFENSE_QUERY ×(2 per action)   # action_id 升序
```

相对座次：0=自身，1=下家，2=对面，3=上家。没有 RIVER_SUMMARY、没有被鸣锚行、
没有自身逐张牌河 token、没有 `tiles_left`、没有 MJAI 历史事件 token。

## 3. 关键类别字段（摘要）

- **TABLE**：同 V18 字段不变。
- **SELF_STATE_ANALYSIS**：同 V18 字段不变。
- **PLAYER**（纯静态卡）：相对/绝对座次、自风、是否庄家、名次、点数/点差（numeric）、
  暗牌数、副露数、杠数、是否门清。删除了 river_length、立直 5 字段与副露番/宝牌字段
  （后者由 MELD/RIICHI_CARD 完整承载）。
- **RIVER_DISCARD**：压缩河内序（只计未被鸣走的牌，1..m 连续）、牌种/赤牌、手切/摸切、
  立直前/宣言/立直后三态、相对最新舍牌的年龄桶。删除了 relative_seat、supplied。
- **RIICHI_CARD**（恒发射 ×3）：riichi_status / riichi_turn / 宣言牌 type/red /
  立直后手切数 / 立直后摸切数 / 宣言牌是否被鸣走；未立直时全零。
- **MELD**：保留 V18 全部构成字段，新增 meld_turn（被鸣牌在供牌者河中的原始下标+1）
  与 called_tsumogiri（被鸣牌是否为供牌者摸切）。
- **TILE_STATE**（34 个）：自身暗手/舍牌张数、未知枚数与四张全见（public_count/known_count
  已删，为线性可推）、宝牌倍率、场风/自风/赤五对应、进张/和牌标记、对三家现物与筋类别、
  壁类别、宝牌邻张。
- **OPPONENT_ANALYSIS**（行为统计卡）：relative_seat、最近六张手切/摸切数、全期手切数、
  自己手中对该家现物牌种/实体数、对手临时振听（见逃）标记。立直/门清/副露字段全部移出。
- **BELIEF token**：模型内部由共享表示经信念网络 + 转换矩阵生成，不代表编码器行；
  query 读信念、信念只读共享段、信念互见、分析不读信念（D32）。
- **逐动作信念读出（模型内部，60% 方案）**：每个合法动作 Query 行的
  `primary_tile_code`（第 3 列）在各家信念特征（danger/loss/wait/向听等）上取数，
  经零初始化投影后加到 `pair_hiddens`；是模型内部计算，不改变 30 个信念 token、
  mask、协议行布局或契约 hash。SFT 阶段 detach 特征，PPO 阶段不 detach。
- **ACTION_QUERY**：同 V18（15 个嵌入特征、action_id 专用 241 维表）。

## 4. RoPE、分隔符与结构化注意力

- 所有有效 token（含 BOS、分隔符、河行、立直卡、信念、Query、Critic 私有与 Value Query）
  使用连续唯一单调递增 position ID；信念 token 插在 SEP_ACTIONS 之后、第一对 Query 之前，
  因此最后一个信念 token 距第一对 Query 恒距 1。
- 分隔符为类别专属 learned separator，集合单点定义，顺序严格校验 fail closed。
- Shared 公共 backbone 使用**双向 GQA**；Actor-only 层：
  - Shared 只读 Shared；
  - 信念 token 读 Shared ∪ 信念（彼此的信念互见），不读分析、不读 Query；
  - 三个 Analysis 读 Shared ∪ Analysis（v1 不读信念）；
  - SEP_ACTIONS 独立角色；每个 Action Query 读 Shared ∪ Analysis ∪ SEP_ACTIONS ∪
    信念 ∪ 自己动作对；不同动作对互不可见；padding 不可见。
- Critic 分支读 Shared 表示 + 三家闭手 + Value Query（Value 可读全部）；
  Analysis/Action/Belief token 不进入 Critic shape/length/embedding/mask。

## 5. 模型与持久化

固定拓扑：`d_model=256`、16 Q heads / 4 KV heads（GQA）、`head_dim=16`、`ffn_dim=704`、
3 Shared + 2 Actor + 1 Critic 层（总 block 数 6 不变），`dense_slot_dim=32`、
`dense_fusion_dim=512`，`context_tokens=320`；信念分支为 1 层 FFN=512 backbone
（完整 shared_hidden + 每玩家 3 查询共 9 个）+ 五头逐查询平均 + 282→2560 三家共享
转换矩阵 + 逐动作信念读出（零初始化）。RMSNorm/RoPE/gated FFN。密集类别使用槽位独立
embedding 表 + 共享输入投影（512）+ 共享 gated MLP；总参数约 7.11M。无 MHA 双分支、
无 Q scorer/Q boost。checkpoint 只接受 V19 `current_state_snapshot` 配置与精确 state keys。

encoded manifest format 为 `riichi-sft-encoded-v19`，含 `state_protocol=
riichi-current-state-v19-1`；协议 hash、token 行宽、字段 schema、RoPE/mask 语义与信息
隔离均由生产校验器与测试共同验证。
