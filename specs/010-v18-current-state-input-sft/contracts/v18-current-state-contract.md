# V18 当前局面输入协议契约（current-state contract）

本文件是 `010-v18-current-state-input-sft` 的实施基准。Python 唯一来源为
`riichi_ppo_v1/model/encoding_protocol.py`；Rust 编码器
(`RiichiEnv/riichienv-python/src/current_state_encoding.rs`) 镜像同一常量。
任何字段增删/顺序变化必须同步两处并更新 contract SHA256 与测试。

## 0. 全局常量

| 常量 | 值 | 说明 |
|---|---|---|
| `ENCODING_PROTOCOL_VERSION` | 18 | 保持 V18（契约 hash 变化） |
| `TOKEN_ROW_WIDTH` | 32 | 每 token 行整数列数（含 segment/kind 2 列 + 30 字段） |
| `TOKEN_NUMERIC_WIDTH` | 8 | 每 token 行浮点列数 |
| `NUM_ACTIONS` | 241 | 领域常量（`model/schema.py`） |
| `TILE_KINDS` | 34 | 领域常量 |
| `CONTEXT_TOKENS` | 256 | 生产 context 上限（严格上界见 research §2.7） |
| `QUERY_ROW_WIDTH` | 15 | 动作 Query 行宽（不变） |

行布局：`row[0]=segment`，`row[1]=token_kind`，`row[2..]=该类别的字段（按契约顺序）`。
`numeric` 行与 `row` 一一对应，未用槽位必须为 0。

## 1. Segment 与 token kind

| segment | 名称 | 包含 kind |
|---|---|---|
| 1 | SHARED | 1..9, 101..108 |
| 2 | ANALYSIS | 10, 109 |
| 3 | ACTIONS | 11, 12, 110 |
| 4 | CRITIC_PRIVATE | 13, 111 |
| 5 | CRITIC_FUTURE | 14 |

| kind | 名称 | 类别 | segment |
|---|---|---|---|
| 1 | BOS | SIMPLE | 1 |
| 2 | TABLE | DENSE | 1 |
| 3 | SELF_HAND | SIMPLE | 1 |
| 4 | SELF_STATE_ANALYSIS | DENSE | 1 |
| 5 | PLAYER | DENSE | 1 |
| 6 | RIVER_SUMMARY | DENSE | 1 |
| 7 | RIVER_DISCARD | SIMPLE | 1 |
| 8 | MELD | DENSE | 1 |
| 9 | TILE_STATE | SIMPLE | 1 |
| 10 | OPPONENT_ANALYSIS | DENSE | 2 |
| 11 | ACTION_OFFENSE_QUERY | DENSE | 3 |
| 12 | ACTION_DEFENSE_QUERY | DENSE | 3 |
| 13 | CRITIC_HAND | SIMPLE | 4 |
| 14 | CRITIC_FUTURE | SIMPLE | 5 |
| 101 | SEP_SELF_HAND | SEPARATOR | 1 |
| 102 | SEP_PLAYERS | SEPARATOR | 1 |
| 103 | SEP_RIVERS | SEPARATOR | 1 |
| 104 | SEP_SHIMOCHA_RIVER | SEPARATOR | 1 |
| 105 | SEP_TOIMEN_RIVER | SEPARATOR | 1 |
| 106 | SEP_KAMICHA_RIVER | SEPARATOR | 1 |
| 107 | SEP_MELDS | SEPARATOR | 1 |
| 108 | SEP_TILE_STATE | SEPARATOR | 1 |
| 109 | SEP_OPPONENT_ANALYSIS | SEPARATOR | 2 |
| 110 | SEP_ACTIONS | SEPARATOR | 3 |
| 111 | SEP_CRITIC | SEPARATOR | 4 |

separator id（单点定义，用于分隔符 embedding 表）：SEP_SELF_HAND=1, SEP_PLAYERS=2,
SEP_RIVERS=3, SEP_SHIMOCHA_RIVER=4, SEP_TOIMEN_RIVER=5, SEP_KAMICHA_RIVER=6, SEP_MELDS=7,
SEP_TILE_STATE=8, SEP_OPPONENT_ANALYSIS=9, SEP_ACTIONS=10, SEP_CRITIC=11。
kind = 100 + separator_id。

## 2. 规范序列

### 2.1 Actor 序列（`actor_factors` 行，长度 `actor_lengths`）

```
1  BOS(kind 1)
2  TABLE(2)
3  SEP_SELF_HAND(101)
4  SELF_HAND × K(3)           # K=自身暗手非零牌种数，按 tile_type 升序
5  SELF_STATE_ANALYSIS(4)
6  SEP_PLAYERS(102)
7  PLAYER × 4(5)             # SELF, SHIMOCHA, TOIMEN, KAMICHA
8  SEP_RIVERS(103)
9  SEP_SHIMOCHA_RIVER(104)
10 RIVER_SUMMARY first_six(6)
11 RIVER_DISCARD × N_shimo(7) # 本地 index 1..N 升序（旧→新）
12 RIVER_SUMMARY recent_six(6)
13 SEP_TOIMEN_RIVER(105) → 同上
14 SEP_KAMICHA_RIVER(106) → 同上
15 SEP_MELDS(107)
16 MELD × M(8)               # 拥有者 SELF→SHIMO→TOI→KAMI，同一拥有者按 meld_index 升序
17 SEP_TILE_STATE(108)
18 TILE_STATE × 34(9)        # tile_type 1..34 升序
19 SEP_OPPONENT_ANALYSIS(109)
20 OPPONENT_ANALYSIS × 3(10) # SHIMOCHA, TOIMEN, KAMICHA
21 SEP_ACTIONS(110)
22 ACTION_OFFENSE_QUERY/ACTION_DEFENSE_QUERY ×(2 per action)  # action_id 升序，先 Offense 后 Defense
```

相对座次约定：0=自身，1=下家(SHIMOCHA)，2=对面(TOIMEN)，3=上家(KAMICHA)。

### 2.2 Critic 序列（`critic_factors` 行；模型在共享前缀后拼接）

```
1 SEP_CRITIC(111, segment 4)
2 CRITIC_HAND × H(13, segment 4)   # 相对座次 1,2,3 顺序；同座次按 tile_type 升序
3 CRITIC_FUTURE × 5(14, segment 5) # position 1..5 升序（摸牌顺序）
+ 模型自动追加 Value Query（不属于存储行）
```

## 3. 类别字段表（row[2..] 偏移）

所有“bucket”域均要求 0 为合法值（除非标注 N/A），padding 行整行必须为 0。

### 3.1 TABLE（kind 2, DENSE）

| 偏移 | 字段 | 域 |
|---|---|---|
| 2 | round_wind | 0=E,1=S,2=W,3=N |
| 3 | kyoku_index | 0..3（0 基） |
| 4 | honba_bucket | 0..19 精确, 20=20+ |
| 5 | riichi_sticks_bucket | 0..3 精确, 4=4+ |
| 6 | oya_seat | 0..3 |
| 7 | self_seat | 0..3 |
| 8 | decision_mode | 0=主动舍牌,1=响应舍牌,2=响应加杠 |
| 9 | drawn_tile_type | 0=N/A, 1..34 |
| 10 | drawn_tile_red | 0/1（仅红5为1） |
| 11 | drawn_is_current | 0/1（=mode 0） |
| 12 | self_riichi_status | 0=未宣言,1=宣言待受理,2=已受理 |
| 13..17 | dora_indicator_type_slot_1..5 | 0=N/A, 1..34（按出现顺序） |
| 18..22 | dora_indicator_red_slot_1..5 | 0/1 |
| 23 | own_rank | 1..4（同分按绝对座次稳定排序） |
| 24..28 | 保留 | 必须 0 |

numeric 偏移：0..3 = scores[0..3]/100000（clip [-1,1]）；4..6 = 观察者 vs 相对座次 1..3 的
点差/100000（clip）；7 = 保留 0。

### 3.2 SELF_HAND（kind 3, SIMPLE）

| 偏移 | 字段 | 域 |
|---|---|---|
| 2 | tile_type | 1..34 |
| 3 | count | 1..4 |
| 4 | has_red | 0/1 |
| 5 | is_drawn | 0/1（tile_type == drawn_tile_type） |
| 6 | locked_under_riichi | 0/1（riichi_accepted[self] 且 (drawn_tile 为 None 或 kind≠drawn)） |

### 3.3 SELF_STATE_ANALYSIS（kind 4, DENSE）

| 偏移 | 字段 | 域 |
|---|---|---|
| 2 | menzen | 0/1 |
| 3 | concealed_count | 0..14（自身暗手张数，不含副露） |
| 4 | meld_count | 0..4 |
| 5 | overall_shanten | 0=和牌(-1),1=0向听,…,7=6向听,8=7向听+ |
| 6 | standard_shanten | 同上 |
| 7 | chiitoitsu_shanten | 同上；开放手 N/A=9 |
| 8 | kokushi_shanten | 同上；开放手 N/A=9 |
| 9 | advance_kind_count | 0..33 精确, 34=34+ |
| 10 | advance_remaining | 0..99 精确, 100=100+ |
| 11 | wait_kind_count | 0=N/A(非听牌), 1..33, 34=34+ |
| 12 | wait_remaining | 0=N/A, 1..99, 100=100+ |
| 13 | permanent_furiten | 0/1（missed_agari_riichi） |
| 14 | doujun_furiten | 0/1（missed_agari_doujun） |
| 15 | riichi_furiten | 0/1（立直已受理且永久振听） |
| 16 | own_dora_count | 0..4 精确, 5=5+ |
| 17 | own_aka_count | 0..4 精确, 5=5+ |
| 18 | own_yakuhai_han | 0..5 精确, 6=6+ |
| 19 | base_han_total | 0..9 精确, 10=10+ |

说明：shanten 基于 13 张归一形状（暗手 counts + 暗杠 4 张 + 3×三张副露数，超 13 张时按
`kernel_shape` 从最小 type 扣减到 13）。advance 集 = 摸入后 overall shanten 下降的牌种；
wait 集 = 归一形状 already tenpai 且摸入后 shanten<0 的牌种。`advance_remaining/wait_remaining`
按 `4 - (公开可见 + 自己暗手)` 的剩余实体数求和。

### 3.4 PLAYER（kind 5, DENSE）

| 偏移 | 字段 | 域 |
|---|---|---|
| 2 | relative_seat | 0=自身,1..3 |
| 3 | absolute_seat | 0..3 |
| 4 | seat_wind | 0=E,1=S,2=W,3=N |
| 5 | is_oya | 0/1 |
| 6 | rank | 1..4 |
| 7 | concealed_count | 0..14（暗手张数；对手=13+1(有摸牌)-3×三张副露-4×杠） |
| 8 | meld_count | 0..4 |
| 9 | kan_count | 0..4 |
| 10 | menzen | 0/1 |
| 11 | river_length | 0..24 |
| 12 | riichi_status | 0=未,1=宣言,2=已受理 |
| 13 | riichi_turn | 0=N/A, 1..25, 26=26+ |
| 14 | riichi_decl_tile_type | 0=N/A, 1..34 |
| 15 | riichi_decl_red | 0/1 |
| 16 | post_riichi_discard_count | 0..15, 16=16+ |
| 17 | open_meld_yakuhai_han | 0..5, 6=6+ |
| 18 | visible_meld_dora_aka_han | 0..7, 8=8+ |

numeric：0 = 点数/100000（clip）；1 = 相对观察者点差/100000（clip）；2..7=0。

### 3.5 RIVER_SUMMARY（kind 6, DENSE，内部 6 槽）

| 偏移 | 字段 | 域 |
|---|---|---|
| 2 | valid_length | 0..6 |
| 3+4*(i-1) | slot_i tile_type | 0=N/A, 1..34 |
| 4+4*(i-1) | slot_i red | 0/1 |
| 5+4*(i-1) | slot_i cut | 0=手切,1=摸切,2=N/A |
| 6+4*(i-1) | slot_i riichi_stage | 0=立直前,1=宣言牌,2=立直后,3=N/A |

i=1..6；slot_i 有效当且仅当 i <= valid_length（first_six：discards[0..min(6,N)]；recent_six：
discards[max(0,N-6)..N]，内部由旧到新）。padding 槽全部 0。

### 3.6 RIVER_DISCARD（kind 7, SIMPLE）

| 偏移 | 字段 | 域 |
|---|---|---|
| 2 | relative_seat | 1..3 |
| 3 | river_index | 1..24（1 基，由旧到新） |
| 4 | tile_type | 1..34 |
| 5 | red | 0/1 |
| 6 | cut | 0=手切,1=摸切 |
| 7 | riichi_stage | 0=立直前,1=宣言牌,2=立直后 |
| 8 | supplied | 0/1（存在某副露 from_who==该对手且 called_tile_index 命中本河牌下标；被鸣河牌只标记恰好那一张） |
| 9 | age_bucket | 0=最新,1=1-2张前,2=3-5张前,3=6+张前 |

### 3.7 MELD（kind 8, DENSE）

| 偏移 | 字段 | 域 |
|---|---|---|
| 2 | owner_relative | 0=自身,1..3 |
| 3 | meld_type_code | 1=chi,2=pon,3=daiminkan,4=ankan,5=kakan |
| 4..11 | tile0_type/tile0_red/tile1_type/tile1_red/tile2_type/tile2_red/tile3_type/tile3_red | type 1..34 / red 0/1；tile3 允许 0=N/A（三张副露） |
| 12 | called_tile_type | 0=N/A, 1..34 |
| 13 | called_tile_red | 0/1 |
| 14 | supplier_relative | 0=N/A, 1..3 |
| 15 | open | 0/1（ankan=0） |
| 16 | meld_index | 1..4（该拥有者第几个当前副露） |
| 17 | yakuhai_han | 0..5, 6=6+ |
| 18 | visible_dora_aka_han | 0..7, 8=8+ |

### 3.8 TILE_STATE（kind 9, SIMPLE）

| 偏移 | 字段 | 域 |
|---|---|---|
| 2 | tile_type | 1..34 |
| 3 | self_concealed_count | 0..4 |
| 4 | self_discard_count | 0..4 |
| 5 | self_ever_discarded | 0/1 |
| 6 | public_count | 0..4（实体口径：四家牌河 + 公开副露 + 宝牌指示牌中该 kind 的数量，被鸣河牌只计一次（计在副露），cap 4） |
| 7 | known_count | 0..4（public + self_concealed，cap 4） |
| 8 | unknown_count | 0..4 = 4-known |
| 9 | all_seen | 0/1（unknown==0） |
| 10 | dora_multiplicity | 0..5（重复指示牌产生的倍率） |
| 11 | is_dora | 0/1（multiplicity>0） |
| 12 | round_wind_match | 0/1 |
| 13 | seat_wind_match | 0/1 |
| 14 | red_five_kind | 0/1（kind ∈ {5m,5p,5s}） |
| 15 | is_advance | 0/1（归一 13 张形状摸入后向听下降） |
| 16 | is_win | 0/1（归一形状已听且该 kind 为和牌） |
| 17..19 | genbutsu_shimo/genbutsu_toimen/genbutsu_kamicha | 0/1（该对手牌河含该 kind） |
| 20..22 | suji_shimo/suji_toimen/suji_kamicha | 0=非筋,1=单侧筋,2=双侧筋,3=不适用(字牌) |
| 23 | wall_class | 0=无壁,1=one-chance,2=no-chance（按同花色±2 内相关牌已知数≥3/≥4） |
| 24 | dora_neighbor | 0/1（同花色与任一 dora kind 差 1；字牌 0） |

### 3.9 OPPONENT_ANALYSIS（kind 10, DENSE）

| 偏移 | 字段 | 域 |
|---|---|---|
| 2 | relative_seat | 1..3 |
| 3 | riichi_status | 0=未,1=宣言,2=已受理 |
| 4 | riichi_turn | 0=N/A, 1..25, 26=26+ |
| 5 | riichi_decl_tile_type | 0=N/A, 1..34 |
| 6 | riichi_decl_red | 0/1 |
| 7 | menzen | 0/1 |
| 8 | concealed_count | 0..14 |
| 9 | meld_count | 0..4 |
| 10 | kan_count | 0..4 |
| 11 | open_meld_yakuhai_han | 0..5, 6=6+ |
| 12 | visible_meld_dora_aka_han | 0..7, 8=8+ |
| 13 | post_riichi_tedashi | 0..15, 16=16+ |
| 14 | post_riichi_tsumogiri | 0..15, 16=16+ |
| 15 | recent6_tedashi | 0..6 |
| 16 | recent6_tsumogiri | 0..6 |
| 17 | own_genbutsu_kind_count | 0..33, 34=34+ |
| 18 | own_genbutsu_entity_count | 0..99, 100=100+ |
| 19 | river_length | 0..24 |

### 3.10 ACTION_QUERY（kind 11/12, DENSE）

actor 行字段 = 15 个嵌入特征（order 即 schema 顺序）：
`fields[2]=action_type_code(1..11)、fields[3]=primary_tile_code(0=N/A,1..34)、
fields[4]=source_seat_code(0=N/A,1..3)、fields[5]=tsumogiri_mode(0=非当前摸牌,1=当前摸牌；
仅 discard id∈[1,75) 有意义，由 action_id-1 的奇偶恢复)、fields[6]=action_id(0..240)、
fields[7..16]=answer_0..answer_9`。
action_id 进入 embedding（241 维专用离散表），是**完整 consume 组合 + 赤牌身份 + tsumogiri
的规范编码**（discard：牌种+赤五+tsumogiri；chi：consume 对；pon：红/普通对；kan：牌种；
hora/ryukyoku/pass/reach 各一）；因此 consume 不再需要单独进入 token 字段。
完整的 15 宽 query 行（query_type/action_id/action_type/primary_tile/source_seat/answers）
作为 `query_rows` 存储单独保留，供一致性校验使用。

### 3.11 CRITIC_HAND（kind 13, SIMPLE）

| 偏移 | 字段 | 域 |
|---|---|---|
| 2 | relative_seat | 1..3 |
| 3 | tile_type | 1..34 |
| 4 | red | 0/1 |
| 5 | count | 1..4 |

### 3.12 CRITIC_FUTURE（kind 14, SIMPLE）

| 偏移 | 字段 | 域 |
|---|---|---|
| 2 | position | 1..5 |
| 3 | tile_type | 1..34 |
| 4 | red | 0/1 |

## 4. 分隔符与 RoPE

- 所有 separator/BOS/有效 token 进入长度与 RoPE；position ID 由模型按序列位置生成
  （0,1,2,…），不在类别边界重置；分支间：Shared 0..P-1，Actor 尾部 P..，Critic 分支独立从 0 起
  但共享前缀部分仍为 0..P-1（Critic 私有段从 P 续接）。
- 分隔符：类别专属 learned embedding（`separator_id` 索引），不得共用无类型 [SEP]。
- 每个内容 token 额外有 token-kind embedding 与 segment embedding；RoPE 不替代类别信息。

## 5. 注意力

- Shared backbone：有效公共 token（segment 1）之间双向 GQA（无因果）。
- Actor 层：
  - Shared(segment 1) ↔ Shared 双向；Shared 不可见 Analysis/Action。
  - Analysis(segment 2) → Shared ∪ Analysis（Analysis 之间全可见）。
  - SEP_ACTIONS（kind 110）为 Actions 段标记：独立角色，SEP_ACTIONS 只可读自己，
    Action 行可读 SEP_ACTIONS；SEP_ACTIONS 不并入 Analysis 的可见域。
  - Action pair i（segment 3 的第 2i,2i+1 行）→ Shared ∪ Analysis ∪ SEP_ACTIONS ∪ 本 pair；
    不同 pair 互不可见。
  - 所有有效 query 只作用有效 key；padding 行不可见。
- Critic 层：Shared ∪ Critic 全部（含 SEP_CRITIC/闭手/未来/Value Query）双向；Value Query 位于
  末尾并读取全部。

## 6. 校验与 fail-closed

1. segment/kind 序列严格等于 §2.1/§2.2；缺失、重复、错序 → 拒绝。
2. 每个 kind 的字段域按 §3 校验；padding 行整行 0；numeric 有限且在 [-1,1]。
3. 三家各恰好 1 个 first_six + 1 个 recent_six 摘要；summary valid_length ≤ 6 且与牌河长度一致；
   内部 slot 顺序与牌河一致。
4. 恰好 34 个 TILE_STATE，tile_type 1..34 各一次；known+unknown=4；public/self/discard 计数守恒。
5. 自身无 RIVER_DISCARD token；RIVER_DISCARD 只允许 relative_seat 1..3。
6. 无 `tiles_left`、无独立当前供牌字段、无事件历史 token（segment 1 仅含 §2.1 类别）。
7. action pair：action_id 升序、唯一、与 legal_mask 集合相等；每个 action 恰好 Offense+Defense；
   O/D 行 query_type 正确；chi/pon/daiminkan/ron 的 source_seat ∈ 1..3，其余 =0。
8. Critic：三家闭手各至少 1 行、相对座次 1..3 各出现；未来恰 5 行、position 1..5 升序；
   不含 Analysis/Action。
9. 上下文：actor/critic 长度 ≤ CONTEXT_TOKENS，否则拒绝（上限见 research §2.7）。

## 7. 一致性（同一局面多入口）

`Observation/Replay` → `riichienv.prepare_current_state_batch` + `riichi.encode_query_batch`
→ Python 装配 actor 行（含 separator/query 规范排序）→ `EncodedSample` → shard →
`collate_samples` → 模型，字节级一致；precompute 与在线桥使用同一 Rust 编码器。
