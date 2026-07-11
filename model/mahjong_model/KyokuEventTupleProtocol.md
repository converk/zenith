# KyokuEventTuple V3 协议

当前实现：`riichi/src/MjaiKyokuStateMachine/`。

事件语义参考 Mortal/libriichi 的 `mjai::Event` 与 `PlayerState`，但本仓库不再保存
vendored `libriichi` 源码；状态机以本仓库 Rust 实现为准。

## 1. 目标和边界

本协议定义 Transformer encoder 的输入。一次输入只描述**一小局**：从
`start_kyoku` 开始，到该局结束为止。每位玩家持有一条只追加的序列；旧 token 永远
不会因后续事件而修改或重排。每个 token 都是固定的九维离散整数元组：

```text
token = (TYPE, ACTOR, TARGET, TILE, TILE2, TILE3, VALUE, FLAG, STEP)
```

本版本的目标是：状态机能接收 `libriichi::mjai::Event` 的每一种事件；其中与当前小局
有关的事件能够转换为模型 token，`start_game/end_game` 则由牌桌管理器处理，不进入模型
输入。`PlayerState` 中每一项玩家可见的游戏事实都能由初始化前缀、追加事件，或少量
追加状态增量共同表达。

这不是 `PlayerState` 内存布局的逐字段序列化。以下项目是由状态机为规则计算而保存的
缓存或决策结果，不作为模型输入事实：

- `ActionCandidate`、`ankan_candidates`、`kakan_candidates`、`forbidden_tiles`：由环境
  转成 action mask。
- `shanten`、`waits`、`next_shanten_discards`、`keep_shanten_discards`：由手牌和规则
  推导，模型不直接接收人工计算结果。
- `dora_factor`、`doras_owned`、`doras_seen`、`is_menzen`、`kans_on_board`、
  `rank`：由已编码的宝牌、手牌、副露、分数和历史推导；其中 `STATE_RANK` 仅作为可选
  冗余快照。

`Event::None`、`can_act`、`request_action`、`possible_actions`、`action_ack` 不是已经
确认的局内事实。前者若出现在牌谱中可编码为 `EVENT_NONE`，其余只由环境控制，不进入
输入序列。RiichiEnv/RiichiLab 的 `request_action.possible_actions` 由状态机单独转换为
模型 action mask，模型响应时再回显对应 `request_id`。

## 2. 输入序列、追加规则和可见性

```text
[EVENT_START_KYOKU]
[INIT tokens ...]
[SEP]
[EVENT / STATE_DELTA tokens ...]
```

- `start_kyoku` 到来时，状态机清空该玩家上一小局的序列，依固定顺序追加
  `EVENT_START_KYOKU`、初始局面 token 和 `SEP`。这是模型序列的唯一开头。
- 后续每个 MJAI 增量事件都只在序列尾部追加一个 event chunk。一个 chunk 可以包含主
  事件 token、杠牌 continuation token，以及该事件导致的私有状态增量 token。
- 例如第二张宝牌指示牌出现时只追加 `EVENT_DORA`；绝不能回头修改第一张
  `STATE_DORA`，也不能重新生成全部宝牌状态。
- 普通流程不重复追加 `STATE_HAND`、`STATE_MELD`、`STATE_RIVER_TILE` 等“当前快照”。
  它们由初始化前缀和事件重放得到。只有将来发生历史压缩时，才允许在尾部追加完整的
  checkpoint 区块；checkpoint 仍然不能改写旧 token。
- `start_game`、`end_kyoku`、`end_game` 由每张牌桌的管理器消费。它们可写入离线牌谱，
  但不属于任何一次模型决策的输入前缀。
- 输入视角固定为一个玩家，但起手牌可由状态机的 `reveal_opponent_initial_hands` 配置
  选择两种模式：
  - **特权训练模式（当前默认）**：`start_kyoku.tehais` 的四家真实起手牌都编码为
    `STATE_HAND`，并以相对座位标记归属。这只用于训练实验，不能代表真实玩家可见信息。
  - **真实玩家模式**：只编码 `SELF` 的真实 13 张牌；每名对手各追加一个
    `STATE_HAND(actor=对手, tile=UNKNOWN, value=13)`，表达“该玩家有 13 张未知闭合牌”。
- 无论模式如何，起手牌 token 都是 append-only 初始化历史；后续敌方摸牌仍必须为
  `UNKNOWN`，不能利用离线牌谱泄漏其私有牌。
- `tsumo` 只有 `SELF` 的 `pai` 可见；其他玩家统一编码为 `UNKNOWN`，即使离线牌谱
  文件恰好带有真实牌值，也不得泄漏给模型。
- 事件历史默认保留本局完整历史。若模型将来需要截断历史，必须额外在尾部追加
  `STATE_SEEN_TILE`、`STATE_RIVER_TILE`、`STATE_MELD` 等 checkpoint token；不能静默
  丢弃早期牌河。
- 每个事件映射可产生一至多个连续 token。例如杠牌有四张牌而 token 只有三个牌槽，
  使用紧随其后的 continuation token 保存剩余牌。

## 3. 九个维度

| 维度 | Key | 含义 |
|---|---|---|
| 1 | `TYPE` | 状态、事件、连续 token 或特殊 token 的类型 |
| 2 | `ACTOR` | 状态所属玩家或事件执行者 |
| 3 | `TARGET` | 鸣牌来源、和牌放铳者或事件目标 |
| 4 | `TILE` | 主牌 |
| 5 | `TILE2` | 关联牌 1 |
| 6 | `TILE3` | 关联牌 2 |
| 7 | `VALUE` | 计数、序号或分数/点差桶 |
| 8 | `FLAG` | 单个定性属性；复合属性使用多个 token 表达 |
| 9 | `STEP` | 当前巡目桶；事件区的真实顺序仍以序列位置为准 |

未使用的玩家、牌、数值和标志必须分别填 `NONE`、`NONE`、`VALUE_NONE`、
`FLAG_NONE`。

## 4. 相对座位

`ACTOR` 和 `TARGET` 共用下表。绝对座位不能直接写入 token。

| ID | Key | 含义 |
|---:|---|---|
| 0 | `NONE` | 不绑定玩家 |
| 1 | `SELF` | 当前模型视角玩家 |
| 2 | `SHIMOCHA` | 下家 |
| 3 | `TOIMEN` | 对家 |
| 4 | `KAMICHA` | 上家 |

先计算：

```text
delta = (absolute_seat - self_seat) mod 4
```

再由状态机按实际座位编号方向转换：`delta=0` 必须映射为 `SELF(1)`，绝不能写成
`NONE(0)`。`delta=1/2/3` 分别映射为哪个相对方位，必须与环境采用的绝对座位方向保持
一致，并由单元测试覆盖。

当前 `libriichi` 是固定四麻实现；三麻的 `TOIMEN` 空位、拔北和规则差异属于未来扩展，
不应假装已被当前 Rust 状态机支持。

## 5. 牌枚举

`TILE`、`TILE2`、`TILE3` 使用同一套 39 项枚举：

```text
0 NONE
1..9   1m..9m
10..18 1p..9p
19..27 1s..9s
28 E, 29 S, 30 W, 31 N, 32 P, 33 F, 34 C
35 5mr, 36 5pr, 37 5sr
38 UNKNOWN
```

普通五和赤五必须保持不同的牌值。对于 `STATE_HAND` 与 `STATE_SEEN_TILE`，`5m` 的
计数只表示普通五万，`5mr` 单独表示赤五万；筒、索同理。

## 6. VALUE 和 STEP

`VALUE` 使用小型 bucket，数值 0 与未使用必须可区分：

| ID | Key | 含义 |
|---:|---|---|
| 0 | `VALUE_NONE` | 未使用 |
| 1..17 | `VALUE_0..VALUE_16` | 精确值 0..16 |
| 18 | `VALUE_17_PLUS` | 17 及以上 |

使用规则：

- 手牌、已见牌计数：精确 `VALUE_0..VALUE_4`。
- 宝牌指示牌序号：从 `VALUE_0` 开始。
- 本场、供托、局号、巡目：超过 17 截断。
- 分数：`floor(score / 5000)` 后截断；点差：`floor(abs(delta) / 1000)` 后截断。
- `STEP_0..STEP_16, STEP_17_PLUS` 表示巡目。事件的先后顺序由 token 在序列中的位置
  保证，不能只依赖相同 `STEP` 的大小。

## 7. TYPE 枚举

| ID | Key | 类别 | 含义 |
|---:|---|---|---|
| 0 | `PAD` | 特殊 | padding |
| 1 | `SEP` | 特殊 | 状态区与事件区分隔 |
| 2 | `STATE_GAME_MODE` | 状态 | 东风/半庄等游戏模式 |
| 3 | `STATE_BAKAZE` | 状态 | 场风 |
| 4 | `STATE_JIKAZE` | 状态 | 自风 |
| 5 | `STATE_OYA` | 状态 | 当前庄家 |
| 6 | `STATE_KYOKU_INDEX` | 状态 | 当前局号 |
| 7 | `STATE_HONBA` | 状态 | 本场数 |
| 8 | `STATE_KYOTAKU` | 状态 | 供托数 |
| 9 | `STATE_SCORE` | 状态 | 各家分数桶 |
| 10 | `STATE_RANK` | 状态 | 各家当前名次，可选冗余 |
| 11 | `STATE_LEFT_TILES` | 状态 | 牌山剩余张数 |
| 12 | `STATE_DORA` | 状态 | 已公开宝牌指示牌 |
| 13 | `STATE_HAND` | 状态 | 各家起始闭合手牌计数；真实玩家模式下对手使用 `UNKNOWN x 13` |
| 14 | `STATE_DRAW` | 状态 | 自己当前尚未打出的摸牌 |
| 15 | `STATE_REACH` | 状态 | 各家立直状态 |
| 16 | `STATE_FURITEN` | 状态 | 自己当前振听状态 |
| 17 | `STATE_MELD` | 状态 | 当前副露或暗杠 |
| 18 | `STATE_MELD_CONT` | 状态 | `STATE_MELD` 的第四张牌 |
| 19 | `STATE_SEEN_TILE` | 状态 | 当前可见牌计数，可选冗余快照 |
| 20 | `STATE_RIVER_TILE` | 状态 | 各家牌河的一张弃牌，可选冗余快照 |
| 21 | `STATE_ALL_LAST` | 状态 | 是否处于 all-last 或西入终局阶段 |
| 22 | `STATE_IPPATSU` | 状态 | 各家一发状态，可选冗余 |
| 23 | `STATE_RINSHAN` | 状态 | 自己当前岭上状态，可选冗余 |
| 24 | `EVENT_NONE` | 事件 | `Event::None`；不改变状态 |
| 25 | `EVENT_START_GAME` | 事件 | `start_game` |
| 26 | `EVENT_START_KYOKU` | 事件 | `start_kyoku` |
| 27 | `EVENT_DRAW` | 事件 | `tsumo` |
| 28 | `EVENT_DISCARD` | 事件 | `dahai` |
| 29 | `EVENT_CHI` | 事件 | `chi` |
| 30 | `EVENT_PON` | 事件 | `pon` |
| 31 | `EVENT_DAIMINKAN` | 事件 | `daiminkan` |
| 32 | `EVENT_KAKAN` | 事件 | `kakan` |
| 33 | `EVENT_ANKAN` | 事件 | `ankan` |
| 34 | `EVENT_MELD_CONT` | 事件 | 杠牌的第四张或剩余 consumed 牌 |
| 35 | `EVENT_DORA` | 事件 | `dora` |
| 36 | `EVENT_REACH` | 事件 | `reach` 宣言 |
| 37 | `EVENT_REACH_ACCEPTED` | 事件 | `reach_accepted` 确认 |
| 38 | `EVENT_HORA` | 事件 | `hora` |
| 39 | `EVENT_RYUKYOKU` | 事件 | `ryukyoku` |
| 40 | `EVENT_SCORE_DELTA` | 事件 | `hora/ryukyoku.deltas` 的一项 |
| 41 | `EVENT_URA_DORA` | 事件 | `hora.ura_markers` 的一张 |
| 42 | `EVENT_END_KYOKU` | 事件 | `end_kyoku` |
| 43 | `EVENT_END_GAME` | 事件 | `end_game` |
| 44..47 | `RESERVED_TYPE_*` | 预留 | 后续扩展 |

`EVENT_NUKIDORA` 和 `EVENT_ABORTIVE_RYUKYOKU` 不在本版本枚举中：当前 `libriichi::Event`
没有这两种事件。未来真正实现三麻解析后，使用预留 TYPE 扩展。

## 8. FLAG 枚举

| ID | Key | 用途 |
|---:|---|---|
| 0 | `FLAG_NONE` | 无标志 |
| 1-2 | `GAME_4P_EAST` / `GAME_4P_SOUTH` | `STATE_GAME_MODE` |
| 3-4 | `TSUMOGIRI` / `TEDASHI` | `EVENT_DISCARD`、`STATE_RIVER_TILE` |
| 5-8 | `REACH_NONE` / `REACH_PENDING` / `REACH_NORMAL` / `REACH_DOUBLE` | `STATE_REACH` |
| 9-10 | `REACH_DECLARE` / `DOUBLE_REACH_DECLARE` | `EVENT_REACH` |
| 11-15 | `MELD_CHI` / `MELD_PON` / `MELD_DAIMINKAN` / `MELD_KAKAN` / `MELD_ANKAN` | `STATE_MELD`、杠牌 continuation |
| 16-18 | `CHI_LOW` / `CHI_MID` / `CHI_HIGH` | `EVENT_CHI` |
| 19-21 | `DELTA_POSITIVE` / `DELTA_NEGATIVE` / `DELTA_ZERO` | `EVENT_SCORE_DELTA` |
| 22-23 | `RON` / `TSUMO` | `EVENT_HORA` |
| 24-25 | `FURITEN_TRUE` / `FURITEN_FALSE` | `STATE_FURITEN` |
| 26-27 | `IPPATSU_TRUE` / `IPPATSU_FALSE` | `STATE_IPPATSU` |
| 28-29 | `RINSHAN_TRUE` / `RINSHAN_FALSE` | `STATE_RINSHAN` |
| 30-31 | `RESERVED_FLAG_*` | 后续扩展 |

一个 token 只能承载一个 `FLAG`。需要多个定性属性时，拆成多个 TYPE token，不能把两个
FLAG 强行编码成一个新组合枚举。

## 9. 初始化 token、状态增量和 checkpoint

状态机不会在每次决策重建状态区。以下是 `PlayerState` 游戏事实到 token 的映射：

| PlayerState 事实 | V3 编码 |
|---|---|
| `bakaze` / `jikaze` / `oya` / `kyoku` / `honba` / `kyotaku` | `start_kyoku` 后追加一次对应 `STATE_*` 初始化 token |
| `scores` | `start_kyoku` 后每家追加一个 `STATE_SCORE`；立直扣点由后续 `EVENT_REACH_ACCEPTED` 表达 |
| `rank` / `is_all_last` | 可选初始化 token；只在历史压缩 checkpoint 中再次出现 |
| `tiles_left` / `at_turn` | 初始 `STATE_LEFT_TILES`；后续由 `EVENT_DRAW` 和事件位置推导 |
| `dora_indicators` | 第一张为初始化 `STATE_DORA`；后续指示牌一律追加 `EVENT_DORA` |
| `tehai` + `akas_in_hand` | `start_kyoku` 后，每种数量大于 0 的牌追加一个 `STATE_HAND`。特权训练模式对四家都如此；真实玩家模式仅保留自己真实手牌，对手各用一个 `STATE_HAND(UNKNOWN, 13)`。后续由摸、切、鸣牌事件重放 |
| `last_self_tsumo` | 由最近尚未被消费的 `EVENT_DRAW(SELF, ...)` 表达，不重复生成 `STATE_DRAW` |
| `riichi_declared` + `riichi_accepted` + `is_w_riichi` | 初始可省略或追加 `STATE_REACH(false)`；后续由 `EVENT_REACH`、`EVENT_REACH_ACCEPTED` 表达 |
| `at_furiten` | 若状态变化不能由公开事件唯一恢复，在导致变化的 event chunk 末尾追加 `STATE_FURITEN` 增量 |
| `fuuro_overview` + `ankan_overview` | 普通流程由吃、碰、杠事件重放；checkpoint 中才追加 `STATE_MELD` / `STATE_MELD_CONT` |
| `tiles_seen` + `akas_seen` | 普通流程由完整历史重放；checkpoint 中追加 `STATE_SEEN_TILE` |
| `at_ippatsu` / `at_rinshan` | 普通流程由事件重放；checkpoint 中追加 `STATE_IPPATSU` / `STATE_RINSHAN` |
| `kawa`、`last_tedashis`、`riichi_sutehais`、`last_kawa_tile` | 普通流程由事件重放；checkpoint 中追加 `STATE_RIVER_TILE` |

`STATE_MELD` 不再叫 `STATE_OPEN_MELD`，因为它同时表示明副露和暗杠。其格式：

```text
(STATE_MELD, actor, target_or_NONE, tile, tile2, tile3, VALUE_NONE, MELD_*, step)
(STATE_MELD_CONT, actor, target_or_NONE, tile4, NONE, NONE, VALUE_0, MELD_*, step)  # 仅四张
```

吃、碰使用三个牌槽即可；大明杠、加杠和暗杠用 continuation 保存第四张牌，从而不丢失
赤五位置。

`STATE_RIVER_TILE` 格式：

```text
(STATE_RIVER_TILE, discarder, called_by_or_NONE, tile, NONE, NONE,
 river_index, TSUMOGIRI_or_TEDASHI, step_of_discard)
```

`TARGET` 为 `NONE` 表示该弃牌未被鸣走；否则表示鸣走该牌的玩家。副露的具体种类和
完整牌组成由 `STATE_MELD` 快照给出。这样即使事件历史截断，牌河、摸切/手切和弃牌被
鸣走的事实也不会丢失。

`STATE_SEEN_TILE`、`STATE_RIVER_TILE`、`STATE_MELD` 是为历史压缩准备的 checkpoint token。
完整历史训练时可以关闭它们；若启用 checkpoint，整个 checkpoint 区块只能追加在尾部。

## 10. MJAI Event 到 V3 的完整映射

下表以 `libriichi::mjai::Event` 为准。一个事件后若有 continuation，continuation 必须
紧邻主 token。

| MJAI Event | V3 token 映射 |
|---|---|
| `None` | `(EVENT_NONE, NONE, NONE, NONE, NONE, NONE, VALUE_NONE, FLAG_NONE, step)`；通常不写入确认历史 |
| `StartGame { names, seed }` | `EVENT_START_GAME`。`names`、`seed` 是元数据，不作为模型特征；游戏模式由 `STATE_GAME_MODE` 给出 |
| `StartKyoku { bakaze, dora_marker, kyoku, honba, kyotaku, oya, scores, tehais }` | 一个 `EVENT_START_KYOKU` 头 token，随后由第 9 节状态快照表达初始字段；`tehais` 按第 2 节的可见性模式编码为四家真实起手牌，或“自己真实、对手 UNKNOWN” |
| `Tsumo { actor, pai }` | `(EVENT_DRAW, actor, NONE, pai_if_SELF_else_UNKNOWN, NONE, NONE, VALUE_NONE, FLAG_NONE, step)` |
| `Dahai { actor, pai, tsumogiri }` | `(EVENT_DISCARD, actor, NONE, pai, NONE, NONE, VALUE_NONE, TSUMOGIRI_or_TEDASHI, step)` |
| `Chi { actor, target, pai, consumed }` | `(EVENT_CHI, actor, target, pai, consumed[0], consumed[1], VALUE_NONE, CHI_*, step)` |
| `Pon { actor, target, pai, consumed }` | `(EVENT_PON, actor, target, pai, consumed[0], consumed[1], VALUE_NONE, FLAG_NONE, step)` |
| `Daiminkan { actor, target, pai, consumed[3] }` | 主 token 存 `pai, consumed[0], consumed[1]`；紧跟 `EVENT_MELD_CONT(actor, target, consumed[2], ..., VALUE_0, MELD_DAIMINKAN, step)` |
| `Kakan { actor, pai, consumed[3] }` | 主 token 存 `pai, consumed[0], consumed[1]`；紧跟 `EVENT_MELD_CONT(actor, NONE, consumed[2], ..., VALUE_0, MELD_KAKAN, step)` |
| `Ankan { actor, consumed[4] }` | 主 token 存 `consumed[0], consumed[1], consumed[2]`；紧跟 `EVENT_MELD_CONT(actor, NONE, consumed[3], ..., VALUE_0, MELD_ANKAN, step)` |
| `Dora { dora_marker }` | `(EVENT_DORA, NONE, NONE, dora_marker, NONE, NONE, indicator_index, FLAG_NONE, step)` |
| `Reach { actor }` | `(EVENT_REACH, actor, NONE, NONE, NONE, NONE, VALUE_NONE, REACH_DECLARE_or_DOUBLE_REACH_DECLARE, step)` |
| `ReachAccepted { actor }` | `(EVENT_REACH_ACCEPTED, actor, NONE, NONE, NONE, NONE, VALUE_NONE, FLAG_NONE, step)` |
| `Hora { actor, target, deltas, ura_markers }` | `EVENT_HORA(actor, target_or_NONE_for_tsumo, NONE, ..., RON_or_TSUMO)`；每个 delta 紧跟一个 `EVENT_SCORE_DELTA`；赢家可见的每张里宝牌紧跟一个 `EVENT_URA_DORA` |
| `Ryukyoku { deltas }` | `EVENT_RYUKYOKU`，随后每个 delta 一个 `EVENT_SCORE_DELTA`；当前 Rust 事件没有流局原因字段 |
| `EndKyoku` | `EVENT_END_KYOKU` |
| `EndGame` | `EVENT_END_GAME` |

主事件 token 的固定格式如下；表中省略的字段均为 `NONE`：

```text
EVENT_START_GAME  = (EVENT_START_GAME, NONE, NONE, NONE, NONE, NONE, VALUE_NONE, FLAG_NONE, STEP_0)
EVENT_START_KYOKU = (EVENT_START_KYOKU, NONE, oya, bakaze, dora_marker, NONE, kyoku_index, FLAG_NONE, STEP_0)
EVENT_DAIMINKAN   = (EVENT_DAIMINKAN, actor, target, pai, consumed[0], consumed[1], VALUE_NONE, MELD_DAIMINKAN, step)
EVENT_KAKAN       = (EVENT_KAKAN, actor, NONE, pai, consumed[0], consumed[1], VALUE_NONE, MELD_KAKAN, step)
EVENT_ANKAN       = (EVENT_ANKAN, actor, NONE, consumed[0], consumed[1], consumed[2], VALUE_NONE, MELD_ANKAN, step)
EVENT_MELD_CONT   = (EVENT_MELD_CONT, actor, target_or_NONE, remaining_tile, NONE, NONE, VALUE_0, MELD_*, step)
EVENT_HORA        = (EVENT_HORA, winner, target_or_NONE, NONE, NONE, NONE, VALUE_NONE, RON_or_TSUMO, step)
EVENT_RYUKYOKU    = (EVENT_RYUKYOKU, NONE, NONE, NONE, NONE, NONE, VALUE_NONE, FLAG_NONE, step)
```

`Hora` 在当前 `libriichi` 事件结构中没有和了牌字段。因此 `EVENT_HORA.TILE` 必须为
`NONE`；荣和牌或自摸牌只能从紧邻的前序 `Dahai` / `Tsumo` / `Kakan` 等事件按规则推导，
不能伪造为原始字段。

`EVENT_SCORE_DELTA` 格式：

```text
(EVENT_SCORE_DELTA, affected_actor, NONE, NONE, NONE, NONE,
 delta_abs_bucket, DELTA_POSITIVE_or_NEGATIVE_or_ZERO, step)
```

它是局终牌谱信息，不会出现在一局尚未结束时的决策样本中；保留它是为了让完整牌谱也能
用同一协议表示。

## 11. 生命周期和训练样本

状态机可以在模型序列之外保存完整牌谱；但训练或推理的模型输入严格只取当前小局：

```text
[EVENT_START_KYOKU]
[初始化 token]
[SEP]
[本局内已追加的 event / state-delta token]
```

`EVENT_HORA`、`EVENT_RYUKYOKU` 可以作为当前小局的终止事件追加到离线完整序列；终止后
没有同一小局的下一次 action sample。`EVENT_START_GAME`、`EVENT_END_KYOKU`、
`EVENT_END_GAME` 是牌桌生命周期事件，不由模型序列消费。

合法动作不写入九维序列。RiichiEnv/RiichiLab 风格下，环境通过 `request_action` 单独发送
`possible_actions`；状态机只做格式转换，不自己计算麻将规则合法性：

```text
环境 MJAI event -> 状态机追加九维序列
环境 request_action.possible_actions -> 状态机生成 241 维 action mask
模型输入序列 + action mask -> logits 中采样或贪婪选择 action_id
状态机 action_id -> 带 request_id 的 MJAI response
环境执行 -> 服务器确认的 Event -> 写回事件历史和状态机
```

`action_ack` 只确认服务器接收了响应，不是小局事实；当前状态机直接忽略。

## 12. 实现和校验要求

1. Rust 状态机按绝对座位维护事件，序列生成器在输出 token 的最后一步转换成相对座位。
2. 每个 `StartKyoku` 重置该玩家的本局状态和 token 向量，并重新追加初始化前缀；PPO
   rollout 边界、环境 batch 边界都不能触发这一重置。
3. 连续 token 校验：`EVENT_MELD_CONT` 必须紧随对应杠牌事件，`STATE_MELD_CONT` 必须
   紧随对应四张面子状态，二者的 `ACTOR/TARGET/STEP/FLAG` 必须匹配。
4. 隐私校验：敌方摸牌必须为 `UNKNOWN`，不应公开的里宝牌不得进入决策前输入。若
   `reveal_opponent_initial_hands=False`，敌方初始手牌和敌方闭合手牌也必须保持未知；若为
   `True`，仅允许在 `start_kyoku` 初始化前缀中写入敌方真实起手牌，并必须将模型标记为
   特权训练模型，禁止作为真实玩家视角直接部署。
5. 一致性校验：从事件历史重放后，应与状态区的手牌、立直、分数、供托、宝牌、副露、
   剩余牌数和可见牌计数一致。
6. 精确副露校验：快照生成器必须保留每个面子的完整四张 `Tile` 和来源玩家。不能只读
   `PlayerState.fuuro_overview` / `ankan_overview`：后者对暗杠会 deaka，且概览本身不保证
   保留每个面子的来源玩家。精确资料应由原始 MJAI 事件重放后维护。
7. 状态机必须从历史维护所有玩家的双立直和一发状态；`PlayerState` 中的
   `is_w_riichi`、`at_ippatsu` 只直接服务当前视角玩家，不足以单独生成“各家”快照。
8. 当前版本仅承诺覆盖本仓库引入的四麻 `libriichi::mjai::Event`。三麻、拔北、带流局
   原因的 MJAI 扩展，必须先扩展 Rust Event 和 PlayerState，再占用预留 TYPE。
