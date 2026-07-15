# KyokuEventTuple V3 协议

当前实现：`riichi/src/MjaiKyokuStateMachine/`。

事件语义以本仓库 Rust 状态机
`riichi/src/MjaiKyokuStateMachine/` 为准。当前交互边界采用 RiichiEnv 风格：环境广播已经
确认的 MJAI 事件；需要玩家决策时，由 `env.reset()` / `env.step(actions)` 返回的
`dict[int, Observation]` 表达，合法动作来自对应 `Observation.legal_actions()`。

## 1. 目标和边界

本协议定义 Transformer causal decoder 的输入。一次输入只描述**一小局**：从
`start_kyoku` 开始，到该局结束为止。每位玩家持有一条只追加的序列；旧 token 永远
不会因后续事件而修改或重排。每个 token 都是固定的八维离散整数元组：

```text
token = (TYPE, ACTOR, TARGET, TILE, TILE2, TILE3, VALUE, FLAG)
```

本版本的目标是：状态机能接收本仓库支持的四麻 MJAI 事件；其中与当前小局有关的确认
事件能够转换为模型 token，`start_game/end_game` 则由牌桌管理器处理，不进入模型输入。
玩家可见的游戏事实应能由初始化前缀、追加事件，或少量追加状态增量共同表达。

实际送入模型的序列由两段组成：

```text
append-only 小局事件历史 + 当前决策快照 snapshot
```

事件历史永久保存在玩家状态机中；snapshot 只在当前需要做动作的 forward 中临时拼到末尾，
不会写回历史 token。

当前 PPO 路径不使用 rollout KV cache，也不使用小局 packed sequence。每次需要决策时，
状态机返回一条已经拼好的单决策序列；PPO update 阶段直接重放这些单决策样本。

这不是环境内部状态的逐字段序列化。以下信息不作为模型输入 token：

- 合法动作候选：由 RiichiEnv `Observation.legal_actions()` 给出，状态机只转换为
  `action_mask`，不写入八维序列。
- 向听、有效牌、推荐弃牌等规则计算结果：可由环境或分析工具计算，但模型输入不直接接收
  这类人工特征。
- 宝牌数量、门前状态、杠数、巡目、名次等可由事件历史和初始化分数推导的派生量：默认不重复写入；
  若未来做历史压缩，只能作为尾部 checkpoint token 追加。

RiichiEnv 的控制信息不进入八维序列：

- `start_game.id`：用于设置单个玩家状态机的绝对座位；每局自风由该绝对座位和
  `start_kyoku.oya` 计算。
- `Observation.legal_actions()`：只用于生成 241 维 action mask 和动作转换，不进入八维序列。
- RiichiEnv 当前训练路径没有 `request_action` / `action_ack` 事件；旧 RiichiLab 风格消息
  不属于本 PPO v1 路径。
- `observation` 中的 base64 完整局面：当前状态机不解析，内部只依赖增量事件维护序列。

只有环境确认并广播的牌局事件才追加为 token。

## 2. 输入序列、追加规则和可见性

```text
[EVENT_START_KYOKU]
[INIT tokens ...]
[SEP]
[EVENT / STATE_DELTA tokens ...]
[STATE_SNAPSHOT_BEGIN]
[当前 Observation 快照 tokens ...]
[STATE_SNAPSHOT_END]
```

- `start_kyoku` 到来时，状态机清空该玩家上一小局的序列，依固定顺序追加
  `EVENT_START_KYOKU`、初始局面 token 和 `SEP`。这是模型序列的唯一开头。
- 后续每个 MJAI 增量事件都只在序列尾部追加一个 event chunk。一个 chunk 可以包含主
  事件 token、杠牌 continuation token，以及该事件导致的私有状态增量 token。
- 例如第二张宝牌指示牌出现时只追加 `EVENT_DORA`；绝不能回头修改第一张
  `STATE_DORA`，也不能重新生成全部宝牌状态。
- 普通事件流程不把当前快照永久写入历史。每次需要模型决策时，状态机从 RiichiEnv
  `Observation` 读取一组客观状态，临时追加为 snapshot token。
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

## 3. 八个维度

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

当前 Rust 状态机是固定四麻实现；三麻的 `TOIMEN` 空位、拔北和规则差异属于未来扩展，
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

## 6. VALUE

`VALUE` 使用小型 bucket，数值 0 与未使用必须可区分：

| ID | Key | 含义 |
|---:|---|---|
| 0 | `VALUE_NONE` | 未使用 |
| 1..17 | `VALUE_0..VALUE_16` | 精确值 0..16 |
| 18 | `VALUE_17_PLUS` | 17 及以上 |

使用规则：

- 手牌、已见牌计数：精确 `VALUE_0..VALUE_4`。
- 宝牌指示牌序号：初始化和 snapshot 中从 `VALUE_0` 开始；历史 `EVENT_DORA` 不维护序号。
- 本场、供托、局号：超过 17 截断。
- 分数：`floor(score / 5000)` 后截断；点差：`floor(abs(delta) / 1000)` 后截断。

## 7. TYPE 枚举

| ID | Key | 类别 | 含义 |
|---:|---|---|---|
| 0 | `PAD` | 特殊 | padding |
| 1 | `SEP` | 特殊 | 状态区与事件区分隔 |
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
| 14 | `STATE_SNAPSHOT_BEGIN` | 状态 | 当前决策快照开始 |
| 15 | `STATE_SNAPSHOT_END` | 状态 | 当前决策快照结束 |
| 16 | `STATE_SELF_ID` | 状态 | 当前玩家绝对座位 id |
| 17 | `STATE_RIICHI_STICKS` | 状态 | 当前立直棒数量 |
| 18 | `STATE_DRAWN_TILE` | 状态 | 当前玩家最近摸牌；无则 `TILE_NONE` |
| 19 | `STATE_RIICHI_DECLARED` | 状态 | 四家是否已经立直 |
| 20 | `STATE_LAST_DISCARD` | 状态 | 最近一张弃牌；无则 `TILE_NONE` |
| 21 | `STATE_LAST_TEDASHI` | 状态 | 四家最近手切牌；无则 `TILE_NONE` |
| 22..25 | `RESERVED_TYPE_*` | 预留 | 后续状态扩展；当前 Rust 实现不写入 |
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
| 42..47 | `RESERVED_TYPE_*` | 预留 | 后续扩展 |

`EVENT_NUKIDORA` 和 `EVENT_ABORTIVE_RYUKYOKU` 不在本版本枚举中：当前 Rust 状态机没有
这两种事件。未来真正实现三麻或带原因的中止流局后，使用预留 TYPE 扩展。

## 8. FLAG 枚举

| ID | Key | 用途 |
|---:|---|---|
| 0 | `FLAG_NONE` | 无标志 |
| 3-4 | `TSUMOGIRI` / `TEDASHI` | `EVENT_DISCARD`、`STATE_RIVER_TILE` |
| 5-8 | `REACH_NONE` / `REACH_PENDING` / `REACH_NORMAL` / `REACH_DOUBLE` | `STATE_REACH` |
| 9 | `REACH_DECLARE` | `EVENT_REACH` |
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

状态机不会在每次决策重建状态区。以下是游戏事实到 token 的映射：

| 游戏事实 | V3 编码 |
|---|---|
| `bakaze` / `jikaze` / `oya` / `kyoku` / `honba` / `kyotaku` | `start_kyoku` 后追加一次对应 `STATE_*` 初始化 token |
| `scores` | `start_kyoku` 后每家追加一个 `STATE_SCORE`；立直扣点由后续 `EVENT_REACH_ACCEPTED` 表达 |
| 当前名次 / 终局阶段 | 可选初始化 token；只在历史压缩 checkpoint 中再次出现 |
| 剩余牌数 | 初始 `STATE_LEFT_TILES`；后续不由状态机维护 |
| 宝牌指示牌 | 第一张为初始化 `STATE_DORA`；后续指示牌一律追加 `EVENT_DORA`；完整当前列表由 snapshot 表达 |
| 起手牌 | `start_kyoku` 后，每种数量大于 0 的牌追加一个 `STATE_HAND`。特权训练模式对四家都如此；真实玩家模式仅保留自己真实手牌，对手各用一个 `STATE_HAND(UNKNOWN, 13)`。当前手牌由决策 snapshot 表达 |
| 自己当前摸牌 | 由最近尚未被消费的 `EVENT_DRAW(SELF, ...)` 表达，不重复生成 `STATE_DRAW` |
| 立直宣言 / 立直成立 | 后续由 `EVENT_REACH`、`EVENT_REACH_ACCEPTED` 表达；状态机不推断双立直 |
| 当前振听 | 若状态变化不能由公开事件唯一恢复，在导致变化的 event chunk 末尾追加 `STATE_FURITEN` 增量 |
| 副露和暗杠 | 普通流程由吃、碰、杠确认事件重放；状态机不维护内部副露表 |
| 已见牌 | 普通流程由完整历史重放；checkpoint 中追加 `STATE_SEEN_TILE` |
| 一发 / 岭上状态 | 普通流程由事件重放；checkpoint 中追加 `STATE_IPPATSU` / `STATE_RINSHAN` |
| 牌河 / 摸切手切 / 立直宣言牌 / 最近弃牌 | 普通流程由事件重放；checkpoint 中追加 `STATE_RIVER_TILE` |

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

## 10. MJAI 确认事件到 V3 的完整映射

下表以当前 Rust 状态机支持的 MJAI 确认事件为准。一个事件后若有 continuation，
continuation 必须紧邻主 token。`Observation.legal_actions()` 不属于本表；它只影响
action mask 和动作转换，不写入八维事件序列。

| MJAI Event | V3 token 映射 |
|---|---|
| `none` / 空事件 | `(EVENT_NONE, NONE, NONE, NONE, NONE, NONE, VALUE_NONE, FLAG_NONE, step)`；通常不写入确认历史 |
| `StartGame { id, names, seed }` | 管理器记录 `id` 作为真实 bot 的固定绝对座位；`names`、`seed` 是元数据，不作为模型特征 |
| `StartKyoku { bakaze, dora_marker, kyoku, honba, kyotaku, oya, scores, tehais }` | 一个 `EVENT_START_KYOKU` 头 token，随后由第 9 节状态快照表达初始字段；`tehais` 按第 2 节的可见性模式编码为四家真实起手牌，或“自己真实、对手 UNKNOWN” |
| `Tsumo { actor, pai }` | `(EVENT_DRAW, actor, NONE, pai_if_SELF_else_UNKNOWN, NONE, NONE, VALUE_NONE, FLAG_NONE, step)` |
| `Dahai { actor, pai, tsumogiri }` | `(EVENT_DISCARD, actor, NONE, pai, NONE, NONE, VALUE_NONE, TSUMOGIRI_or_TEDASHI, step)` |
| `Chi { actor, target, pai, consumed }` | `(EVENT_CHI, actor, target, pai, consumed[0], consumed[1], VALUE_NONE, CHI_*, step)` |
| `Pon { actor, target, pai, consumed }` | `(EVENT_PON, actor, target, pai, consumed[0], consumed[1], VALUE_NONE, FLAG_NONE, step)` |
| `Daiminkan { actor, target, pai, consumed[3] }` | 主 token 存 `pai, consumed[0], consumed[1]`；紧跟 `EVENT_MELD_CONT(actor, target, consumed[2], ..., VALUE_0, MELD_DAIMINKAN, step)` |
| `Kakan { actor, pai, consumed[3] }` | 主 token 存 `pai, consumed[0], consumed[1]`；紧跟 `EVENT_MELD_CONT(actor, NONE, consumed[2], ..., VALUE_0, MELD_KAKAN, step)` |
| `Ankan { actor, consumed[4] }` | 主 token 存 `consumed[0], consumed[1], consumed[2]`；紧跟 `EVENT_MELD_CONT(actor, NONE, consumed[3], ..., VALUE_0, MELD_ANKAN, step)` |
| `Dora { dora_marker }` | `(EVENT_DORA, NONE, NONE, dora_marker, NONE, NONE, VALUE_NONE, FLAG_NONE, step)` |
| `Reach { actor }` | `(EVENT_REACH, actor, NONE, NONE, NONE, NONE, VALUE_NONE, REACH_DECLARE, step)` |
| `ReachAccepted { actor }` | `(EVENT_REACH_ACCEPTED, actor, NONE, NONE, NONE, NONE, VALUE_NONE, FLAG_NONE, step)` |
| `Hora { actor, target, deltas, ura_markers }` | `EVENT_HORA(actor, target_or_NONE_for_tsumo, NONE, ..., RON_or_TSUMO)`；每个 delta 紧跟一个 `EVENT_SCORE_DELTA`；赢家可见的每张里宝牌紧跟一个 `EVENT_URA_DORA` |
| `Ryukyoku { deltas }` | `EVENT_RYUKYOKU`，随后每个 delta 一个 `EVENT_SCORE_DELTA`；当前 Rust 事件没有流局原因字段 |
| `EndKyoku` | PPO episode 边界，不进入当前小局模型输入 |
| `EndGame` | rollout 可结束边界，不进入当前小局模型输入 |

主事件 token 的固定格式如下；表中省略的字段均为 `NONE`：

```text
EVENT_START_GAME  = (EVENT_START_GAME, NONE, NONE, NONE, NONE, NONE, VALUE_NONE, FLAG_NONE)
EVENT_START_KYOKU = (EVENT_START_KYOKU, NONE, oya, bakaze, dora_marker, NONE, kyoku_index, FLAG_NONE)
EVENT_DAIMINKAN   = (EVENT_DAIMINKAN, actor, target, pai, consumed[0], consumed[1], VALUE_NONE, MELD_DAIMINKAN, step)
EVENT_KAKAN       = (EVENT_KAKAN, actor, NONE, pai, consumed[0], consumed[1], VALUE_NONE, MELD_KAKAN, step)
EVENT_ANKAN       = (EVENT_ANKAN, actor, NONE, consumed[0], consumed[1], consumed[2], VALUE_NONE, MELD_ANKAN, step)
EVENT_MELD_CONT   = (EVENT_MELD_CONT, actor, target_or_NONE, remaining_tile, NONE, NONE, VALUE_0, MELD_*, step)
EVENT_HORA        = (EVENT_HORA, winner, target_or_NONE, NONE, NONE, NONE, VALUE_NONE, RON_or_TSUMO, step)
EVENT_RYUKYOKU    = (EVENT_RYUKYOKU, NONE, NONE, NONE, NONE, NONE, VALUE_NONE, FLAG_NONE, step)
```

`Hora` 在当前事件结构中没有和了牌字段。因此 `EVENT_HORA.TILE` 必须为
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
[STATE_SNAPSHOT_BEGIN]
[当前 Observation 快照 token]
[STATE_SNAPSHOT_END]
```

`EVENT_HORA`、`EVENT_RYUKYOKU` 可以作为当前小局的终止事件追加到离线完整序列；终止后
没有同一小局的下一次 action sample。`start_game`、`end_kyoku`、`end_game`
是牌桌生命周期事件，不由模型序列消费。

合法动作不写入八维序列。RiichiEnv 风格下，训练代码从 active `Observation` 读取
`legal_actions()`；状态机只做格式转换，不自己计算麻将规则合法性：

```text
环境 MJAI event -> 状态机追加八维序列
Observation.legal_actions() -> 状态机生成 241 维 action mask
模型输入序列 + action mask -> logits 中采样或贪婪选择 action_id
状态机 action_id -> MJAI action JSON
Observation.select_action_from_mjai(...) -> RiichiEnv Action
env.step(actions) -> 环境广播确认 Event -> 写回事件历史和状态机
```

RiichiEnv 没有 `action_ack`；动作是否生效由后续 `new_events()` 中的确认事件表达。

### 决策快照 snapshot

每次模型需要行动时，PPO 从当前 active `Observation` 构造 snapshot JSON，状态机将其临时
编码到历史序列末尾。固定顺序为：

```text
STATE_SNAPSHOT_BEGIN
STATE_SELF_ID
STATE_OYA
STATE_JIKAZE
STATE_BAKAZE
STATE_KYOKU_INDEX
STATE_HONBA
STATE_RIICHI_STICKS
STATE_SCORE x 4
STATE_DORA x N
STATE_HAND x 当前手牌中出现的牌种
STATE_DRAWN_TILE
STATE_RIICHI_DECLARED x 4
STATE_LAST_DISCARD
STATE_LAST_TEDASHI x 4
STATE_SNAPSHOT_END
```

snapshot 字段只来自 RiichiEnv `Observation`：

```text
player_id, oya, round_wind, kyoku_index, honba, riichi_sticks,
scores[4], dora_indicators, hands[player_id], drawn_tile,
riichi_declared[4], last_discard, last_tedashis[4]
```

不写入 snapshot 的字段包括四家副露、四家牌河、`tsumogiri_flags`、`riichi_sutehais`、
`waits`、`is_tenpai`、`legal_actions/action_mask`。

snapshot 是当前决策上下文，不是 MJAI 确认事件。它不会写回玩家状态机历史，因此后续决策
只会看到真实事件历史和自己的当前 snapshot。

## 12. 实现和校验要求

1. Rust 状态机按绝对座位维护事件，序列生成器在输出 token 的最后一步转换成相对座位。
2. 每个 `StartKyoku` 重置该玩家的本局状态和 token 向量，并重新追加初始化前缀；PPO
   rollout 边界、环境 batch 边界都不能触发这一重置。
3. 连续 token 校验：`EVENT_MELD_CONT` 必须紧随对应杠牌事件，`STATE_MELD_CONT` 必须
   紧随对应四张面子状态，二者的 `ACTOR/TARGET/FLAG` 必须匹配。
4. 隐私校验：敌方摸牌必须为 `UNKNOWN`，不应公开的里宝牌不得进入决策前输入。若
   `reveal_opponent_initial_hands=False`，敌方初始手牌和敌方闭合手牌也必须保持未知；若为
   `True`，仅允许在 `start_kyoku` 初始化前缀中写入敌方真实起手牌，并必须将模型标记为
   特权训练模型，禁止作为真实玩家视角直接部署。
5. 一致性校验：从事件历史重放后，应与状态区的手牌、立直、分数、供托、宝牌、
   剩余牌数和可见牌计数一致。
6. 决策快照校验：snapshot 只包含本协议列出的 Observation 字段，不包含四家副露和四家牌河。
7. 状态机不维护副露、巡目、宝牌计数或双立直推断；这些当前局面信息如需给模型使用，
   必须来自 RiichiEnv `Observation` 生成的决策快照。
8. 当前版本仅承诺覆盖本仓库 Rust 状态机支持的四麻 MJAI 事件。三麻、拔北、带流局
   原因的 MJAI 扩展，必须先扩展 Rust Event 和状态机，再占用预留 TYPE。

## 13. RiichiEnv 覆盖校验

当前实现按 RiichiEnv 4P 真实交互接口校验：

| RiichiEnv 输入 | V3 处理 |
|---|---|
| `start_game` | 只设置 `absolute_seat` |
| `start_kyoku` | 清空本小局并生成初始化前缀 |
| `tsumo/dahai/chi/pon/daiminkan/ankan/kakan/dora/reach/reach_accepted/hora/ryukyoku` | 追加对应事件 token；杠牌和结算会追加 continuation / score / ura token |
| `end_kyoku` | 标记 episode 边界，不追加 token |
| `end_game` | 标记 rollout 可结束边界，不追加 token |
| `Observation.legal_actions()` | 不进入八维序列，只转换为 241 维 action mask |

对应自动测试覆盖：

- Rust：每个四麻 RiichiEnv event 至少一个 JSON 样例，检查是否产生预期 token 或边界行为。
- Rust：每类 RiichiEnv legal action 检查对应 241 维 mask id。
- Python：真实 `RiichiEnv(game_mode="4p-red-half")` 跑完整 game，逐个 active
  `Observation` 校验 `legal_actions -> mask -> model_action_to_mjai -> select_action_from_mjai`。
