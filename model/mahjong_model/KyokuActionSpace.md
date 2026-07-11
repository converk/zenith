# KyokuActionSpace V2

## 1. 文档作用与范围

本文档定义模型在**标准四人立直麻将**一小局中可输出的离散动作空间，以及每个动作到 MJAI 玩家事件的映射。

只枚举由玩家主动选择并向 MJAI 服务端发送的事件：

```text
none / dahai / reach / chi / pon / daiminkan / ankan / kakan / hora / ryukyoku
```

以下是牌局或服务端产生的事件，不是模型动作，因此不占输出维度：

```text
start_game / start_kyoku / tsumo / dora / reach_accepted / end_kyoku / end_game
hora 的得点、ura_dora、score delta 等结算信息
```

本版本固定为四人麻将；不包含三麻的 `nukidora`。若将来支持三麻，应新建动作空间版本并重新训练，不在本版本预留无效输出槽位。

## 2. 设计原则

```text
action_id
-> ActionTemplate
-> 当前合法窗口补全为 ActionInstance
-> 一条 MJAI 玩家事件
```

| 原则 | 说明 |
|---|---|
| 原子映射 | 一个模型动作对应一条 MJAI 玩家事件；不使用 `REACH_DISCARD` 之类的复合宏动作。 |
| 固定语义 | 同一 `action_id` 永远表示同一种模板动作，不依赖候选动作的排列顺序。 |
| 只编码真实选择 | 被当前弃牌、当前手牌或副露唯一决定的字段由解码器补全，不重复占用输出维度。 |
| 赤五可区分 | 当保留红五或普通五确实会导致不同后续手牌时，动作空间显式区分。 |
| 合法掩码约束 | 每个决策点仅允许当前规则、手牌和 MJAI 响应窗口下合法的动作参与采样和 PPO 梯度。 |

`ActionInstance` 中 `actor` 固定为自己；`target`、被鸣牌和和牌来源由当前响应窗口补全。动作成功后，服务端回传的真实 MJAI 增量事件才会追加到模型输入序列。

## 3. 牌表示与固定顺序

### 3.1 `ACTION_TILE37`

弃牌动作需区分普通五与红五，使用 37 个可见牌：

```text
1m 2m 3m 4m 5m 6m 7m 8m 9m
1p 2p 3p 4p 5p 6p 7p 8p 9p
1s 2s 3s 4s 5s 6s 7s 8s 9s
E S W N P F C
5mr 5pr 5sr
```

其中前 34 项是去赤牌种，最后三项为红五。令其零基下标为 `tile37_index`。

### 3.2 `TILE34`

暗杠、加杠按去赤牌种选择：

```text
1m..9m, 1p..9p, 1s..9s, E S W N P F C
```

令其零基下标为 `tile34_index`。红五是否参与该副露由当前手牌和已有副露唯一决定，故不再额外展开动作。

## 4. 动作总数

本版本没有预留维度，模型策略头的输出维度严格为：

```text
NUM_ACTIONS = 241
```

| 类别 | 数量 | 需要区分的选择 |
|---|---:|---|
| `PASS` | 1 | 放弃当前响应机会 |
| `DAHAI` | 74 | 37 张可见牌 x 手切/摸切 |
| `REACH` | 1 | 宣告立直 |
| `CHI` | 57 | 三色中 57 个不重复、含赤五区分的 consumed 对子 |
| `PON` | 37 | 31 个非五牌对子，加上每种五牌的两种 consumed 组合 |
| `DAIMINKAN` | 1 | 当前被鸣牌和所需三张手牌唯一确定 |
| `ANKAN` | 34 | 选择要暗杠的去赤牌种 |
| `KAKAN` | 34 | 选择要加杠的去赤牌种/已有碰 |
| `HORA` | 1 | 自摸、荣和、抢杠由当前窗口补全 |
| `RYUKYOKU` | 1 | 当前可宣告的中止流局由窗口补全 |
| 合计 | **241** | |

## 5. 固定 `action_id` 分段

| ID 范围 | 动作 | 数量 |
|---:|---|---:|
| `0` | `PASS` | 1 |
| `1..74` | `DAHAI(tile37, discard_mode)` | 74 |
| `75` | `REACH` | 1 |
| `76..132` | `CHI(consumed_pair)` | 57 |
| `133..169` | `PON(consumed_pair)` | 37 |
| `170` | `DAIMINKAN` | 1 |
| `171..204` | `ANKAN(tile34)` | 34 |
| `205..238` | `KAKAN(tile34)` | 34 |
| `239` | `HORA` | 1 |
| `240` | `RYUKYOKU` | 1 |

```text
1 + 74 + 1 + 57 + 37 + 1 + 34 + 34 + 1 + 1 = 241
```

没有任何预留槽位；合法掩码形状为 `bool[B, 241]`。

## 6. 各类动作与 MJAI 映射

### 6.1 `PASS`

```text
action_id = 0
ActionTemplate = PASS
MJAI = {"type": "none"}
```

只在当前有吃、碰、杠、荣和或中止流局等响应窗口但选择放弃时合法。通常打牌窗口不应把 `PASS` 打开。

### 6.2 `DAHAI`

`DAHAI` 的 74 个动作按以下方式编码：

```text
action_id = 1 + 2 * tile37_index + mode
mode = 0: TEDASHI（手切）
mode = 1: TSUMOGIRI（摸切）
```

解码结果：

```json
{"type":"dahai", "actor":SELF, "pai":TILE37[tile37_index], "tsumogiri":mode == 1}
```

即使牌面相同，摸切和手切仍是不同 MJAI 事件；它们会影响公开牌谱和其他玩家可观察的信息，因此必须是不同策略动作。若该牌不是刚摸到的牌，则对应的摸切动作由 mask 屏蔽。

### 6.3 `REACH`

```text
action_id = 75
MJAI = {"type":"reach", "actor":SELF}
```

`reach` 本身不携带弃牌牌张。它是 MJAI 的原子玩家事件，不能在动作空间中与 `dahai` 合并。协议执行器在服务端要求下一条弃牌时再请求模型输出一个合法 `DAHAI`；服务端回传的 `reach` / `reach_accepted` / `dahai` 增量事件照常追加到输入序列。

### 6.4 `CHI`

吃牌只由上家弃牌触发。模型动作编码自己要从手牌消费的两个牌；被吃的弃牌由当前窗口提供。

每一门共有 15 个去赤不重复 consumed 对子：

```text
相邻对： [1,2] [2,3] [3,4] [4,5] [5,6] [6,7] [7,8] [8,9]
间隔对： [1,3] [2,4] [3,5] [4,6] [5,7] [6,8] [7,9]
```

其中 `[4,5] [5,6] [3,5] [5,7]` 的五牌可分别为普通五或红五，故每门为 `15 + 4 = 19`，三门合计 `57`。

每门的固定局部编号如下；`5` 表示普通五，`5r` 表示红五：

| `local_chi_index` | `consumed_pair` |
|---:|---|
| `0` | `[1, 2]` |
| `1` | `[2, 3]` |
| `2` | `[3, 4]` |
| `3` | `[4, 5]` |
| `4` | `[4, 5r]` |
| `5` | `[5, 6]` |
| `6` | `[5r, 6]` |
| `7` | `[6, 7]` |
| `8` | `[7, 8]` |
| `9` | `[8, 9]` |
| `10` | `[1, 3]` |
| `11` | `[2, 4]` |
| `12` | `[3, 5]` |
| `13` | `[3, 5r]` |
| `14` | `[4, 6]` |
| `15` | `[5, 7]` |
| `16` | `[5r, 7]` |
| `17` | `[6, 8]` |
| `18` | `[7, 9]` |

令 `suit_index(m)=0`、`suit_index(p)=1`、`suit_index(s)=2`，则：

```text
chi_index = 19 * suit_index + local_chi_index
action_id = 76 + chi_index
```

实现必须使用这个固定表，不允许按当前候选列表动态编号。

```json
{"type":"chi", "actor":SELF, "target":CURRENT_TARGET,
 "pai":CURRENT_DISCARD, "consumed":[TILE_A, TILE_B]}
```

### 6.5 `PON`

对 31 个非五牌，每牌只有一个 `PON(tile,tile)` 模板。每种五牌有两个 consumed 模板：

```text
PON(5x, 5x)       # 消耗两张普通五
PON(5x, 5xr)      # 消耗普通五和红五
```

这是实际选择：例如上家打出普通 `5m`、自己持有两张普通 `5m` 和一张红 `5mr` 时，保留红五或保留普通五会形成不同后续手牌，不能合并。

`pon_index` 使用以下固定顺序：

```text
0..30: TILE34 删除 5m、5p、5s 后的顺序，每项对应 PON(tile, tile)
31:    PON(5m, 5m)
32:    PON(5m, 5mr)
33:    PON(5p, 5p)
34:    PON(5p, 5pr)
35:    PON(5s, 5s)
36:    PON(5s, 5sr)
```

```text
action_id = 133 + pon_index
```

```json
{"type":"pon", "actor":SELF, "target":CURRENT_TARGET,
 "pai":CURRENT_DISCARD, "consumed":[TILE_A, TILE_B]}
```

### 6.6 `DAIMINKAN`

```text
action_id = 170
ActionTemplate = DAIMINKAN
```

当前被鸣牌的物理类型和手中其余三张牌会唯一确定 `pai` 与 `consumed`，不存在可由策略选择的赤五保留分支。因此只需一个模板：

```json
{"type":"daiminkan", "actor":SELF, "target":CURRENT_TARGET,
 "pai":CURRENT_DISCARD, "consumed":CURRENT_REQUIRED_THREE_TILES}
```

### 6.7 `ANKAN`

```text
action_id = 171 + tile34_index
```

```json
{"type":"ankan", "actor":SELF, "consumed":CURRENT_FOUR_TILES_OF_TILE34}
```

若选择 `5m/5p/5s`，当前手牌中的红五组成是唯一的；模型只需选择哪一种去赤牌种暗杠。

### 6.8 `KAKAN`

```text
action_id = 205 + tile34_index
```

```json
{"type":"kakan", "actor":SELF, "pai":ADDED_TILE,
 "consumed":EXISTING_PON_THREE_TILES}
```

`pai` 是新加进副露的那张牌；`consumed` 是原有碰的三张牌，不包含 `pai`。已有碰的
三张牌与剩余可加杠牌由当前状态唯一确定；模型只选择要加杠的牌种。

### 6.9 `HORA`

```text
action_id = 239
```

```json
{"type":"hora", "actor":SELF, "target":CURRENT_HORA_TARGET}
```

`libriichi::mjai::Event::Hora` 不携带 `pai`；和牌牌张由当前服务端窗口/牌局状态确定。自摸、荣和、抢杠的差别同样由当前窗口中的 `target` 唯一确定，不需要 `WIN_RON` 和 `WIN_TSUMO` 两个输出槽位。

### 6.10 `RYUKYOKU`

```text
action_id = 240
```

```json
{"type":"ryukyoku"}
```

当前标准规则下主要用于九种九牌。`libriichi::mjai::Event::Ryukyoku` 不携带玩家、原因等字段；没有中止流局机会时必须屏蔽该动作。

## 7. 合法动作掩码与策略分布

状态机/规则模块根据当前玩家私有手牌、公开副露、立直状态、牌河、剩余牌、当前 MJAI 请求窗口和规则构造：

```text
action_mask: bool[B, 241]
```

策略分布只在合法动作上定义：

```text
masked_logits = logits.masked_fill(~action_mask, -inf)
distribution = Categorical(logits=masked_logits)
```

因此被 mask 的动作概率严格为零，不会被采样，也不会作为所选动作参与 PPO 的 `log_prob`、ratio 或策略梯度计算。mask 至少应保证每一行存在一个合法动作；若协议窗口无动作可发，应由环境侧决定是否根本不请求模型，而不是产生全 false mask。

## 8. 解码职责边界

动作解码器必须做以下工作：

1. 使用固定表将 `action_id` 解为 `ActionTemplate`。
2. 使用当前 MJAI 决策窗口补全 `target`、被鸣牌与 `consumed`，并校验和牌或中止流局机会。
3. 校验/构造 `consumed`，特别是红五的物理牌组成。
4. 将 `ActionInstance` 序列化为一条 MJAI 玩家事件。
5. 等待环境/服务端的真实增量事件，再交给状态机追加模型输入。

模型、动作空间和输入事件序列各司其职：模型输出玩家意图；动作解码器生成协议消息；状态机只根据确认后的 MJAI 增量事件维护本局 append-only 输入。

## 9. 版本兼容性

`KyokuActionSpace V2` 的输出维度和 id 语义是模型 checkpoint 的一部分。任何新增规则动作、三麻动作或编号改变都必须：

1. 升级动作空间版本；
2. 更新模型输出头和合法 mask；
3. 在 checkpoint 元数据中记录 `action_space_version=V2` 或新版本；
4. 不将旧 checkpoint 与新动作空间混用。
