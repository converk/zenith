# KyokuActionSpace

## 1. 文档作用与范围

本文档定义模型在**标准四人立直麻将**一小局中可输出的离散动作空间，以及每个动作到
MJAI 玩家动作 JSON 和 RiichiEnv `Action` 的兼容映射。

只枚举玩家在 RiichiEnv `Observation.legal_actions()` 决策窗口中可能选择的动作：

```text
none / dahai / reach / chi / pon / daiminkan / ankan / kakan / hora / ryukyoku
```

以下是环境广播的确认事件，不是模型动作，因此不占输出维度：

```text
start_game / start_kyoku / tsumo / dora / reach_accepted / end_kyoku / end_game
hora 的得点、ura_dora、score delta 等结算信息
```

动作空间固定为四人麻将；不包含三麻的 `nukidora`。若将来支持三麻，必须重新设计动作编号并重新训练，不在当前空间预留无效输出槽位。

## 2. 设计原则

```text
action_id
-> ActionTemplate
-> 当前 Observation.legal_actions() 补全为 ActionInstance
-> 一条 MJAI 玩家动作 JSON
-> RiichiEnv Observation.select_action_from_mjai(...)
-> RiichiEnv Action
```

| 原则 | 说明 |
|---|---|
| 原子映射 | 一个模型动作对应一条 MJAI 玩家动作；不使用 `REACH_DISCARD` 之类的复合宏动作。 |
| 固定语义 | 同一 `action_id` 永远表示同一种模板动作，不依赖候选动作的排列顺序。 |
| 只编码真实选择 | 被当前弃牌、当前手牌或副露唯一决定的字段由解码器补全，不重复占用输出维度。 |
| 赤五可区分 | 当保留红五或普通五确实会导致不同后续手牌时，动作空间显式区分。 |
| 合法掩码约束 | 每个决策点仅允许 RiichiEnv `Observation.legal_actions()` 中可匹配的动作参与采样和 PPO 梯度。 |

`ActionInstance` 中 `actor` 固定为自己；`target`、被鸣牌和和牌来源都来自当前
`Observation.legal_actions()` 中的 MJAI action JSON。状态机不再通过最近事件上下文自行补字段。
动作成功后，环境广播的真实 MJAI 增量事件才会追加到模型输入序列。

RiichiEnv 没有 `request_action` / `action_ack` 事件。请求行动由 `env.step(actions)` 或
`env.reset()` 返回的 `dict[int, Observation]` 表示；这个 dict 中出现的 `player_id`
就是当前需要决策的玩家。动作是否生效由后续 `obs.new_events()` 中的 `dahai`、`chi`、
`pon`、`hora` 等广播事件确认。

## 2.1 与 MJAI action 和 RiichiEnv Action 的兼容关系

本动作空间是模型自己的 241 维稳定编号，不等于 RiichiEnv 内置 `obs.mask()` 的动作编码。
四人 RiichiEnv 的内置动作编码约为 82 维，并会合并一些本动作空间显式区分的选择，例如
赤五吃碰、手切/摸切等。因此：

```text
obs.legal_actions()
-> set_legal_actions_batch(...) + action_mask()
-> bool[241]
-> 模型采样 action_id
-> model_action_to_mjai(action_id)
-> MJAI action JSON
-> obs.select_action_from_mjai(mjai_json)
-> RiichiEnv Action
```

实现时应优先基于 `obs.legal_actions()` 生成 241 维 mask，而不是直接使用 `obs.mask()`。
`obs.mask()` 只能作为 RiichiEnv 原生动作空间的 mask，不可直接喂给 241 维策略头。

当一个 `action_id` 无法在当前 `obs.legal_actions()` 中找到匹配的 RiichiEnv `Action` 时，
说明模型输出与当前合法窗口不一致。bridge 必须立即报错并中止当前 rollout，不能静默替换为
`Pass` 或其他合法动作。

本文后续 JSON 示例用于描述 MJAI action 语义。实际提交给 RiichiEnv 时，建议先生成
MJAI action JSON，再调用 `obs.select_action_from_mjai(...)` 得到 RiichiEnv `Action`，
最终提交 `Action` 给 `env.step(actions)`。

RiichiEnv 4P `ActionType` 到本动作空间的覆盖关系如下：

| RiichiEnv `ActionType` | MJAI action `type` | 本动作空间 |
|---|---|---|
| `Pass` | `none` | `0` |
| `Discard` | `dahai` | `1..74`，按牌和手切/摸切展开 |
| `Riichi` | `reach` | `75` |
| `Chi` | `chi` | `76..132`，按 consumed 对子展开 |
| `Pon` | `pon` | `133..169`，按 consumed 对子展开 |
| `Daiminkan` | `daiminkan` | `170` |
| `Ankan` | `ankan` | `171..204` |
| `Kakan` | `kakan` | `205..238` |
| `Ron` / `Tsumo` | `hora` | `239` |
| `KyushuKyuhai` | `ryukyoku` | `240` |
| `Kita` | `kita` | 不支持；三麻专用，不属于四麻动作空间 |

自动校验要求是：当前 `Observation.legal_actions()` 打开的每个动作，都必须能映射到上表
中的一个 241 维 id；mask 中任意打开的 id，都必须能通过
`Observation.select_action_from_mjai(...)` 回转为 RiichiEnv `Action`。

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

没有预留维度，模型策略头的输出维度严格为：

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

即使牌面相同，摸切和手切仍是不同 MJAI 事件；它们会影响公开牌谱和其他玩家可观察的信息，因此必须是不同策略动作。

`obs.legal_actions()` 是弃牌合法性的来源，但状态机会额外用 observation 中的手牌、
最近摸牌和 RiichiEnv 合法动作做
`DAHAI` 一致性过滤：

- `tsumogiri=true`：只有该牌等于最近摸牌时才打开；
- `tsumogiri=false`：若该牌不是最近摸牌，只要手里有这张牌即可打开；
- `tsumogiri=false` 且该牌等于最近摸牌时，手里必须至少有两张同一物理牌，才表示可以
  手切原本持有的那一张。

例如只摸到唯一一张 `5m` 时，只打开 `5m` 摸切；若原本已有 `1m` 又摸到 `1m`，则
`1m` 手切和 `1m` 摸切都可以同时打开。

### 6.3 `REACH`

```text
action_id = 75
MJAI = {"type":"reach", "actor":SELF}
```

`reach` 本身不携带弃牌牌张。它是 MJAI 的原子玩家动作，不能在动作空间中与 `dahai`
合并。RiichiEnv 中 `Riichi` 合法动作本身不携带弃牌；立直宣言后，环境会在后续
Observation 中给出对应的合法弃牌动作，模型再输出一个合法 `DAHAI`。环境广播的
`reach` / `reach_accepted` / `dahai` 增量事件照常追加到输入序列。

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

RiichiEnv 内部的 `Tsumo` 和 `Ron` 在 MJAI action 中都表示为 `hora`。模型动作
只需要表达“我要和牌”；自摸、荣和、抢杠由当前 Observation 合法动作窗口决定，不需要
`WIN_RON` 和 `WIN_TSUMO` 两个输出槽位。状态机可以保留或补齐 `target/pai` 等字段，但
这些字段不是模型动作空间的一部分。

### 6.10 `RYUKYOKU`

```text
action_id = 240
```

```json
{"type":"ryukyoku"}
```

当前标准规则下主要用于九种九牌等环境允许的中止流局动作。若 RiichiEnv
`obs.legal_actions()` 包含 `KyushuKyuhai`，状态机必须打开第 `240` 维；没有中止流局机会时，
第 `240` 维必须保持屏蔽。

## 7. 合法动作掩码与策略分布

当前实现中，合法动作由 RiichiEnv 作为唯一规则裁判给出。状态机不重新计算立直、振听、
役种、食替等规则合法性，而是把 RiichiEnv 的 `Observation.legal_actions()` 转成：

```text
action_mask: bool[B, 241]
```

状态机只做少量协议一致性过滤和补字段：例如 `DAHAI` 会用 observation 中的手牌、
`drawn_tile` 和 RiichiEnv 合法动作区分手切/摸切是否同时可行；`chi/pon/kan/hora`
等规则合法性仍以 `obs.legal_actions()` 为准。

策略分布只在合法动作上定义：

```text
masked_logits = logits.masked_fill(~action_mask, -inf)
distribution = Categorical(logits=masked_logits)
```

因此被 mask 的动作概率严格为零，不会被采样，也不会作为所选动作参与 PPO 的
`log_prob`、ratio 或策略梯度计算。mask 至少应保证每个出现在 `obs_dict` 中、且
`legal_actions()` 非空的玩家存在一个合法动作；未出现在 `obs_dict` 中的玩家不参与
环境动作提交，训练代码可在批处理张量中用 `PASS` 兜底避免全 false logits，但不能把
这个兜底动作提交给环境。

### 7.1 覆盖测试职责

动作空间有两类测试：

- Rust 定向单测验证完整固定编号表，包括 `CHI` 的 57 个 consumed 模板、`PON` 的
  37 个 consumed 模板，以及每个动作分段的 mask id。
- Python 随机集成测试验证 RiichiEnv 实际返回的 `Observation.legal_actions()` 是否都能
  映射到 241 维 mask，并且每个打开的 action id 都能通过
  `Observation.select_action_from_mjai(...)` 回转成 RiichiEnv `Action`。

默认随机覆盖命令：

```bash
cd riichi_ppo_v1
conda run -n Mahjong-AI python -m unittest discover -s tests/protocol -t . -v
```

随机对局不保证自然出现所有稀有动作；`hora`、`reach`、`ryukyoku` 等仍以定向单测作为
完整覆盖来源。

## 8. 解码职责边界

动作解码器必须做以下工作：

1. 使用固定表将 RiichiEnv `Observation.legal_actions()` 中的 MJAI action JSON 映射到
   241 维 mask。
2. 模型输出 `action_id` 后，取回当前 mask 中对应槽位保存的原始 MJAI action JSON。
3. 使用 `obs.select_action_from_mjai(mjai_json)` 将 MJAI JSON 转成 RiichiEnv 可接受的
   `Action`；若无法匹配，则说明 mask 或解码器有问题，必须报错。
4. 等待环境广播的真实增量事件，再交给状态机追加模型输入。

模型、动作空间和输入事件序列各司其职：模型输出玩家意图；动作解码器只在当前
`legal_actions()` 候选内做 mask/回转；状态机只根据确认后的 MJAI 增量事件维护本局
append-only 输入。

## 9. 变更约束

输出维度和 `action_id` 语义是训练产物的业务接口。任何新增规则动作、三麻动作或编号改变都必须：

1. 同步更新模型输出头、合法 mask、编码器与解码器；
2. 重新执行固定编号表、真实环境回转和 mask 覆盖测试；
3. 使用更新后的动作空间重新训练，不对已有训练产物提供转换。
