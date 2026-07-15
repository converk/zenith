# MJAI 小局状态机

本文件说明编译后的 `riichi` Python 扩展中
[`MjaiKyokuStateMachineManager`](mod.rs) 的状态机边界。
模型动作编号见
[KyokuActionSpace V2](../../../model/mahjong_model/KyokuActionSpace.md)。

## 对象层次

```text
MjaiKyokuStateMachineManager x num_envs
└── PlayerKyokuStateMachine x 4
    ├── absolute_seat
    └── 从 start_kyoku 开始的 append-only Vec<Token>
```

`start_game.id` 只设置单个玩家状态机的绝对座位。`start_kyoku` 清空该玩家的
token 向量，并追加 `EVENT_START_KYOKU + INIT + SEP` 前缀。PPO rollout 的 128 步边界不参与
状态机重置。每一小局的庄家由 `start_kyoku.oya` 给出，状态机用 `absolute_seat + oya`
计算该玩家本局自风。

状态机不维护副露、手牌变化、宝牌计数、巡目计数或双立直推断。当前局面事实由
RiichiEnv `Observation` 在决策时生成 snapshot；历史序列只记录已经确认的 MJAI 事件。

## Rust 接口

`riichi.MjaiKyokuStateMachineManager` 提供新 PPO 专用批处理接口：

```text
apply_events_batch(env_indices, events_by_env_player) -> end_kyoku, end_game
prepare_decisions(batch_indices, legal_action_jsons, snapshot_jsons)
  -> input_ids, attention_mask, sequence_lengths, legal_mask,
     history_lengths, history_generations
decode_actions(batch_indices, action_ids) -> [event_json]
```

新 PPO 每个 tick 接收所有环境、所有四家 observation；每条
`Observation.new_events()` 只喂给对应玩家状态机，绝不广播到其他玩家。

当前并行化覆盖：

- `apply_events_batch()`：按环境分块并行应用四家增量；
- `prepare_decisions()`：只导出 active 决策行的 mask 和带 snapshot 的输入。

状态机不解析 `observation` 的 base64 内容。模型输入只来自状态机自己维护的 append-only 八维
事件序列；合法动作来自 `Observation.legal_actions()` 转成的 MJAI action JSON。

PPO 决策路径使用 `prepare_decisions(...)`。它不会修改玩家历史 token，而是把当前
RiichiEnv `Observation` 转成的 snapshot 临时拼在对应玩家历史序列后面，只返回 active
decision rows。

`start_kyoku` 会重置玩家历史并递增对应玩家的 `history_generation`。rollout GPU inference
使用该 generation 与 `history_length` 校验 KV cache；snapshot 永远不写入状态机历史或 cache，
只作为本次决策 forward 的临时后缀。

## RiichiEnv 事件字段

下面是按 RiichiEnv 源码生成路径核对过的 MJAI 事件字段。这里说的是训练/交互时状态机
应该接收的事件，而不是模型返回的动作。RiichiEnv 当前 PPO 路径没有 `request_action` /
`action_ack` 事件；请求行动由 `env.step(actions)` 返回的 active `Observation` 表达。

`start_game` 有两个常见来源：

- 训练 `Observation.new_events()` / bot 视角：会带 `id`，表示当前玩家自己的绝对座位；
- 全局 `mjai_log` / replay 牌谱：可能只有 `type`，测试牌谱里也可能带 `names`、`seed` 或牌谱
  标识类 `id`。

当前状态机把数值型 `start_game.id` 保存为当前 bot 的绝对座位；若没有 `id`，训练管理器仍按
`player_index=0..3` 的固定玩家视角工作。

| 事件 `type` | RiichiEnv 字段 | 示例 |
|---|---|---|
| `start_game` | `id` 可选；`names`、`seed` 在 replay/测试中可能出现 | `{"type":"start_game","id":0}` |
| `start_kyoku` | `bakaze`, `kyoku`, `honba`, `kyotaku`, `oya`, `scores`, `dora_marker`, `tehais` | `{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"2p","tehais":[["1m","2m", "..."],["?","?", "..."],["?","?", "..."],["?","?", "..."]]}` |
| `tsumo` | `actor`, `pai`；非本人视角的摸牌可被 mask 为 `"?"` | `{"type":"tsumo","actor":0,"pai":"3m"}` |
| `dahai` | `actor`, `pai`, `tsumogiri` | `{"type":"dahai","actor":0,"pai":"3m","tsumogiri":true}` |
| `chi` | `actor`, `target`, `pai`, `consumed` | `{"type":"chi","actor":1,"target":0,"pai":"5m","consumed":["4m","6m"]}` |
| `pon` | `actor`, `target`, `pai`, `consumed` | `{"type":"pon","actor":2,"target":0,"pai":"E","consumed":["E","E"]}` |
| `daiminkan` | `actor`, `target`, `pai`, `consumed` | `{"type":"daiminkan","actor":2,"target":0,"pai":"5p","consumed":["5p","5p","5p"]}` |
| `ankan` | `actor`, `pai`, `consumed` | `{"type":"ankan","actor":0,"pai":"1s","consumed":["1s","1s","1s","1s"]}` |
| `kakan` | `actor`, `pai`, `consumed` | `{"type":"kakan","actor":0,"pai":"5m","consumed":["5m","5m","5m"]}` |
| `dora` | `dora_marker` | `{"type":"dora","dora_marker":"7p"}` |
| `reach` | `actor` | `{"type":"reach","actor":1}` |
| `reach_accepted` | `actor` | `{"type":"reach_accepted","actor":1}` |
| `hora` | `actor`, `target`, `deltas`, `ura_markers`；自摸时额外带 `tsumo:true` | `{"type":"hora","actor":0,"target":0,"deltas":[12000,-4000,-4000,-4000],"tsumo":true,"ura_markers":["3m"]}` |
| `ryukyoku` | `reason`, `deltas` | `{"type":"ryukyoku","reason":"exhaustive_draw","deltas":[1500,1500,-1500,-1500]}` |
| `end_kyoku` | 只有 `type` | `{"type":"end_kyoku"}` |
| `end_game` | 只有 `type` | `{"type":"end_game"}` |

三麻 RiichiEnv 还会出现 `kita`：

| 事件 `type` | RiichiEnv 字段 | 示例 |
|---|---|---|
| `kita` | `actor`, `pai` | `{"type":"kita","actor":0,"pai":"N"}` |

当前状态机目标是四麻小局输入，因此 `kita` 只作为未来三麻扩展记录在文档中。四麻实现中，
`none/pass` 不会作为环境确认事件广播；它只是 bot 在响应窗口中选择放弃的动作。

## 环境事件到八维输入的当前边界

当前状态机已经能把 RiichiEnv 四麻确认事件转换为八维 token 序列。事件覆盖范围包括：
`start_kyoku`、`tsumo`、`dahai`、`chi`、`pon`、`daiminkan`、`ankan`、`kakan`、
`dora`、`reach`、`reach_accepted`、`hora`、`ryukyoku`。`start_game`、`end_kyoku`
和 `end_game` 由牌桌管理器消费，不进入模型的小局输入序列。

几个固定约定如下：

1. **玩家 id / seat**

   每个 `PlayerKyokuStateMachine` 自身保存一个 `absolute_seat`，取值为 `0..3`。
   这个值就是 RiichiEnv 事件中 `actor` 和 `target` 使用的绝对座位编号。
   例如 `absolute_seat=2` 的玩家看到 `{"actor":2}` 时，这是自己；看到
   `{"actor":0}` 时，这是对家。

2. **相对座位**

   状态机输出模型 token 时不会直接使用绝对座位，而是先按玩家自己的
   `absolute_seat` 转换为相对座位：

   ```text
   (actor - self_seat) mod 4 == 0 -> SELF
   (actor - self_seat) mod 4 == 1 -> SHIMOCHA / 下家
   (actor - self_seat) mod 4 == 2 -> TOIMEN / 对家
   (actor - self_seat) mod 4 == 3 -> KAMICHA / 上家
   ```

3. **自风**

   `start_kyoku.oya` 是当前庄家的绝对座位。状态机用
   `self_seat` 和 `oya` 计算自己的自风，并写入 `STATE_JIKAZE`：

   ```text
   self_seat == oya             -> 东
   (self_seat - oya) mod 4 == 1 -> 南
   (self_seat - oya) mod 4 == 2 -> 西
   (self_seat - oya) mod 4 == 3 -> 北
   ```

   `STATE_OYA` 中保存的庄家则会先转换成“相对自己”的座位。

4. **分数**

   状态机对象本身不维护一份独立分数表。每小局开始时，它只读取
   `start_kyoku.scores` 并写入初始化 token。`hora.deltas` 和 `ryukyoku.deltas`
   会被追加为 `EVENT_SCORE_DELTA` token，用于表达结算事件，但不会在状态机内部更新成
   下一局分数；下一局分数仍以新的 `start_kyoku.scores` 为准。

5. **RiichiEnv 事件流形态**

   RiichiEnv 的 `Observation.new_events()` 是“单玩家视角事件流”，里面的 `start_game.id`
   表示这条事件流属于哪个玩家，并且 `start_kyoku.tehais`、非本人 `tsumo.pai` 可能已经被
   mask 成 `"?"`。这种事件流只能喂给对应玩家的状态机，不能再作为桌级事件广播给四家。
   新 PPO 训练使用 `apply_events_batch(env_indices, events_by_env_player)`，其中没有收到
   observation 的玩家事件列表为空。

6. **动作空间**

   本节只描述环境确认事件到模型输入的转换。`Observation.legal_actions()` 到动作 mask、
   以及模型 action id 到 RiichiEnv `Action` 的转换属于动作空间层，见下节。

构造函数的 `reveal_opponent_initial_hands` 控制 `start_kyoku.tehais` 的初始化编码：

- `True`（当前默认）：四家真实起手牌均作为 `STATE_HAND` 写入每位玩家的序列，用于
  特权信息训练；
- `False`：仅写入自己的真实起手牌，三名对手各写入一个 `STATE_HAND(UNKNOWN, 13)`，
  适用于真实玩家可见信息。

即使前者开启，后续敌方 `tsumo` 仍为 `UNKNOWN`。特权模式训练出的模型不能直接用于真实
对局部署，后续应重新以未知模式训练或进行相应蒸馏。

## Python 使用方式

```python
import riichi
import torch

state_machine = riichi.MjaiKyokuStateMachineManager(
    64,
    True,  # 特权训练；真实玩家视角请设为 False
)
state_machine.apply_events_batch([env_index], [[[start_game_json, start_kyoku_json, tsumo_json], [], [], []]])

snapshot_json = {
    "player_id": obs.player_id,
    "oya": obs.oya,
    "round_wind": obs.round_wind,
    "kyoku_index": obs.kyoku_index,
    "honba": obs.honba,
    "riichi_sticks": obs.riichi_sticks,
    "scores": obs.scores,
    "dora_indicators": ["2p"],
    "hand": ["1m", "1m", "5mr"],
    "drawn_tile": "5mr",
    "riichi_declared": obs.riichi_declared,
    "last_discard": None,
    "last_tedashis": [None, None, None, None],
}
input_ids, attention_mask, sequence_lengths, action_mask = state_machine.prepare_decisions(
    [batch_index],
    [[action.to_mjai() for action in obs.legal_actions()]],
    [json.dumps(snapshot_json)],
)
# input_ids: [active_decisions, L, 8]
# attention_mask: [active_decisions, L]
# sequence_lengths: [active_decisions]

input_ids = torch.as_tensor(input_ids, dtype=torch.long, device="cuda")
attention_mask = torch.as_tensor(attention_mask, dtype=torch.bool, device="cuda")

action_ids = policy(input_ids, legal_mask=action_mask).argmax(dim=-1)
mjai_json = state_machine.decode_actions([batch_index], [int(action_ids[0])])[0]
env_action = obs.select_action_from_mjai(mjai_json)
```

之后的新 PPO/RiichiEnv 训练代码应直接调用
`riichi.MjaiKyokuStateMachineManager`，不要再通过 Python 状态机封装层间接加载。

`decode_actions` 不从状态机内部牌局窗口推断动作字段。当前路径先把
RiichiEnv `Observation.legal_actions()` 的 MJAI JSON 存到对应玩家的 241 维槽位中；
模型输出 action_id 后，状态机只取回该槽位上的原始 MJAI JSON。若模型选择的 action_id
不在当前 mask 中，状态机会报错，而不是自行补字段生成一个可能非法的响应。

因此 `reach`、`hora`、`ryukyoku` 的向听、振听、役种、九种九牌等**规则合法性**完全由
RiichiEnv `Observation.legal_actions()` 提供。状态机不在 Python 或 Rust 中重建第二份规则裁判。

## 决策快照

快照字段来自 RiichiEnv 当前 active `Observation`，由 PPO Python 侧把 RiichiEnv 136 牌 id
转换为 MJAI 牌字符串后传给 Rust：

```text
player_id, oya, round_wind, kyoku_index, honba, riichi_sticks,
scores[4], dora_indicators, hand, drawn_tile,
riichi_declared[4], last_discard, last_tedashis[4]
```

状态机编码顺序固定为：

```text
STATE_SNAPSHOT_BEGIN
STATE_SELF_ID
STATE_OYA
STATE_JIKAZE
STATE_BAKAZE
STATE_KYOKU_INDEX
STATE_HONBA
STATE_RIICHI_STICKS
STATE_SCORE x4
STATE_DORA xN
STATE_HAND x当前手牌牌种
STATE_DRAWN_TILE
STATE_RIICHI_DECLARED x4
STATE_LAST_DISCARD
STATE_LAST_TEDASHI x4
STATE_SNAPSHOT_END
```

四家副露、四家牌河、`tsumogiri_flags`、`riichi_sutehais`、`waits`、`is_tenpai` 和
`legal_actions/action_mask` 不进入 snapshot。

PPO update 直接重放 rollout 时保存的单决策输入。状态机仍只负责提供真实事件历史和每次决策的
snapshot token，不负责构造训练 batch。

## RiichiEnv 合法动作边界

状态机现在支持 RiichiEnv `Observation.legal_actions()` 到 241 维模型动作
掩码的转换。完整合法性仍由环境作为唯一牌局裁判决定；状态机不自己计算立直、振听、役种、
和牌、食替等规则合法性。

当前已完成的职责是：

```text
MJAI event -> append-only 八维序列
Observation.legal_actions() -> action_mask: bool[num_envs * 4, 241]
241 维 action_id -> 一条 MJAI action JSON -> Observation.select_action_from_mjai(...)
```

`Observation.legal_actions()` 中的每个 action 会先由 Python 调用 `Action.to_mjai()` 转成
MJAI-format action JSON，再映射到固定动作编号：

```text
none -> 0
dahai -> 1..74
reach -> 75
chi -> 76..132
pon -> 133..169
daiminkan -> 170
ankan -> 171..204
kakan -> 205..238
hora -> 239
ryukyoku -> 240
```

对 `dahai`，手切/摸切是否合法也以 RiichiEnv `legal_actions()` 给出的 MJAI JSON 为准。
状态机只把这些 JSON 映射到 `1..74` 的手切/摸切动作槽位，不额外根据内部手牌过滤。

Python 训练代码随后调用 `Observation.select_action_from_mjai(...)` 得到 RiichiEnv `Action`。

当前 Rust 核心已维护自己可见的手牌计数、事件顺序和 token 序列；它尚未移植
`libriichi::PlayerState` 的完整向听、振听和合法动作计算。`STATE_FURITEN` 等私有状态
增量仍是协议保留能力，应在动作空间解码器与规则计算模块接入后追加。

## 覆盖校验矩阵

本状态机的自动测试按 RiichiEnv 4P 源码中的真实事件和动作类型校验，不把三麻 `kita`
纳入本版本动作空间。

### 事件到八维序列

| RiichiEnv event | 状态机处理 | 八维 token |
|---|---|---|
| `start_game` | 设置单玩家 `absolute_seat` | 不进入模型输入 |
| `start_kyoku` | 重置小局状态并追加初始化前缀 | `EVENT_START_KYOKU` + `STATE_*` + `SEP` |
| `tsumo` | 更新可见手牌/摸牌窗口 | `EVENT_DRAW` |
| `dahai` | 更新手牌、牌河和响应窗口 | `EVENT_DISCARD` |
| `chi` | 追加吃牌确认事件 | `EVENT_CHI` |
| `pon` | 追加碰牌确认事件 | `EVENT_PON` |
| `daiminkan` | 追加大明杠确认事件 | `EVENT_DAIMINKAN` + `EVENT_MELD_CONT` |
| `ankan` | 追加暗杠确认事件 | `EVENT_ANKAN` + `EVENT_MELD_CONT` |
| `kakan` | 追加加杠确认事件 | `EVENT_KAKAN` + `EVENT_MELD_CONT` |
| `dora` | 追加新宝牌指示牌 | `EVENT_DORA` |
| `reach` | 追加立直宣言 | `EVENT_REACH` |
| `reach_accepted` | 追加立直成立 | `EVENT_REACH_ACCEPTED` |
| `hora` | 追加和牌、点差、里宝牌 | `EVENT_HORA` + `EVENT_SCORE_DELTA` + `EVENT_URA_DORA` |
| `ryukyoku` | 追加流局和点差 | `EVENT_RYUKYOKU` + `EVENT_SCORE_DELTA` |
| `end_kyoku` | PPO episode 边界 | 不进入模型输入 |
| `end_game` | rollout 可结束边界 | 不进入模型输入 |

### legal_actions 到 241 维 mask

| RiichiEnv/MJAI action | action id | 校验点 |
|---|---:|---|
| `none` | `0` | 仅响应窗口放弃 |
| `dahai` | `1..74` | 区分 37 张牌与手切/摸切 |
| `reach` | `75` | 立直宣言本身不携带弃牌 |
| `chi` | `76..132` | 按 consumed 对子区分含赤五组合 |
| `pon` | `133..169` | 按 consumed 对子区分含赤五组合 |
| `daiminkan` | `170` | 当前弃牌唯一确定牌种和 consumed |
| `ankan` | `171..204` | 按去赤 34 牌种 |
| `kakan` | `205..238` | 按去赤 34 牌种 |
| `hora` | `239` | 自摸/荣和由当前 legal action 决定 |
| `ryukyoku` | `240` | 对应 RiichiEnv `KyushuKyuhai` |

校验方式：

```text
Observation.legal_actions()
-> Action.to_mjai()
-> prepare_decisions(...)
-> decode_actions(...)
-> Observation.select_action_from_mjai(...)
```

只要 `Observation.select_action_from_mjai(...)` 返回非空 `Action`，说明当前 241 维 mask
和 MJAI 回转对 RiichiEnv 是可接受的。

### 随机真实环境覆盖

仓库提供 `tests/riichienv_state_machine_smoke.py` 作为真实 RiichiEnv 随机覆盖脚本：

```bash
python tests/riichienv_state_machine_smoke.py --games 64 --seed 20260713
```

脚本会随机跑多个 `4p-red-half` 半庄，并在每个 active `Observation` 上执行完整闭环：

```text
new_events -> apply_events_batch -> prepare_decisions
legal_actions -> prepare_decisions -> decode_actions -> Observation.select_action_from_mjai
```

随机覆盖用于发现真实 RiichiEnv `legal_actions()` 与 241 维动作空间之间的不兼容；它不要求
每次自然出现所有稀有事件。`reach`、`reach_accepted` 等低频事件由 Rust 定向单测覆盖。
