# MJAI 小局状态机

本文件说明编译后的 `riichi` Python 扩展中
[`MjaiKyokuStateMachineManager`](mod.rs) 的状态机边界。
模型动作编号见
[KyokuActionSpace V2](../../../model/mahjong_model/KyokuActionSpace.md)。

## 对象层次

```text
MjaiKyokuStateMachineManager x num_envs
├── GameContext
│   ├── start_game 的玩家名称、随机种子和规则模式
│   └── 不进入模型输入
└── PlayerKyokuStateMachine x 4
    ├── 自己可见的手牌计数和局内派生状态
    └── 从 start_kyoku 开始的 append-only Vec<Token>
```

`start_game` 更新 `GameContext`；`start_kyoku` 清空四位玩家的局内状态和 token 向量，
并为每人追加 `EVENT_START_KYOKU + INIT + SEP` 前缀。PPO rollout 的 128 步边界不参与
状态机重置。

## Rust 接口

`riichi.MjaiKyokuStateMachineManager` 提供：

```text
reset()
reset_env(env_index)
apply_event(env_index, event_json)
apply_events([event_json_or_none] * num_envs)
apply_env_message(env_index, player_index, message_json)
apply_request(env_index, player_index, request_action_json)
apply_requests([request_action_json_or_none] * (num_envs * 4))
model_inputs() -> input_ids, attention_mask, sequence_lengths
action_mask() -> bool[num_envs * 4, 241]
player_tokens(env_index, player_index)
action_to_mjai(env_index, player_index, action_id) -> event_json
actions_to_mjai([action_id] * (num_envs * 4)) -> [event_json_or_none]
model_to_env([action_id] * (num_envs * 4)) -> [response_json_or_none]
```

批量接口以每 8 个玩家为一组并行处理；在四人麻将中等价于每个线程处理 2 个环境。
`apply_events` 中每个环境收到一个事件后，会将其按不同视角广播给四位玩家。敌方摸牌
被编码为 `UNKNOWN`，而自己的摸牌保留真实牌值。

当前并行化覆盖：

- `reset()`：按环境分块重置；
- `apply_events()`：按 2 个环境/8 个玩家分块应用事件；
- `apply_requests()`：按 2 个环境/8 个玩家分块记录请求；
- `action_mask()`：按 2 个环境/8 个玩家分块生成合法动作 mask；
- `model_inputs()`：按 2 个环境/8 个玩家分块拷贝 token 序列；
- `actions_to_mjai()` / `model_to_env()`：按 8 个玩家分块解码模型动作。

`apply_event(env_index, ...)`、`apply_request(env_index, player_index, ...)`、
`reset_env(env_index)` 这类单点调试接口仍是单线程。

`apply_env_message` 是 RiichiEnv/RiichiLab 风格的入口：

- `request_action`：不进入模型事件序列；只记录该玩家的 `request_id` 和
  `possible_actions`，用于后续生成 action mask；
- `action_ack`：当前直接忽略；
- 其他 MJAI 消息：按普通事件处理，追加到四位玩家的小局序列。

状态机不解析 `observation` 的 base64 内容。模型输入只来自状态机自己维护的 append-only 九维
事件序列；合法动作来自 `possible_actions` 的格式转换。

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
    1,     # 1=四麻东风，2=四麻半庄
    True,  # 特权训练；真实玩家视角请设为 False
)
state_machine.apply_event(0, start_game_json)
state_machine.apply_event(0, start_kyoku_json)
state_machine.apply_event(0, tsumo_json)

input_ids, attention_mask, sequence_lengths = state_machine.model_inputs()
# input_ids: [64 * 4, L, 9]
# attention_mask: [64 * 4, L]
# sequence_lengths: [64 * 4]

input_ids = torch.as_tensor(input_ids, dtype=torch.long, device="cuda")
attention_mask = torch.as_tensor(attention_mask, dtype=torch.bool, device="cuda")

action_ids = policy(input_ids, attention_mask=attention_mask).argmax(dim=-1)
responses = state_machine.actions_to_mjai(action_ids.cpu().reshape(-1).tolist())
# responses[env_index][player_index]: 一条 MJAI JSON，或该玩家当前无决策时的 None
```

RiichiEnv/RiichiLab 请求式流程：

```python
state_machine.apply_events([event_json_or_none] * 64)
state_machine.apply_requests(requests)  # 展平 [64 * 4]，元素为 request_action JSON 或 None

input_ids, attention_mask, sequence_lengths = state_machine.model_inputs()
action_mask = state_machine.action_mask()
# action_mask: [64 * 4, 241]，由 request_action.possible_actions 转换而来

input_ids = torch.as_tensor(input_ids, dtype=torch.long, device="cuda")
action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device="cuda")
action_ids = policy(input_ids, legal_mask=action_mask).argmax(dim=-1)
responses = state_machine.model_to_env(action_ids.cpu().reshape(-1).tolist())
# responses 里的 JSON 会回显对应 request_id，例如：
# {"type":"dahai","actor":0,"pai":"3m","request_id":42}
```

之后的新 PPO/RiichiEnv 训练代码应直接调用
`riichi.MjaiKyokuStateMachineManager`，不要再通过 Python 状态机封装层间接加载。

`action_to_mjai` 已按 `KyokuActionSpace V2` 的 241 维固定编号输出一条 MJAI 玩家事件。
状态机维护当前决策窗口：自己摸牌后的自回合、弃牌后的响应窗口、以及加杠后的抢杠窗口。
它会校验动作窗口、手中 `consumed` 牌、吃的来源玩家、摸切牌和副露组成；批量接口中没有
决策窗口的玩家返回 `None`。

当前解码器尚未移植完整的立直麻将规则计算，因此 `reach`、`hora`、`ryukyoku` 的向听、
振听、役种、九种九牌等**规则合法性**仍须由后续 Rust 规则模块生成 action mask。解码器不在
Python 重建第二份手牌状态。

## RiichiEnv 合法动作边界

状态机现在支持 RiichiEnv/RiichiLab 的 `request_action.possible_actions` 到 241 维模型动作
掩码的转换。完整合法性仍由环境作为唯一牌局裁判决定；状态机不自己计算立直、振听、役种、
和牌、食替等规则合法性。

当前已完成的职责是：

```text
MJAI event -> append-only 九维序列
request_action.possible_actions -> action_mask: bool[num_envs * 4, 241]
241 维 action_id + request_id -> 一条 RiichiEnv/RiichiLab MJAI response JSON
```

`possible_actions` 中的每个 MJAI-format action 会被映射到固定动作编号：

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

`model_to_env` 优先复用 `possible_actions` 里的原始 action JSON，然后补齐
`request_id` 和缺失的 `actor`。对 `chi/pon/daiminkan/hora`，如果 `possible_actions`
省略了 `target`，状态机会用最近事件维护的反应窗口补齐。若模型选择的 action_id 不在当前
请求的 mask 中，状态机会报错，而不是生成非法响应。

当前 Rust 核心已维护自己可见的手牌计数、事件顺序和 token 序列；它尚未移植
`libriichi::PlayerState` 的完整向听、振听和合法动作计算。`STATE_FURITEN` 等私有状态
增量仍是协议保留能力，应在动作空间解码器与规则计算模块接入后追加。
