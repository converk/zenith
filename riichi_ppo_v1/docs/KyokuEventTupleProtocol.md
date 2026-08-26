# Kyoku 状态协议

本文定义 RiichiEnv、MJAI 状态机、Python bridge 与 V18 模型的状态边界。动作空间和
MJAI 回转见 [KyokuActionSpace.md](KyokuActionSpace.md)，精确张量/schema 见
[v18_input_protocol.md](v18_input_protocol.md)。

## 数据流

```text
Observation.new_events() -> Rust 状态机（仅用于同步/生命周期/动作执行，不作模型输入）
Observation 当前字段 -> Rust/PyO3 当前局面编码器（Shared 快照 + Opponent Analysis）
Observation.legal_actions() -> 每动作 Offense/Defense Query pair + legal mask（按 action_id 升序）
完整 Actor 序列 -> Actor raw logits[action_id]
Shared 公共表示 + 三家闭手 + 未来五张牌 -> Critic value（仅训练）
```

V18 **不再把 MJAI 事件序列编码进模型输入**。事件只用于状态机同步、牌局推进、奖励与
动作执行（`new_events()`/`apply_events_batch`）；模型输入由当前 `Observation` 的快照
字段直接构造。每张桌维护四个观察者状态机。

## 公开事件语义（同步用途）

绝对座次转换为观察者相对座次：自己=0、下家=1、对面=2、上家=3；无 supplier
时为 0=N/A（Query supplier 使用 1..3）。

| MJAI 事件 | 状态变化 |
| --- | --- |
| `start_game` | 设置观察者绝对座位 |
| `start_kyoku` | 初始化局况、分数、手牌与公开宝牌 |
| `tsumo` | 更新当前摸牌；只有自己的摸牌进入 actor 当前事实 |
| `dahai` | 更新牌河与手切/摸切 |
| `chi/pon/daiminkan` | 更新副露与供牌者 |
| `ankan/kakan` | 更新副露（kakan 只表达当前形态） |
| `dora` | 追加公开宝牌指示牌 |
| `reach/reach_accepted` | 更新立直状态/宣言河牌索引 |
| `hora/ryukyoku/end_kyoku/end_game` | 只形成奖励或生命周期边界 |

终局得点、里宝、对手摸牌和未知牌山组成不得进入 Actor 输入。状态机消费环境已经
广播确认的事件；模型选择本身不会提前写入任何历史 token。

## 当前决策输入

Actor 序列为 `[B,T,32]` 的当前局面快照（段/类别/字段 schema 见
`model/encoding_protocol.py` 与 `v18_input_protocol.md`）：桌况、自身手牌与
SELF_STATE_ANALYSIS、四家 PLAYER、三家完整牌河（逐张 + 各两个六张摘要）、当前副露、
34 个 TILE_STATE、三个 OPPONENT_ANALYSIS 与按 action ID 升序的 Offense/Defense Query。
**不包含**：`tiles_left`、独立当前供牌公共字段、54 行 Atomic Snapshot、MJAI 历史事件
token（无历史 generation/cache 标识）。所有类别分隔符与有效 token 都计入长度并应用
RoPE；超过 `context_tokens` 必须拒绝，不能截断。

Query 行宽 15 不变：每动作连续 Offense/Defense 两行；`chi/pon/daiminkan/ron` 的
supplier 必须来自原生最后供牌者；其他动作必须为 N/A。Query action-ID 集合与
`bool[B,241]` legal mask 的打开集合完全相同，且按 action ID 升序规范排序（环境返回
顺序不影响编码结果）。

## 集中式 Critic

Critic 只复用 Shared 公共表示。其 private sequence 必须依次为 `SEP_CRITIC`、相对座次
1、2、3 的真实闭手，然后恰好五张未来牌山，最后是 value query。普通五与赤五须可区分；
任何 private token 都不能改变 Actor raw logits；Critic 不接收 Opponent Analysis 或
Action Query。

## 错误边界

非法 MJAI JSON/牌/座次、错误 shape/dtype、段/分隔符顺序或域错误、空 mask、
重复/缺失 Query、supplier 错误、Critic private 段错序和超长上下文均立即报错。
RiichiEnv 若拒绝由 action ID 回转的动作，也必须报错，不能静默换成 fallback。

V16/V17 状态和输入文档仅作历史归档；活跃代码不提供加载、迁移或协议适配。
PPO/rollout 与 `riichi_lab_bot` 对旧输入契约的引用已盘点为 V18 后续待迁移项。
