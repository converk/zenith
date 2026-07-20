# Kyoku 状态协议

本文定义 `RiichiEnv`、Python bridge、Rust `MjaiKyokuStateMachineManager` 与 PPO 模型之间的状态边界。策略只表达标准四人立直麻将中观察者在决策时可见的信息；集中式 value 在该公共序列之外，可于训练期接收三家对手的闭手牌（以及可选的公开牌河/副露压缩表示）。这些 critic-only 特权信息绝不进入策略分支。

## 1. 数据流与职责

```text
Observation.new_events()
  -> apply_events_batch()
  -> 每个观察者的公开 history

Observation.legal_actions() + Observation 当前字段
  -> legal action MJAI JSON + DecisionSnapshot JSON
  -> prepare_decisions()
  -> token_factors / token_numeric / token_lengths / action_mask / history_generation
  -> 模型追加 learned query，输出 policy logits 与 value

actor 只消费上述 `token_factors/token_numeric` 公共序列，其中自身手牌与摸牌只对该观察者可见。critic 还消费独立的 `critic_factors`：三家对手的闭手牌必定包含，`critic_include_public_state=true` 时还包含三家牌河与副露的压缩 token。critic 的隐藏手牌、对手摸牌、里宝及牌山组成不会回流到 actor。
```

每张桌维护四个按观察者区分的状态机。history 在同一局内只追加公开事件；每次决策根据快照重建当前状态后缀。模型不接收候选动作 token，合法选择由独立的固定 `bool[241]` 掩码表示。动作空间与 MJAI 回转规则见 [KyokuActionSpace.md](KyokuActionSpace.md)。

## 2. 事件 history

状态机接收每位观察者的 MJAI 增量。绝对座位先转换为观察者视角：`SELF=1`、`SHIMOCHA=2`、`TOIMEN=3`、`KAMICHA=4`，`0` 表示没有座位。事件 token 的十个因子均在后文定义。

| MJAI 事件 | history 或状态变化 | 写入的业务字段 |
|---|---|---|
| `start_game` | 仅更新观察者绝对座位 | `id` 必须在 `0..4` |
| `start_kyoku` | 清空旧 history、活牌山置为 70，并追加开局 token | `bakaze` 与 `kyoku` 编为场风/局号组合；初始手牌、分数和宝牌不直接从该事件取值 |
| `tsumo` | 活牌山减一，不追加 token | 自己的摸牌由当前快照的 `drawn_tile` 表达 |
| `dahai` | 追加弃牌 token | 相对 `actor`、`pai`、赤五、`tsumogiri` |
| `chi` | 追加吃 token | 相对 `actor`/`target`、被鸣 `pai`、顺子中被鸣牌位置、是否含赤五 |
| `pon`、`daiminkan` | 各追加一个副露 token | 相对 `actor`/`target`、`pai`、副露是否含赤五 |
| `ankan`、`kakan` | 各追加一个杠 token | 相对 `actor`、牌种、是否含赤五；暗杠来源固定为 `SELF` |
| `dora` | 追加宝牌指示牌 token | `dora_marker` |
| `reach`、`reach_accepted` | 各追加一个立直过程 token | 相对 `actor` |
| `hora`、`ryukyoku`、`end_kyoku`、`end_game`、`none` | 不进入模型 history | 只形成奖励和生命周期边界 |

新 `start_kyoku` 必须丢弃上一局的 history。终局得点、里宝、牌山组成和对手摸牌都不进入 token；这既避免重复，也避免泄露不可见信息。

## 3. 决策快照到状态后缀

Python bridge 为每条决策构造以下 JSON。`player_id` 必须等于该观察者的状态机座位，`oya` 必须是四个绝对座位之一。

| `DecisionSnapshot` 字段 | 来源 | 生成的状态语义 |
|---|---|---|
| `player_id` | `Observation.player_id` | 校验观察者身份，不单独编码 |
| `scores[4]` | `Observation.scores` | 四条分数 token，各自带相对座位 |
| `round_wind`、`kyoku_index`、`honba`、`riichi_sticks` | 对应 Observation 字段 | 四条局况计数 token；局号写入时加一 |
| 活牌山 | history 中每个 `tsumo` | 一条局况计数 token，初值 70 |
| `oya` | `Observation.oya` | 庄家相对座位 token |
| 自风 | `player_id` 与 `oya` | 自风 token，取值 `1..4` |
| `dora_indicators` | `Observation.dora_indicators` | 每张明宝牌一条公开牌 token |
| `hand` | `Observation.hands[player_id]` | 按去赤 34 种牌汇总；每个非零牌种一条手牌计数 token，另带是否含赤五 |
| `drawn_tile` | `Observation.drawn_tile` | 有摸牌时一条单独的公开牌 token |
| `riichi_declared[player_id]` | `Observation.riichi_declared` | 写入自身状态 flag 的立直位 |
| `decision_flags` | 当前合法动作类型 | flag 的“主动决策窗口”位；Python 对 `dahai`、`reach`、`ankan`、`kakan`、`ryukyoku` 任一存在时置位 |

后缀不会重复河牌、副露、立直过程，因为这些信息已由完整 history 恢复。三个对手各生成一条闭手 mask token；不会编码其闭手牌、摸牌或里宝。

## 4. Token 字段与数值通道

`token_factors` 的每行固定为十个 `uint8` 分类因子：

```text
(segment, kind, field, seat, tile_suit, tile_rank, tile_red,
 count_or_source, flag, visibility)
```

所有分类字段的 `0` 表示未设置；整行全零是 batch padding。模型嵌入表容量依次为 `(8, 32, 256, 8, 8, 16, 4, 16, 256, 4)`。

| 字段 | 编码与用途 |
|---|---|
| `segment` | `1=history 事件`，`2=当前状态` |
| `kind` | `1=事件`，`2=分数`，`3=局况计数`，`4=牌计数/单牌`，`7=对手闭手 mask` |
| `field`（事件） | `2=start_kyoku`，`4=dahai`，`5=chi`，`6=pon`，`7=daiminkan`，`8=ankan`，`9=kakan`，`10=dora`，`11=reach`，`12=reach_accepted` |
| `field`（状态） | 分数为 `1`；局况依次为 `1=场风`、`2=局号`、`3=本场`、`4=供托`、`5=活牌山`、`6=庄家`、`7=自风`、`8=自身 flag`；牌字段为 `1=手牌`、`3=明宝牌`、`5=摸牌`；闭手 mask 为 `1` |
| `seat` | 事件 actor、来源、分数所属者或状态所属者的相对座位；无座位时为 `0` |
| `tile_suit`、`tile_rank`、`tile_red` | 牌分解为 `m/p/s=1/2/3`、字牌=4；数牌 rank 为 `1..9`，字牌为 `1..7`；赤五为 1 |
| `count_or_source` | 事件中保存来源座位；牌计数保存张数；吃保存被鸣牌在顺子中的位置与赤五组合；碰和杠保存是否含赤五 |
| `flag` | 弃牌保存 `1=手切`、`2=摸切`；开局保存场风/局号组合；自身状态的 bit 0/1/2 依次为已立直/有摸牌/主动决策窗口 |
| `visibility` | `1=公开`，`2=隐藏`；仅对手闭手 mask 使用隐藏值 |

`token_numeric` 与分类行一一对应，形状为 `[B, L, 8] float32`。只有分数和局况计数写入数值特征：

- 分数使用周期 `(100, 1_000, 10_000, 100_000)`；
- 局况计数使用周期 `(2, 8, 32, 128)`；
- 每个周期依次写入 `sin(2πx/p)`、`cos(2πx/p)`，组成八维；其他 token 的八维均为零。

## 4.1 集中式 critic 的额外输入

`critic_factors` 同样为十因子 `uint8` 行，但独立于 actor 的 `token_factors`，不带 numeric 通道。其 `segment=4`，并且只由 value 支路嵌入：

- 三家对手的闭手牌：`kind=4, field=2`，按相对座位、牌种、赤五状态分别计数；普通五和赤五必须是不同 token，不能折叠。
- 可选公开投影：启用 `critic_include_public_state` 后，`field=3` 为按牌种/赤五聚合的牌河，`kind=5, field=2` 为副露头，`field=4` 为副露组成牌计数。该投影刻意不保留牌河顺序、手切/摸切。

该分支可以利用训练时完整 Observation 的四家闭手信息，但绝不能改变 actor 输入、策略 logits 或部署时的策略信息边界。

## 5. Python/Rust 边界与形状

`prepare_decisions(batch_indices, legal_action_jsons, snapshot_jsons)` 返回：

```text
token_factors      [B, L, 10] uint8
token_numeric      [B, L, 8]  float32
token_lengths      [B]        int64
action_mask        [B, 241]   bool
history_generation [B]        int64
```

`B` 是本次决策行数，`L` 仅等于本 batch 中最长有效 token 长度；短行右侧以零 padding。`token_lengths` 不含模型内部追加的 learned query，因此模型实际输入长度为 `token_lengths + 1`。query 放在每行第 `token_lengths` 个位置，最大实际长度为 4096；Rust 状态机和 Python 模型都会拒绝超长输入，不截断 history。

`history_generation` 随同局公开 history 变化，用于调用方识别可复用的 history 前缀；当前状态后缀始终由本次快照重建。`action_mask` 必须至少有一个合法槽位，且每个打开槽位必须对应传入的当前 MJAI 合法动作。

Python 在提交前将 RiichiEnv 的物理牌号转为 MJAI 牌字符串，并为 `dahai` 补上环境未写入 JSON 的 `tsumogiri`。状态机只消费已由环境广播确认的事件；模型选择不会直接写入 history。

## 6. 错误边界

边界层会拒绝非法 MJAI JSON、非法牌、错误座位、重复或越界 batch 索引、错误张量形状、空合法 mask 和超长上下文。解码后若 RiichiEnv 拒绝 MJAI 动作，bridge 立即报错；这表示合法掩码、动作映射或环境窗口不一致，不能静默替换为其他动作。
