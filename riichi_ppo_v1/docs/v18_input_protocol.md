# V18 输入协议（encoding protocol 18）

V18 是唯一活跃输入契约。Actor 序列为 Objective Facts、固定 29 行 Atomic
Snapshot、每个合法动作一对 Offense/Defense Query；Critic 在共享公共表示之后单独
读取三家闭手与未来五张牌。V16/V17 文档与产物仅作冷存储。

## 1. 张量契约

| 张量 | 形状 | 语义 |
| --- | --- | --- |
| `token_factors` | `[B,L,10] uint8` | 有序 Objective Facts |
| `token_numeric` | `[B,L,8] float32` | Facts 的连续通道 |
| `token_lengths` | `[B] int64` | 每行有效 Facts 数 |
| `snapshot_factors` | `[B,29,4] uint8` | `(field_id,relative_seat,categorical,tile)` |
| `snapshot_numeric` | `[B,29,1] float32` | 仅 score pressure 非零 |
| `snapshot_lengths` | `[B] int64` | 必须全部为 29 |
| `query_rows` | `[B,2A,15] int64` | 每动作连续两行 Query |
| `query_action_ids` | `[B,A] int64` | 有效前缀唯一且无序 |
| `legal_mask` | `[B,241] bool` | action-ID 集合须与 Query 完全相等 |

Snapshot schema 只在 Rust `atomic_snapshot.rs` 定义，Python 通过
`riichi.atomic_snapshot_schema()` 读取，不复制第二份字段表。

## 2. Atomic Snapshot 固定顺序

离散值 `0` 只用于字段规定的 N/A；numeric 必须有限。score pressure 为
`clip((self_score-opponent_score)/100000,-1,1)`。

| 行 | 字段 | 座次 | 域 |
| ---: | --- | ---: | --- |
| 1 | own_rank | 0 | 1..4 |
| 2–4 | score_pressure_1..3 | 1..3 | numeric -1..1 |
| 5/10/15 | opponent N riichi_status | 1/2/3 | 1=未立直，2=宣言，3=已接受 |
| 6/11/16 | opponent N riichi_turn | 1/2/3 | 0=N/A，1..25；与 status 联动 |
| 7/12/17 | opponent N open_meld_count | 1/2/3 | 0..4 |
| 8/13/18 | opponent N tedashi_count | 1/2/3 | 0..25，溢出截断 |
| 9/14/19 | opponent N tsumogiri_count | 1/2/3 | 0..25，溢出截断 |
| 20 | overall_shanten | 0 | 0=N/A，1=和牌，2..8=0..6 向听 |
| 21 | standard_shanten | 0 | 同上 |
| 22 | chiitoitsu_shanten | 0 | 同上；开手为 N/A |
| 23 | kokushi_shanten | 0 | 0=N/A，1=和牌，2..15=0..13 向听 |
| 24–26 | opponent N latest_tedashi | 1..3 | tile 0=N/A，1..37=规范牌码 |
| 27–29 | opponent N tsumogiri_streak | 1..3 | 0..4，溢出截断 |

相对座次固定为 1=下家、2=对面、3=上家。名次同分时按绝对座次稳定排序。
非法观察者、牌计数、面子数、立直状态/巡目组合或 schema 顺序必须被拒绝。

## 3. Query 行与动作 metadata

每行布局固定为：

```text
(query_type, action_id, action_type, primary_tile, source_seat,
 answer_0, ..., answer_9)
```

同一动作的 Offense 行在 `2*i`，Defense 行在 `2*i+1`；二者 action ID、动作类型、
主牌和来源座次必须完全一致。`chi/pon/daiminkan/ron` 的 `source_seat` 必须是
1..3，其余动作必须为 0=N/A。来源由原生 Observation 的最后供牌者计算，不能由
字符串猜测。`primary_tile` 使用 0=N/A、1..34=去赤牌类。

动作类型编码明确区分 `tsumo` 与 `ron`，因此 supplier 域可以 fail closed：
`chi/pon/daiminkan/ron` 必须为 1..3，所有其他动作必须为 0=N/A。

Query 的 10 个 answer slot 沿用 O0–O9/D0–D9 麻将分析语义；枚举与基数的唯一
Python 来源为 `model/encoding_protocol.py`。任何越界 answer、缺对、重复 action
ID、metadata 不一致或 mask 集合差异都必须 fail closed。

## 4. 结构化注意力与 logits 映射

3 个 Shared GQA 层只处理公共 Facts + Snapshot。第 4 个 Actor 层中：

- 公共 token 只按因果顺序看公共 token，永远看不到 Query；
- 每个 Query 可看全部有效公共 token，以及自己动作对的两行；
- 不同动作对互不可见；同一对共享两个局部 position ID；
- pair 的输入排列不会改变按 action ID 对齐后的 raw logits。

唯一输出映射是 `raw_policy_logits[b, action_id]`。不存在 Q scorer、candidate-Q、
Top-K Q boosting 或任何同义输出。

## 5. Actor/Critic 信息边界

Actor 只能读取观察者可见的事件、自身手牌/摸牌、公开场况、29 字段 Snapshot 和
合法动作 Query。它不能读取对手闭手、对手摸牌、未来牌山、里宝或 Critic token。

Critic 不读取动作 Query。它在公共共享状态之后按严格顺序读取：相对座次 1、2、3
的真实闭手 segment，各自明确编码普通五/赤五；随后必须恰好读取未来牌山位置
1..5，再追加 value query。缺段、错序、多牌、少牌或伪造占位全部拒绝。

测试必须证明：只改变隐藏信息时 Actor raw logits 逐 action ID 不变，而改变有效
private hand/future wall 会改变 Critic value。

## 6. 模型与持久化

固定拓扑为 `d_model=256`、16 Q heads、4 KV heads、head dimension 16、FFN 704、
3 Shared + 1 Actor + 2 Critic，参数量范围 4.9M–5.1M。checkpoint 只接受 V18
`isolated_action_query` 配置与精确 state keys；PPO format 为 4。Actor-only SFT
artifact 仅保存 Actor 范围参数并 strict load，不提供旧版本兼容。

encoded manifest format 为 `riichi-sft-encoded-v18`。协议 hash、Snapshot schema、
参数范围、Query/mask 语义和信息隔离均由生产校验器与测试共同验证。
