# V18 输入协议（encoding protocol 18）

V18 是唯一活跃输入契约。Actor 序列为 Objective Facts、固定 54 行 Atomic
Snapshot、每个合法动作一对 Offense/Defense Query；Critic 在共享公共表示之后单独
读取三家闭手与未来五张牌。V16/V17 文档与产物仅作冷存储。

## 1. 张量契约

| 张量 | 形状 | 语义 |
| --- | --- | --- |
| `token_factors` | `[B,L,10] uint8` | 有序 Objective Facts |
| `token_numeric` | `[B,L,8] float32` | Facts 的连续通道 |
| `token_lengths` | `[B] int64` | 每行有效 Facts 数 |
| `snapshot_factors` | `[B,54,4] uint8` | `(field_id,relative_seat,categorical,tile)` |
| `snapshot_numeric` | `[B,54,1] float32` | 仅 score pressure 非零 |
| `snapshot_lengths` | `[B] int64` | 必须全部为 54 |
| `query_rows` | `[B,2A,15] int64` | 每动作连续两行 Query |
| `query_action_ids` | `[B,A] int64` | 有效前缀唯一且无序 |
| `legal_mask` | `[B,241] bool` | action-ID 集合须与 Query 完全相等 |

Snapshot schema 只在 Rust `atomic_snapshot.rs` 定义，Python 通过
`riichi.atomic_snapshot_schema()` 读取，不复制第二份字段表。

Objective Facts 的状态后缀只携带可从决策快照精确重建的自身/公开局况：场风、局数、
本场、立直棒、庄家/自风、状态 flags、当前巡目（已舍牌轮数+1）、宝牌指示牌、自身手牌
与摸牌。**剩余牌山数
不入编码**——它无法由 MJAI 事件流精确推导，旧估计值（固定 70 起、仅按摸牌递减）
已删除，序列长度因此每决策固定减少一行。

## 2. Atomic Snapshot 固定顺序

离散值 `0` 只用于字段规定的 N/A；numeric 必须有限。score pressure 为
`clip((self_score-opponent_score)/100000,-1,1)`。

| 行 | 字段 | 座次 | 域 |
| ---: | --- | ---: | --- |
| 1 | own_rank | 0 | 1..4 |
| 2–4 | score_pressure_1..3 | 1..3 | numeric -1..1 |
| 5/18/31 | opponent N riichi_status | 1/2/3 | 1=未立直，2=宣言，3=已接受 |
| 6/19/32 | opponent N riichi_turn | 1/2/3 | 0=N/A，1..25；与 status 联动 |
| 7/20/33 | opponent N open_meld_count | 1/2/3 | 0..4；暗杠不计开手 |
| 8/21/34 | opponent N tedashi_count | 1/2/3 | 0..25，溢出截断 |
| 9/22/35 | opponent N tsumogiri_count | 1/2/3 | 0..25，溢出截断 |
| 10/23/36 | opponent N post_riichi_tsumogiri_count | 1..3 | 0..15，16=16+；未立直为 0，**不含宣言牌本身** |
| 11–14 / 24–27 / 37–40 | opponent N first_six_{man,pin,sou,terminal_honor}_count | 1..3 | 每项 0..6；不足六张自然为 0 |
| 15/28/41 | opponent N open_meld_yakuhai_han | 1..3 | 0..5，6=6+；连风按场风和自风分别计番 |
| 16/29/42 | opponent N visible_meld_dora_aka_han | 1..3 | 0..7，8=8+；普通宝牌与赤宝牌分别计番 |
| 17/30/43 | opponent N riichi_declaration_tile | 1..3 | tile 0=N/A，1..37=规范牌码（红五 5m/5p/5s=1/11/21，字牌 31..37） |
| 44–47 | overall/standard/chiitoitsu/kokushi shanten | 0 | 与原 V18 向听编码相同 |
| 48 | fully_visible_tile_kind_count | 0 | 0..24，25=25+ |
| 49 | unknown_distinct_dora_copy_count | 0 | 0..15，16=16+；相同宝牌种只计一次 |
| 50 | self_improve_tile_count | 0 | 0..39，40=40+；摸入后综合向听下降的剩余实体牌数（归一 13 张形） |
| 51 | self_win_tile_count | 0 | 0=N/A（非听牌），1..39，40=40+；听牌时可和牌剩余张数 |
| 52–54 | opponent N tsumogiri_streak | 1..3 | 0..4，溢出截断 |

相对座次固定为 1=下家、2=对面、3=上家。名次同分时按绝对座次稳定排序。
前六次舍牌的幺九计数与花色计数可重叠。新增值都是离散类别：`first_six`、
`post_riichi`、`fully_visible`、`unknown_dora` 的 `0` 是有效零计数；`self_win`
的 `0` 表示非听牌 N/A；`improve`/`win` 的溢出桶为 40=40+。每一行的 `field_id`
是稳定 schema ID。非法观察者、牌计数、面子数、立直状态/巡目组合或 schema 顺序
必须被拒绝。

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

Actor 只能读取观察者可见的事件、自身手牌/摸牌、公开场况、54 字段 Snapshot 和
合法动作 Query。它不能读取对手闭手、对手摸牌、未来牌山、里宝或 Critic token。

Critic 不读取动作 Query。它在公共共享状态之后按严格顺序读取：相对座次 1、2、3
的真实闭手 segment，各自明确编码普通五/赤五；随后必须恰好读取未来牌山位置
1..5，再追加 value query。缺段、错序、多牌、少牌或伪造占位全部拒绝。

测试必须证明：只改变隐藏信息时 Actor raw logits 逐 action ID 不变，而改变有效
private hand/future wall 会改变 Critic value。

新增 Snapshot 事实只从观察者自身手牌/副露、四家牌河、已表示的副露牌及已翻开的
宝牌指示牌导出；不读取对手闭手、真实牌山顺序或事后标签。暗杠不会被标记为开手，
但其已表示的牌可参与副露宝牌/赤宝牌统计。普通宝牌由当前所有指示牌推进得到；仅
全局“未知宝牌实体数”会对重复得到的宝牌种去重。

## 6. 模型与持久化

固定拓扑为 `d_model=256`、16 Q heads、4 KV heads、head dimension 16、FFN 704、
3 Shared + 1 Actor + 2 Critic，参数量范围 4.9M–5.1M。checkpoint 只接受 V18
`isolated_action_query` 配置与精确 state keys；PPO format 为 4。Actor-only SFT
artifact 仅保存 Actor 范围参数并 strict load，不提供旧版本兼容。

encoded manifest format 为 `riichi-sft-encoded-v18`。协议 hash、Snapshot schema、
参数范围、Query/mask 语义和信息隔离均由生产校验器与测试共同验证。
