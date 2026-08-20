# V16 Actor 输入协议(encoding_protocol_version = 16)

本协议取代旧 token schema / feature schema / rust-analysis /
decision-analysis 多版本拆分,是 V16 Actor 输入的唯一权威版本,与
`specs/003-v16-model-rework/contracts/actor-input-v16.md` 逐条一致。实现常量以
`model/encoding_protocol.py` 为单一来源。

## 输入结构

```text
Actor 输入 = Objective Facts(历史事件 + 自身手牌 + 摸牌)
          + Compact Snapshot
          + 每个合法动作一对(Offense Query, Defense Query)
```

Snapshot 不含三家完整牌河与完整副露序列(已由历史事件覆盖);旧的
hand/value/placement/threat 摘要与候选 query 编码已删除。

## Compact Snapshot

| 组 | 字段 | 编码约定 |
|----|------|----------|
| 基础场况 | 场风、局数、庄家(相对)、本场、立直棒、剩余牌数 | 场风 E/S、
局数 E1..S4、庄家相对为 categorical;本场/立直棒/剩余牌数为连续归一化 |
| 基础场况 | 宝牌指示 | 每张指示一张公开牌 token |
| 基础场况 | 四家当前点数、当前顺位 | 点数连续归一化;顺位 categorical |
| Score Pressure | 自身点数 − 对手 1/2/3 点数 | 3 个连续相对分差 |
| Opponent Summary | 每对手 7 token:是否立直、立直巡目、副露数、是否门清、舍牌数、
手切次数、摸切次数 | 共 21 token,只含公开 MJAI 状态 |

存储行统一为 `[kind, 4 categorical, 7 numeric]`,归一化刻度见
`encoding_protocol.py` 的 `SNAPSHOT_*_SCALE` 常量(编码期一次性应用)。

## Action Query

每个合法动作固定生成一对结构相同的 Query:

```text
(query_type, action_id, action_type, primary_tile, source_seat,
 answer_0 … answer_9)
```

每 Query 聚合为一个 Transformer token:

```text
E_q = E_action + E_queryType + Σ_i E_{type,i}(answer_i)
    → LayerNorm / Projection → d_model(V16-small 隐藏层 = 192)
```

> d_model 属于网络隐藏层配置,不属于输入协议字段;V16-small 将隐藏层从 256
> 降到 192,输入行宽、slot 基数与编码语义均不变,现有 V16 编码数据可直接复用。

存储行宽 `QUERY_ROW_WIDTH=15`,行字段下标为 `encoding_protocol.py` 的
`QUERY_ROW_*` 常量。

## Offense Query(O0–O9)

| Slot | 问题 | 值域(基数) | N/A 规则 |
|------|------|-----------|----------|
| O0 | 动作后向听数 | AGARI, 0, 1, 2, 3, 4, 5+ (7) | 无;和牌取 AGARI |
| O1 | 有效牌种类数 | 0–9, 10+ (11) | 无;终局取 0 |
| O2 | 有效牌剩余总枚数 | 0, 1-4, 5-8, 9-12, 13-16, 17-20, 21+ (7) | 无;终局取 0 |
| O3 | 合法等待牌种类数 | N/A, 1…13 (14) | 非听牌 N/A |
| O4 | 当前等待是否有役 | N/A, NO_YAKU, PARTIAL_YAKU, ALL_YAKU (4) | 非听牌 N/A |
| O5 | 基础番数范围 | N/A, 1, 2, 3, 4, 5+ (6) | 非听牌 N/A |
| O6 | 是否振听 | N/A, NO_FURITEN, PERMANENT_FURITEN, TEMPORARY_FURITEN (4) | 非听牌 N/A |
| O7 | 是否门清 | YES, NO (2) | 无 |
| O8 | 是否满足立直条件 | YES, NO, N/A (3) | 终局/已立直/不适用 N/A |
| O9 | 保留宝牌/赤牌数 | 0, 1, 2, 3, 4, 5+ (6) | 无 |

## Defense Query(D0–D9)

| Slot | 问题 | 值域(基数) | N/A 规则 |
|------|------|-----------|----------|
| D0–D2 | 候选打牌是否为对手 1/2/3 现物 | GENBUTSU, NOT_GENBUTSU, N/A (3) |
无实际打牌的动作为 N/A |
| D3–D5 | 候选打牌是否对对手 1/2/3 构成筋 | SUJI, NOT_SUJI, N/A (3) | 同上 |
| D6–D8 | 动作后手中对手 1/2/3 现物剩余张数 | 0, 1, 2, 3, 4+ (5) | 无 |
| D9 | 候选牌公开出现张数 | 0, 1, 2, 3, 4, N/A (6) | 无主牌的动作 N/A |

## 终局/边缘动作约定

- 自摸/荣和:O0=AGARI;O1=0;O2=0;O3/O4/O5/O6/O8=N/A;O7=实际门清;
  O9=实际保留 dora/aka;D0–D5=N/A;D6–D8=0;D9=主牌公开出现数。
- 流局类动作:Offense 按动作后手牌现状计算;非听牌 O3–O6=N/A;O8=N/A;
  D0–D5=N/A;D6–D8 按手中现物计算;D9=N/A。
- 立直宣告:本身是打牌,D0–D5 按宣告牌计算;O8=N/A;O3–O6 按立直后手牌。
- 吃/碰/杠:D0–D5=N/A;D6–D8 按动作后手牌计算;D9=主牌公开出现数。

## 不变量

1. Query 只含可从当前真实局面确定计算的客观答案;禁止预计和率/放铳率/EV/最终
   得点/人工危险度/手牌综合评分。
2. Actor 输入不含任何隐藏信息;三家对手手牌与后续牌山只出现在 Critic 输入。
3. 所有 categorical 因子取值必须在声明的 cardinality 内(单一来源常量)。
4. 每个合法动作恰一对 query,query 与 action_id 一一对应;每 Query 仍为一个
   token,序列长度不因 slot 数增加而增加。
