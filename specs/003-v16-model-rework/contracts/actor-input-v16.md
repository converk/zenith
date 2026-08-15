# Contract: V16 Actor 输入协议(encoding_protocol_version = 16)

本契约取代 v13 的 token schema / feature schema / rust-analysis / decision-
analysis 多版本拆分,是 V16 Actor 输入的唯一权威版本。任何实现与本表不一致即
视为语义失败。配套文档:`riichi_ppo_v1/docs/v16_input_protocol.md`(实现期同步)。

## 1. 输入结构

```text
Actor 输入 = Objective Facts(历史事件 + 自身手牌 + 摸牌)
          + Compact Snapshot
          + 每个合法动作一对(Offense Query, Defense Query)
```

Snapshot 不含三家完整牌河与完整副露序列(已由历史事件覆盖);全部 Derived
Features(v13 的 hand/value/placement/threat 摘要与候选 query 编码)删除。

## 2. Snapshot 字段

| 组 | 字段 | 编码约定 |
|----|------|----------|
| 基础场况 | 场风、局数、庄家(相对)、本场、立直棒、剩余牌数 | 局数/庄家等类别字段用
categorical;本场/立直棒/剩余牌数用连续归一化 |
| 基础场况 | 宝牌指示 | 每张指示一张公开牌 token(多重指示保留多重 dora) |
| 基础场况 | 四家当前点数、当前顺位 | 点数连续归一化;顺位 categorical 1..4 |
| Score Pressure | 自身点数 − 对手 1/2/3 点数 | 3 个连续相对分差(符号固定:
self − opponent) |
| Opponent Summary | 每对手 7 token:是否立直、立直巡目、副露数、是否门清、
舍牌数、手切次数、摸切次数 | 共 21 token;从公开 MJAI 状态计算,不含任何预测 |

## 3. Action Query 结构

每个合法动作固定生成 Offense Query 与 Defense Query,两者字段结构相同:

```text
(query_type, action_id, action_type, primary_tile, source_seat,
 answer_0 … answer_9)
```

每 Query 聚合为 **一个** Transformer token:

```text
E_q = E_action + E_queryType + Σ_i E_{type,i}(answer_i)
    → LayerNorm / Projection → d_model = 256
```

## 4. Offense Query(O0–O9)

语义:「执行该动作后,我的手牌进攻状态」。

| Slot | 问题 | 值域(基数) | N/A 规则 |
|------|------|-----------|----------|
| O0 | 动作后向听数 | AGARI, 0, 1, 2, 3, 4, 5+ (7) | 无 N/A;和牌动作取 AGARI |
| O1 | 有效牌种类数 | 0–9, 10+ (11) | 无 N/A;终局动作取 0 |
| O2 | 有效牌剩余总枚数 | 0, 1-4, 5-8, 9-12, 13-16, 17-20, 21+ (7) | 无 N/A;
终局动作取 0 |
| O3 | 合法等待牌种类数 | N/A, 1…13 (14) | 非听牌(含和牌后)为 N/A |
| O4 | 当前等待是否有役 | N/A, NO_YAKU, PARTIAL_YAKU, ALL_YAKU (4) | 非听牌为 N/A |
| O5 | 基础番数范围 | N/A, 1, 2, 3, 4, 5+ (6) | 非听牌为 N/A;只统计可确定基础番,
不含里宝/一发/海底 |
| O6 | 是否振听 | N/A, NO_FURITEN, PERMANENT_FURITEN, TEMPORARY_FURITEN (4) | 非听牌为
N/A |
| O7 | 是否门清 | YES, NO (2) | 无 N/A |
| O8 | 是否满足立直条件 | YES, NO, N/A (3) | 动作已终局/已立直/不适用时 N/A |
| O9 | 保留宝牌/赤牌数 | 0, 1, 2, 3, 4, 5+ (6) | 无 N/A;统计动作后手牌中实际持有的
dora+aka |

## 5. Defense Query(D0–D9)

语义:「执行该动作后,面对三个对手的防守性质」。

| Slot | 问题 | 值域(基数) | N/A 规则 |
|------|------|-----------|----------|
| D0/D1/D2 | 候选打牌是否为对手 1/2/3 现物 | GENBUTSU, NOT_GENBUTSU, N/A (3) |
无实际打牌的动作为 N/A |
| D3/D4/D5 | 候选打牌是否对对手 1/2/3 构成筋 | SUJI, NOT_SUJI, N/A (3) | 同上 |
| D6/D7/D8 | 动作后手中对手 1/2/3 现物剩余张数 | 0, 1, 2, 3, 4+ (5) | 无 N/A |
| D9 | 候选牌公开出现张数 | 0, 1, 2, 3, 4, N/A (6) | 无主牌的动作取 N/A |

不输入人工危险度评分;风险大小由网络自行学习。

## 6. 终局/边缘动作约定(spec.md Assumptions A5)

- 自摸/荣和:O0=AGARI;O1=0;O2=0;O3/O4/O5/O6/O8=N/A;O7=实际门清;O9=实际保留
  dora/aka;D0–D5=N/A;D6–D8=0;D9=主牌公开出现数。
- 流局类动作:Offense 按动作后手牌现状计算;非听牌时 O3–O6=N/A;O8=N/A;
  D0–D5=N/A;D6–D8 按手中现物计算;D9=N/A。
- 立直宣告:本身是打牌,D0–D5 按宣告牌计算;O8=N/A;O3–O6 按立直后手牌。
- 吃/碰/杠:D0–D5=N/A;D6–D8 按动作后手牌计算;D9=主牌公开出现数。

## 7. 不变量

1. Query 只含可从当前真实局面确定计算的客观答案;禁止预计和率/放铳率/EV/最终
   得点/人工危险度/手牌综合评分。
2. Actor 输入不含任何隐藏信息;三家对手手牌与后续牌山只出现在 Critic 输入。
3. 所有 categorical 因子取值必须在声明的 cardinality 内(单一来源常量)。
4. 每个合法动作恰一对 query,query 与 action_id 一一对应;序列长度不因 slot 数
   增加而增加(每 Query 仍为一个 token)。

