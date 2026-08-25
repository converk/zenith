# Kyoku 状态协议

本文定义 RiichiEnv、MJAI 状态机、Python bridge 与 V18 模型的状态边界。动作空间和
MJAI 回转见 [KyokuActionSpace.md](KyokuActionSpace.md)，精确张量/schema 见
[v18_input_protocol.md](v18_input_protocol.md)。

## 数据流

```text
Observation.new_events() -> Rust 状态机 -> Objective Facts
Observation 当前公开/自身字段 -> 固定 54 行 Atomic Snapshot
Observation.legal_actions() -> 每动作 Offense/Defense Query pair + legal mask
三段公共输入 -> Actor raw logits[action_id]
公共共享表示 + 三家闭手 + 未来五张牌 -> Critic value（仅训练）
```

每张桌维护四个观察者状态机。同局 history 只追加观察者可见事件，新
`start_kyoku` 必须清空旧 history。Python bridge 只负责批处理、张量设备转换和
fail-closed 语义校验；Snapshot 字段顺序与域由 Rust 单一来源导出。

## 公开事件语义

绝对座次转换为观察者相对座次：自己=0、下家=1、对面=2、上家=3；无 supplier
时为 0=N/A（Query supplier 使用 1..3）。

| MJAI 事件 | 状态变化 |
| --- | --- |
| `start_game` | 设置观察者绝对座位 |
| `start_kyoku` | 清空 history，初始化局况、分数、手牌与公开宝牌 |
| `tsumo` | 活牌山减一；只有自己的摸牌进入 actor 当前事实 |
| `dahai` | 追加 actor、牌、赤五、手切/摸切 |
| `chi/pon/daiminkan` | 追加 actor、真实 target/supplier、组成与赤五 |
| `ankan/kakan` | 追加 actor、牌种、组成与赤五 |
| `dora` | 追加公开宝牌指示牌 |
| `reach/reach_accepted` | 追加立直过程与 actor |
| `hora/ryukyoku/end_kyoku/end_game` | 只形成奖励或生命周期边界 |

终局得点、里宝、对手摸牌和未知牌山组成不得进入 Actor facts。状态机消费环境已经
广播确认的事件；模型选择本身不会提前写入 history。

## 当前决策输入

Objective Facts 使用 `[B,L,10]` 分类因子和 `[B,L,8]` 连续通道表达有序公开事件、
自身手牌/摸牌、分数与局况。状态后缀只含场风、局数、本场、立直棒、庄家/自风、
状态 flags、当前巡目（已舍牌轮数+1，精确计数）、宝牌指示牌与自身手牌/摸牌；**不包含剩余牌山数**——它无法从 MJAI
事件流精确推导，旧估计值（固定 70 起、摸牌递减）已从 V18 输入删除。短行右侧零
padding，`token_lengths` 给出有效长度；超过 `context_tokens` 必须拒绝，不能截断。

Atomic Snapshot 始终为 `[B,54,4]` + `[B,54,1]`，表达自身名次、三家 score
pressure、每家立直/副露/手切/摸切摘要、立直后摸切数与立直宣言牌、前六张舍牌
花色与幺九统计、役牌/宝牌副露番、四种向听、全局可见四枚牌种/未知宝牌实体数、
自身进张与听牌和牌张数、摸切连打。字段不可合并为异质 token，也不可改变顺序。
新增统计只读取当前观察者合法已知区域，重复宝牌种在未知实体数中去重；暗杠不改
变门清，但其已表示副露牌可计入宝牌统计。进张/和牌张数基于归一十三张形与合法
已知区域的剩余实体牌，不使用任何估计量。

每个合法动作生成连续两行 Query。`chi/pon/daiminkan/ron` 的 supplier 必须来自
原生最后供牌者；其他动作必须为 N/A。Query action-ID 集合与 `bool[B,241]` legal
mask 的打开集合必须完全相同，不要求按 action ID 排序。

## 集中式 Critic

Critic 只复用 Query 之前的公共状态。其 private sequence 必须依次包含相对座次
1、2、3 的真实闭手，然后恰好五张未来牌山，最后是 value query。普通五与赤五须
可区分；任何 private token 都不能改变 Actor raw logits。

## 错误边界

非法 MJAI JSON/牌/座次、错误 shape/dtype、Snapshot schema 或域错误、空 mask、
重复/缺失 Query、supplier 错误、Critic private 段错序和超长上下文均立即报错。
RiichiEnv 若拒绝由 action ID 回转的动作，也必须报错，不能静默换成 fallback。

V16/V17 状态和输入文档仅作历史归档；活跃代码不提供加载、迁移或协议适配。
