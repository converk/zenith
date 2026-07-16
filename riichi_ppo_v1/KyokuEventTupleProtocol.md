# Kyoku State Packet V5 协议

V5 是 `RiichiEnv` 的 MJAI 广播到 PPO 模型之间的公开信息边界。它采用与 `exp/training`
一致的语义化因子 token，而不是 V4 的原始 event block 与 `12×160` board 字段。

```text
[append-only public history tokens ...] [current actor-state tokens ...] [learned query]
```

状态机仅生成前两段；模型追加 learned query，并同时从它输出固定 241 维策略 logits 与 value。

## 可见性与事件

- 仅支持 RiichiEnv 四麻。actor 和 value 只读取同一份公开信息，以及观察者自己的闭手与摸牌。
- 历史事件为 `start_kyoku`、`dahai`、`chi`、`pon`、`daiminkan`、`ankan`、`kakan`、`dora`、
  `reach`、`reach_accepted`。每个事件是一条 token，座位均转换为 `SELF/SHIMOCHA/TOIMEN/KAMICHA`。
- 弃牌保留牌种、赤五和手切/摸切；鸣牌保留 actor、来源、被鸣牌、赤五与可恢复的组成语义。
  吃的牌面、被鸣位置和赤五标记唯一确定顺子组成；碰/杠的牌面与赤五标记唯一确定组成。
- `tsumo` 只更新活牌山，且自己的摸牌由当前 `drawn_tile` token 表达。`hora`、`ryukyoku`、
  `end_kyoku`、`end_game` 只服务奖励和生命周期，不进入模型历史。新的 `start_kyoku` 清空历史。

## 当前状态

每次 decision 从 `DecisionSnapshot` 重建临时 state suffix，包含：

- 四家分数；场风、局号、本场、供托、剩余活牌；庄家与自风；全部明宝牌；
- 自己的手牌类型计数（赤五标记）、自己的当前摸牌、自己立直/是否有摸牌标记；
- 三家对手的闭手 mask token。

河牌、副露、鸣牌、立直过程不在 suffix 重复出现，均由同局完整 history 恢复。对手闭手、对手摸牌、
牌山组成、里宝与终局结算绝不在 token 中出现。合法动作不编码为候选 token，而是独立的 241 维 mask。

## 因子格式与 Python/Rust 边界

`prepare_decisions(batch_indices, legal_action_jsons, snapshot_jsons)` 返回：

```text
token_factors      [B, L, 10] uint8
token_numeric      [B, L, 8]  float32
token_lengths      [B]        int64
action_mask        [B, 241]   bool
history_generation [B]        int64
```

十个分类因子固定为：

```text
(segment, kind, field, seat, tile_suit, tile_rank, tile_red, count_or_source, flag, visibility)
```

`segment` 区分 history/state；牌使用 suit/rank/red 分解；分数和计数写入 8 维 Fourier 数值通道。
第 `L` 维仅按当前 batch 最长决策 padding，零是 padding。模型实际长度为 `token_lengths + 1`，
最大为 4096；超出时状态机和模型均报错，不截断历史。
