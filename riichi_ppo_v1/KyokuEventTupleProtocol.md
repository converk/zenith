# Kyoku State Packet V4 协议

当前实现位于 `riichi/src/MjaiKyokuStateMachine/`。V4 替换 V3 的八维事件 tuple：模型输入是
可缓存的压缩公开事件块，以及每次决策重新生成的 12 个结构化局面 token。

```text
[confirmed event blocks ...] [12 board tokens] [learned decision token]
```

`decision token` 由模型内部生成。只有左侧的 confirmed event blocks 写入 KV cache；board
tokens 永远是临时后缀。

## 1. 可见性与生命周期

- 当前实现只支持 RiichiEnv 四麻。
- 对手闭合手牌、对手摸到的牌、对手振听等私有信息绝不输入模型。
- `tsumo` 由状态机消费，用于更新活牌山和当前局面，但不产生模型事件块。
- `hora`、`ryukyoku`、分差、里宝、`end_kyoku`、`end_game` 用于奖励、边界或离线原始 MJAI
  日志，不进入同局决策输入。
- 原始 MJAI 广播仍由 RiichiEnv 保存；V4 block 只是模型侧的紧凑、可审计投影。

## 2. 事件块

状态机内部以 64-bit append-only block 保存历史；Python 接口输出预解包的 `uint8` 字段，模型
不会把 64-bit 整数当作词表 id，也不需要 GPU 位运算。

### TURN_BLOCK

四个 16-bit micro-event 构成一个 block。每条 micro-event 导出：

```text
(kind, actor, tile, flag)
```

`kind` 为 `PAD / DAHAI / REACH / REACH_ACCEPTED / DORA`；`actor` 是相对座位；牌编码为
`NONE=0, 1m..C=1..34, 5mr/5pr/5sr=35..37`；`flag` 包含弃牌摸切标记。未满四条的普通事件
保留在临时局面状态，直到凑满或被副露事件 flush，因此不会修改已缓存 block。

### MELD_BLOCK

吃、碰、大明杠、加杠、暗杠各占一个 block，字段为：

```text
(meld_kind, actor, target_or_NONE, pai, tile0, tile1, tile2, tile3)
```

例如以 `SELF` 为视角，“上家碰对家的 3p”导出为：

```text
(PON, KAMICHA, TOIMEN, 3p, 3p, 3p, NONE, NONE)
```

普通五与赤五保持不同值；所有 actor/target 均在导出时转换为相对座位。

## 3. 十二个 board token

相对座位按 `SELF, SHIMOCHA, TOIMEN, KAMICHA` 排列，每家连续三个 token：

1. `PLAYER_STATE`：分数桶、立直/双立直/一发、最近手切；SELF 另含 37 种手牌计数、当前
   摸牌和自己的私有规则状态。
2. `RIVER`：最多 32 张弃牌及每张的摸切、立直弃牌、被鸣标记。
3. `MELDS`：最多四个面子的种类、完整组成、赤牌、来源和被鸣牌。

场风、自风、庄家、局号、本场、供托、活牌山、宝牌、首巡/岭上和最后弃牌作为全局条件重复
注入全部 12 个 token。每个 token 原始字段由独立的小词表 embedding 编码，再在 token
组内聚合；不会建立 64-bit 或组合事件的大词表。

分数使用 `floor(score / 5000)` 并在 85000 以上饱和；活牌山使用
`floor(remaining / 4)` 并在 68 以上饱和。

## 4. Python/Rust 边界

`prepare_decisions(batch_indices, legal_action_jsons, snapshot_jsons)` 返回：

```text
block_kinds      [B, L]       uint8
turn_fields      [B, L, 4, 4] uint8
meld_fields      [B, L, 8]    uint8
board_state      [B, 12, 160] uint8
block_lengths    [B]          int64
action_mask      [B, 241]     bool
history_generation [B]        int64
```

其中 `L` 只对当前 batch 的 event block 最大长度 padding。每行实际 Transformer 长度为
`block_lengths + 12`，模型再追加一个 learned decision token。
