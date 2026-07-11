# KyokuTransformer 编码器设计

## 1. 作用与边界

本文件定义 `mahjong_model` 的 Transformer PPO actor-critic。模型消费
`KyokuEventTuple V3` 的单局 append-only 九维事件序列，并输出
`KyokuActionSpace V2` 的 241 维策略 logits 和一个状态价值。

相关接口文档：

- [KyokuEventTuple V3 输入协议](KyokuEventTupleProtocol.md)
- [KyokuActionSpace V2 输出动作空间](KyokuActionSpace.md)

模型不负责麻将规则判定。未来环境给出合法动作集合后，状态机将其转换为
`action_mask[B, 241]`；模型在构造 PPO 策略分布前用该 mask 屏蔽非法动作。

## 2. 总体结构

```text
input_ids[B, L, 9]
-> 九个字段各自 embedding 并相加，得到 token_embedding[B, L, 256]
-> input projection（mid: 256->320；large: 256->384）
-> 对每个有效序列末尾插入 learnable DECISION token
-> Pre-RMSNorm + RoPE full-attention + SwiGLU Transformer encoder
-> 取 DECISION hidden state
-> policy_head: Linear(d_model, 241)
-> value_head:  Linear(d_model, 1)
```

使用 encoder-only、full attention，而不是 causal attention。模型在当前决策时可以阅读本局
全部已确认历史；它的目标是选择动作，不是预测下一个事件 token。

## 3. 输入与九字段 embedding

输入张量：

```text
input_ids: int64[B, L, 9]
attention_mask: bool[B, L]
sequence_lengths: int64[B]
```

九个字段分别有独立 embedding 表：

| 字段 | 词表大小 | embedding |
|---|---:|---:|
| `TYPE` | 48 | `48 x 256` |
| `ACTOR` | 5 | `5 x 256` |
| `TARGET` | 5 | `5 x 256` |
| `TILE` | 39 | `39 x 256` |
| `TILE2` | 39 | `39 x 256` |
| `TILE3` | 39 | `39 x 256` |
| `VALUE` | 19 | `19 x 256` |
| `FLAG` | 32 | `32 x 256` |
| `STEP` | 18 | `18 x 256` |

对每个事件 token：

```text
e = E_TYPE + E_ACTOR + E_TARGET + E_TILE + E_TILE2 + E_TILE3
  + E_VALUE + E_FLAG + E_STEP
```

因此原始 token 表示固定为：

```text
e: float[B, L, 256]
```

起手手牌沿用输入协议：每种实际出现的牌写一个 `STATE_HAND(tile, count)` token，而不是
将同一种牌拆成多个无序的重复 token。后续摸、切、吃、碰、杠按确认事件顺序追加。

## 4. 位置编码：RoPE

模型使用 RoPE，不使用可学习绝对位置 embedding。RoPE 在每层 attention 的 Q/K 投影后应用：

```text
q = RoPE(Wq(x), position_id)
k = RoPE(Wk(x), position_id)
attention = softmax(q @ k^T / sqrt(head_dim)) @ Wv(x)
```

选择理由：

- 输入是带吃、碰、杠分支的事件流；同一巡在不同局面中可能产生不同数量 token，绝对 token
  下标不是稳定的规则语义。
- `STEP` 字段已经提供“第几巡”的绝对规则阶段；RoPE 则表达事件之间的相对距离。
- RoPE 没有可学习位置表的长度上限。训练仍应暂定 `max_seq_len=512` 以控制 full-attention
  的显存和计算，但超过该长度不会遇到未训练的位置 embedding。

每个样本的真实事件位置从 0 开始；`DECISION` token 的位置是该样本的
`sequence_length`。padding 不参与 attention，且不得改变短序列的有效 position id。

## 5. DECISION token

状态机保存的历史不包含 `DECISION` token。模型 forward 时，在每个样本真实 token 的末尾
插入一个可学习向量：

```text
[event_0, ..., event_(length-1), DECISION]
```

对 batch 内较短序列，`DECISION` 放在自己的有效末尾，而不是全 batch 的最右侧 padding 后。
最终取该位置的 hidden state 作为当前局面的统一表示。

## 6. Transformer 规格

两种模型的九字段 embedding 宽度都固定为 `d_embed=256`。模型尺寸通过输入投影、主干宽度、
层数和 FFN 宽度区分。

| 配置 | mid | large |
|---|---:|---:|
| `d_embed` | 256 | 256 |
| input projection | `256 -> 320` | `256 -> 384` |
| `d_model` | 320 | 384 |
| layers | 10 | 12 |
| heads | 10 | 12 |
| head dim | 32 | 32 |
| SwiGLU FFN dim | 800 | 1152 |
| norm | Pre-RMSNorm | Pre-RMSNorm |
| attention | full attention + RoPE | full attention + RoPE |
| dropout | 0.0 | 0.0 |
| 估算参数量 | 约 12.04M | 约 23.32M |

每层结构：

```text
x = x + FullAttentionWithRoPE(RMSNorm(x), attention_mask)
x = x + SwiGLU(RMSNorm(x))
```

SwiGLU：

```text
SwiGLU(x) = W_down(SiLU(W_gate(x)) * W_up(x))
```

PPO 初版设 `dropout=0.0`。策略采样时存在额外 dropout 随机性会使 rollout 的旧策略概率和
更新时重算概率不必要地不一致。

## 7. 输出与 PPO 接口

对 `DECISION` 的最终表示 `h[B, d_model]`：

```text
raw_policy_logits = Linear(d_model, 241)(h)
value = Linear(d_model, 1)(h)
```

模型输出：

```text
raw_policy_logits: float[B, 241]
policy_logits:     float[B, 241]
value:             float[B]
```

若有合法动作 mask：

```text
policy_logits = raw_policy_logits.masked_fill(~action_mask, -inf)
```

这与 PPO 现有接口保持一致：

```python
outputs = agent(input_ids, legal_mask,
                attention_mask=attention_mask,
                sequence_lengths=sequence_lengths)
distribution = Categorical(logits=outputs["policy_logits"])
```

为了兼容现有 PPO 调用形式 `agent(observations, legal_mask)`，若未显式传入
`attention_mask` 和 `sequence_lengths`，模型以 `TYPE != PAD` 推导有效位置和长度。
未来 MJAI 环境接入时应显式传状态机输出的两项元数据。

## 8. 参数量说明

参数量按当前词表、RoPE、直接线性 policy/value head 估算：

| 模块 | mid | large |
|---|---:|---:|
| 九字段 embedding | 约 0.06M | 约 0.06M |
| input projection | 约 0.08M | 约 0.10M |
| Transformer 主干 | 约 11.81M | 约 23.06M |
| policy/value head 与其他参数 | 约 0.09M | 约 0.10M |
| 合计 | **约 12.04M** | **约 23.32M** |

RoPE 只有频率 buffer，不引入可训练位置表参数。
