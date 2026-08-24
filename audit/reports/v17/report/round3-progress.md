# V17 原生性能重构 · Round 3 进度

日期:2026-08-24

## 并发拓扑 sweep(实测试点)

按任务「至少三组拓扑」要求,在 SoA+batch-query 基础上做了 inference 批尺寸
改造。

### T-bigbatch(12 worker × 32 env,inference_batch_target_rows=512 /
wait_ms=20 / target_workers=100)

| iteration | rollout | update | total |
|---|---|---|---|
| 1(预热) | 174.401 | 148.952 | 323.374 |
| 2 | 137.998 | 112.138 | 251.570 |
| 3 | 139.566 | 191.218 | 331.965 |
| 均值(2,3) | 138.782 | 151.678 | 291.767 |

vs 基线:rollout **0.93×(倒退)**,update 1.38×,total 1.17×。
vs 组合(best):rollout **0.87×**,total 0.93×。

**结论(negative result)**:`inference_rows_per_forward` 从 ~103 提到 ~146,
但**等待延迟上升**(wait_ms=20ms)反而拖慢 rollout。说明 rollout 关键路径
不是 GPU 利用率,而是 worker 的编码 + RPC 等待延迟。

### 组合(best,SoA + batch_query,默认 inference)仍为最优

vs 同条件基线:rollout 1.07×,update 1.40×,total 1.25×。

## 关键结论

- **rollout 1.2× 未達成**;瓶颈是 `state/v16_query_assembly`(~52s/迭代,offense
  `shanten` 内核 ~37μs/动作),批量只能降低 PyO3/Python 开销,无法减少 shanten
  计算本身。
- **更大 inference batch 无效**(延迟 > GPU 收益)。
- 后续唯一可行大杠杆 = **完整 Rust 融合**:把 post-shape 计算 + shanten 批内核 +
  快照/query/critic 编码整体迁入 Rust(而非逐动作),并消除 worker 编码段与
  RPC 等待的重叠开销。

## 下一步

- 完整 Rust 融合(`encode_v16_batch`):compact observation → Rust 批内核 →
  SoA 输出。
- 或在不能降低 shanten 计算的前提下,通过双缓冲/流水线让 worker 编码与 GPU
  前向真正重叠,削掉 queue_wait。
