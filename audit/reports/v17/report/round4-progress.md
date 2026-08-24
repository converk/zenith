# V17 原生性能重构 · Round 4 进度

日期:2026-08-25

## 完成

1. **V16 action query 批量生成设为默认热路径**:worker 默认 `v16_batch_query=true`
   (旧逐动作 Python oracle 通过 `v16_batch_query=false` 回退)。
2. **`v17_ppo.yaml` 实际配置验证**(2048 局/update,2 epochs,target_kl=0.01,
   1 迭代):

   | 指标 | 值 | 旧 PROGRESS(约) |
   |---|---|---|
   | transitions | 1,612,024 | — |
   | games | 2,088 | 2,048 |
   | rollout_wall_s | 464.087 | ~408-450 |
   | update_wall_s | **276.544** | ~450-490 |
   | algorithm_wall_s | 740.673 | ~864 |
   | sps | 2176.43 | ~1550 |
   | epochs/minibatches | 2/2100 | — |

   结论:SoA 让真实 2048 配置的 **update ~1.65×**;rollout 与旧值同量级(受
   2088 局数影响,略高)。总时间 ~864→740.7s(1.17×,局数略多)。
3. **阶段性最终报告**:`audit/reports/v17/report/v17-native-performance-final-report.md`。

## 阶段总评(vs 同条件基线,标准 512g4e)

| metric | speedup |
|---|---|
| rollout | 1.07×(未达 1.2×) |
| update | 1.40×(达标) |
| total | 1.25×(达标) |

## 下一步

- 完整 Rust 融合(offense shanten 批内核 + 编码)与双缓冲/流水线削 queue_wait,
  把 rollout 推向 1.2×。
- 紧凑 rollout buffer、PPO update 深化(pinned slabs / non_blocking H2D /
  DDP static graph)。
