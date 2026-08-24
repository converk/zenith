# V17 原生性能重构 · Round 1 进度

日期:2026-08-24

## 已完成

1. **架构与热路径调研**:通读 rollout worker / inference / bridge / learner / DDP /
   RiichiEnv core+python+state-machine;确认两大优化支点:
   - rollout 的 `state/v16_query_assembly`(逐动作 Python + 2~3 次 PyO3/动作,
     实测在 8 env × 200 steps 上占 `prepare_v16` 的 ~90%,3.1s);
   - learner 的 `update/collate_host_padding_copy`(逐 Transition `torch.tensor`
     拷贝 + 同步 H2D)。
2. **学习者一次性物化(SoA buffer)**:
   - 新增 `riichi_ppo_v1/training/rollout_buffer.py`:`RolloutBuffer` 把整个
     rollout 一次性抽成 flat SoA(变长字段用 offset 索引),`collate(indices)`
     用向量化 gather 替代逐样本拷贝。
   - `PPOLearner.update` 接入 `update_use_soa` 开关(默认 true),旧路径保留为
     oracle。
   - 新增 `tests/unit/test_rollout_buffer.py`(3 项)。实测 host 拷贝从
     ~37.9ms/512 行降到 ~2.2ms/512 行(≈17×),且与 `materialize_host_batch`
     逐元素等价;`update_use_soa=true/false` 两大路径在同一权重起点下 loss
     一致。
3. **基准配置**:新增 `riichi_ppo_v1/configs/v17_ppo_perf_512g4e.yaml`(SoA 开)
   与 `v17_ppo_perf_512g4e_noso.yaml`(SoA 关),标准基线
   games_per_update=512 / update_epochs=4 / target_kl=0.0 / learner_gpus=2 /
   eval1v3_enabled=false。
4. **golden baseline 冻结脚本**:`audit/reports/v17/scripts/freeze_golden_baseline.py`
   (纯 CPU,冻结 V16 编码 oracle 输出;已产出
   `audit/reports/v17/eval/golden_baseline_v17.npz`)。
5. **设计/治理产物**:`audit/reports/v17/design/v17-ppo-native-performance-design.md`;
   `specs/007-v17-ppo-native-performance/{spec,plan,tasks}.md`。

## 标准 512g4e 三轮基准(SoA 开)结果

配置:`v17_ppo_perf_512g4e.yaml`;3 轮,第 1 轮预热,第 2/3 轮作统计。

| iteration | rollout_wall_s | update_wall_s | algorithm_wall_s |
|---|---|---|---|
| 1(预热) | 161.576 | 148.842 | 310.434 |
| 2 | 127.315 | 119.350 | 248.094 |
| 3 | 127.346 | 184.657 | 313.167 |
| 均值(2,3) | 127.331 | 152.004 | 280.630 |

与旧基线(`logs/v17/perf_base_true_512g4e.log`,第 2/3 轮均值
rollout=155.30 / update=185.79 / total=342.16)对比:

| metric | baseline(s) | new(s) | speedup |
|---|---|---|---|
| rollout | 155.303 | 127.331 | 1.22× |
| update | 185.789 | 152.004 | 1.22× |
| total | 342.160 | 280.630 | 1.22× |

说明:update 第 2 轮 119.4s(SoA host 拷贝从 ~105s 降到 ~8s),第 3 轮因 GPU
前向抖动(update_forward_s 27.5→47.1s)回到 184.7s;均值仍显著低于基线。

## 同条件基线(SoA 关)结果

同一机器条件、同一 512g4e 基准,仅关闭 SoA(`v17_ppo_perf_512g4e_noso.yaml`):

| iteration | rollout_wall_s | update_wall_s | algorithm_wall_s |
|---|---|---|---|
| 1(预热) | 171.503 | 233.001 | 404.519 |
| 2 | 125.632 | 163.102 | 290.133 |
| 3 | 133.116 | 256.029 | 390.223 |
| 均值(2,3) | 129.374 | 209.565 | 340.178 |

### 最终对照(同条件,SoA 开 vs 关,第 2/3 轮均值)

| metric | baseline(s) | SoA(s) | speedup |
|---|---|---|---|
| rollout | 129.374 | 127.331 | 1.02× |
| update | 209.565 | 152.004 | 1.38× |
| total | 340.178 | 280.630 | 1.21× |

**结论**:SoA 一次性物化显著改善 update(1.38×)与总时间(1.21×),但 rollout
基本不变(1.02×)。原因是 rollout 的 `state/v16_query_assembly`(实测 80s/迭代)
未受影响;**rollout 1.2× 目标尚未达成**,必须在下一轮实现 batched/Rust 融合的
V16 query/snapshot/critic 编码(micro-profile 显示其占 `prepare_v16` 的 ~90%)。

## 进行中 / 下一步

- 实现 batched/Rust V16 query 编码(消除逐动作 PyO3 与 JSON/dict 往返),
  目标让 rollout_wall_s ≥1.20×。
- 紧凑 rollout buffer(worker 内 SoA + Ray 大数组/shared memory)。
- PPO update 深化(pinned slabs、non_blocking H2D、DDP static graph)。
- 并发拓扑 sweep(T1/T2/T3)。
- `v17_ppo.yaml`(2048 局)实际配置验证。

## 下一步

- Rust 融合决策批(V16 query/snapshot/critic 编码迁移到 `riichienv-state-machine`,
  或 PyO3 批量汇聚),消除逐动作 PyO3 与 JSON/Python dict 往返。
- 紧凑 rollout buffer(worker 内 SoA + Ray 大数组/shared memory)。
- PPO update 深化(pinned slabs、non_blocking H2D、DDP static graph 等)。
- 并发拓扑 sweep(T1/T2/T3)。
- 完整正确性回归(golden trace)与三轮基准 + 报告。
