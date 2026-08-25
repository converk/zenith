# V17 原生性能重构 · Round 2 进度

日期:2026-08-24

## 已完成

1. **V16 action query 批量生成(`analyze_action_queries_batch`)**
   - 背景:micro-profile 显示 `state/v16_query_assembly` 实测占 `prepare_v16`
     的 ~90%(~80s/迭代),其中 `_analyze_offense_row`(offense 内核)~42μs/动作、
     `_analyze_defense_row` 的 Python/numpy 组装 ~31μs/动作。
   - 实现:同一 observation 的不变量事实只算一次(`_observation_facts`);把所有
     动作的 offense/defense/shanten 内核调用汇聚为每内核 1 次 batch 调用,消除
     逐动作 numpy 组装与结果提取的 Python 开销。
   - 接入:`bridge.prepare_v16` 增加 `batch_query` 开关(default false),worker
     从配置 `v16_batch_query` 传入。
   - 正确性:`tests/unit/test_action_query_batch.py` 证明 `prepare_v16`(开/关)
     逐元素一致,`analyze_action_queries_batch` 与逐动作 oracle 完全一致
     (0 mismatch / 1233 actions;325 tensor 比较全一致)。
   - 性能:`analyze_action_queries` ~119→101 μs/action(≈1.18×);`prepare_v16`
     端到端 ~1.14×。

## 进行中

- 组合基准(SoA + `v16_batch_query`):历史使用的独立 bq 配置已在 Rust-only
  单路径清理中删除;对应日志与本报告继续作为阶段证据。
  运行三轮,完成后给出与同条件基线(`perf_512g4e_noso`)和
  SoA-only(`perf_512g4e_soa`)的对照。

## 关键结论(round 2)

- `v16_query_assembly` 的改动是**结构性**的(消除冗余事实计算与逐动作 PyO3
  组装),不是参数调整或 Python cache。
- 但 offense shanten 内核(~37μs/动作)是**计算下限**:批量只降低 PyO3/Python
  开销,无法减少 shanten 计算本身。要进一步提高 rollout,需要:
  a) 更大 inference batch 提高 GPU 利用率(当前 ~50 行/forward,利用率低);
  b) 后续把快照/历史/query 全量迁进 Rust(完整融合)。

## 下一步

- 组合基准结果与拓扑 sweep(大 inference batch)。
- 紧凑 rollout buffer(worker 内 SoA + Ray 大数组)。
- `v17_ppo.yaml`(2048 局)实际配置验证。
