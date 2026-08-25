# Tasks: V17 PPO 原生性能重构

按依赖排序;每项为「每主题一个 commit + 测试通过 + 可独立回滚」粒度。

## 已完成

- [x] T1 冻结基线:创建标准 512g4e 基准配置
      (`riichi_ppo_v1/configs/v17_ppo_perf_512g4e{,_noso}.yaml`)。
- [x] T2 学习者一次性物化:`riichi_ppo_v1/training/rollout_buffer.py`
      (`RolloutBuffer`) + `PPOLearner.update` 接入 `update_use_soa` 开关 +
      `tests/unit/test_rollout_buffer.py`。
- [x] T3 golden trace 冻结脚本:`audit/reports/v17/scripts/freeze_golden_baseline.py`
      (纯 CPU,冻结 V16 编码 oracle 输出)。
- [x] T4(b) V16 action query 批量生成:`analyze_action_queries_batch`
      (`riichi_ppo_v1/model/action_query.py`)+ `bridge.prepare_v16` 的
      `batch_query` 开关 + `tests/unit/test_action_query_batch.py`。与逐动作
      oracle 逐元素一致;`analyze_action_queries` ~119→101 μs/action,
      `prepare_v16` 端到端 ~1.14×。**仍待与 convolution 并发/GPU 深挖**。

## 本轮完成

- [x] T4 Rust V16 批量编码(`riichienv-state-machine` 增加 `encode_v16_batch` /
      compact observation 入口),消除逐动作 PyO3 与 JSON 往返。
- [x] T5 粗粒度 Observation/Action 边界:`riichienv.prepare_v16_compact_facts`
      每个唯一 Observation 只跨一次 PyO3,在 core 内直接物化 20 个连续 SoA
      buffer;state-machine 直接返回 action id→原始 Action 下标,删除 query 热路径
      的 decode JSON→canonical JSON→PyObject 匹配往返。
- [x] T10 完整正确性回归:全部 action kind 合成回归、真实 env bridge 全字段
      oracle 对照、golden trace 69,733 元素与 decode/边界 JSON 全等;
      `riichi_ppo_v1/tests/unit` 164 项通过。
- [x] T11 三轮基准(标准 512g4e)+ 实际 2048 配置验证 + 最终报告。

## 后续可选深化(本次 rollout ≥1.20× 验收不再阻塞)

- [ ] T6 direct action-id step:`bridge.decode` 之后由 Rust 直接应用 action id
      驱动 `env.step_batch`。
- [ ] T7 紧凑 rollout buffer:worker 内 pending 从 `list[Transition]` 改 SoA;
      driver→learner 改 Ray 大数组/shared memory;记录 RPC/序列化/object-store
      /materialization 时间与峰值内存。
- [ ] T8 PPO update 深化:pinned host slabs、non_blocking H2D 并验证重叠;
      评估 DDP static graph / fused AdamW / torch.compile / bf16 autocast;
      两个 DDP rank 等量 shard。
- [ ] T9 并发拓扑 sweep:T1/T2/T3 三组,测量 inference batch 分布、queue wait、
      GPU/actor idle、CPU run queue、线程过度订阅、active decision rows。
