# Tasks: V17 PPO 原生性能重构

按依赖排序;每项为「每主题一个 commit + 测试通过 + 可独立回滚」粒度。

## 已完成

- [x] T1 冻结基线:创建标准 512g4e 基准配置;临时逐对象对照配置在单路径
      验收后删除。
- [x] T2 学习者一次性物化:`riichi_ppo_v1/training/rollout_buffer.py`
      (`RolloutBuffer`) + `PPOLearner.update` 单路径 +
      `tests/unit/test_rollout_buffer.py`。
- [x] T3 golden trace 冻结脚本:`audit/reports/v17/scripts/freeze_golden_baseline.py`
      (纯 CPU,冻结 V16 编码 oracle 输出)。
- [x] T4(b) V16 action query 批量生成:`analyze_action_queries_batch`
      (`riichi_ppo_v1/model/action_query.py`)+ `bridge.prepare_v16` 的
      `batch_query` 开关 + `tests/unit/test_action_query_batch.py`。与逐动作
      oracle 逐元素一致;`analyze_action_queries` ~119→101 μs/action,
      `prepare_v16` 端到端 ~1.14×。该过渡实现已在 T12 清理,不再存在于 PPO
      runtime。

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
- [x] T12 Rust-only 单路径清理:删除 PPO Python batch/逐动作回退、配置开关与
      旧路径测试;听牌 O4/O5 facts 重建迁入 Rust;golden/unit/integration/bot
      全部回归通过。
- [x] T13 V16 Action Query 语义单一来源:`analyze_action_queries` 改为 Rust
      融合编码器的兼容适配层,删除 Python 的动作分派、post-shape、向听、有效牌、
      防守与役种重复实现;SFT、在线审计和测试共用同一 Rust 语义。

## 本轮新增完成

- [x] T7 紧凑 rollout 返回:worker 在最终 GAE 后把 `list[Transition]` 压为
      flat+offsets SoA;driver 合并并让 learner/DDP 直接消费;旧对象路径保留为
      oracle。标准计分轮 Ray `result_get` 19.37s→0.0026s,数组数
      355.6 万→348,rollout 1.44×。
- [x] T9(a) CPU 线程/Ray 资源 A/B/C/D sweep:rollout actor 独立设置
      BLAS/OpenMP/PyTorch=1 与 `num_cpus=2`;GPU 三轮确认 step_threads=2 无收益并
      回退,正式保留 step_threads=4。
- [x] T14 返回链路 profiling:worker min/max/p50/p90、语义汇总、SoA pack、
      object-store 发布差距、driver get/merge、数组数/字节数、线程/上下文切换、
      games/kyokus/drain 分布全部落入 performance JSONL。
- [x] T15 正确性:SoA merge/select/重复 DDP 下标/round-trip/GAE/returns/loss
      逐元素或 float32 严格对照;unit 167、integration/protocol 34、Rust 10 通过。
- [x] T16 标准 512g4e 三轮与真实 2048 验证:最终 50.684/130.099/181.274s;
      2048 为 2078 games、1,610,326 transitions、2/2 epochs、2100/2100
      minibatches、3,220,652 executed samples。
- [x] T17 SoA 单路径清理:删除 worker/driver/learner/DDP 的旧对象 fallback、配置
      开关、对象恢复/逐对象 collate oracle 与三份临时对照配置;测试改为直接验证
      SoA 契约和冻结数学公式。

## 后续可选深化

- [ ] T6 direct action-id step:`bridge.decode` 之后由 Rust 直接应用 action id
      驱动 `env.step_batch`;当前仅约 0.20s/worker/512,不再是主瓶颈。
- [ ] T8 PPO update 深化:pinned host slabs、non_blocking H2D 并验证重叠;
      评估 DDP static graph / fused AdamW / torch.compile / bf16 autocast;
      两个 DDP rank 等量 shard。
- [ ] T9(b) actor 拓扑 sweep:T1/T2/T3 三组,测量 inference batch 分布、queue wait、
      GPU/actor idle、CPU run queue、线程过度订阅、active decision rows。
