# Spec: V17 PPO 原生性能重构

## 概述

在严格保持 V16 信息编码协议、V17 训练语义与计算量的前提下,重构 rollout 与
PPO update 的性能关键路径:Rust 融合执行、紧凑 SoA 数据通路、学习者一次性
物化与并发拓扑优化。目标:`rollout_wall_s`、`update_wall_s` 与两者之和各
加速 ≥1.20×(争取 1.30×)。

## 非目标 / 禁止项

- 不减少 games、epochs、transitions、minibatches、optimizer steps、模型
  forward/backward、SFT KL、GRP、critic、指标或有效输入。
- 不做参数微调式的"性能调整";只做结构性重构。
- 不启动正式长期训练。

## 现状与瓶颈

见 `audit/reports/v17/design/v17-ppo-native-performance-design.md` 的 1.2 节。
两大支点:(A) rollout 的 V16 query/snapshot/critic 编码;(B) learner 的逐样本
host collate / H2D。

## 需求(功能项)

1. **Rust 融合快速路径**:消除高频 MJAI JSON serialize/parse;消除大量
   Observation PyObject/dict/list/字符串;把 `analyze_action_queries`、query row、
   snapshot、critic feature 编码迁移或等价实现到 Rust;inference 返回 action id
   后由 Rust 直接应用;每个 batch tick 只保留少量粗粒度 PyO3 调用;Rust 计算释
   放 GIL 并优先持久线程池;返回连续 SoA NumPy/共享内存 buffer。
2. **紧凑 rollout buffer**:不再以百万级 `list[Transition]` 为主要数据格式;实现
   至少一种紧凑方案(Rust-owned trajectory SoA / flat numpy+offsets / Ray
   object store 大数组 / shared memory ring buffer / learner 直接消费 shard);
   记录 Ray RPC 次数、序列化字节数、object-store 时间、driver materialization
   时间与峰值 host 内存。
3. **rollout 并发重构**:独立控制 rollout actor 数、每 actor 逻辑环境数、每 actor
   Rust 线程数、inference actor/queue/batch、PyTorch/OpenMP/MKL/Rayon 线程;
   至少三组明显不同拓扑并测量 inference batch 分布、queue wait、GPU/actor idle、
   CPU run queue、线程过度订阅与 active decision rows;尝试双缓冲/流水线。
4. **PPO update 优化**:rollout 完成后只 materialize 一次;length/bucket/offset/
   padding plan 跨 epoch 复用;shuffle 只重排 index;预分配 pinned host slabs 或
   GPU/bucket-resident buffer;向量化 gather/copy;non_blocking H2D 并与计算重叠;
   两个 DDP rank 直接消费等量 shard。实测 DDP `find_unused_parameters`、static
   graph、bucket、fused AdamW、`torch.compile`、bf16/autocast、reference forward
   与各项 GPU 同步。
5. **正确性验证**:冻结旧路径为 oracle,固定 seed 生成 golden trace,比较
   action opportunities、legal action id/mask、history/snapshot/query/critic
   tensor、action id↔env action、state-machine events、end_kyoku/end_game、
   scores/ranks、transitions、rewards/advantages/returns、done/GAE boundary。
6. **基准**:标准基线 `CUDA_DEVICE=0,1`、`learner_gpus=2`、`target_kl=0.0`、
   `update_epochs=4`、`games_per_update=512`、相同 seed、3 轮(第 1 轮预热);另用
   `v17_ppo.yaml` 实际完整参数验证一次。

## 验收标准

- 标准基线下 `rollout_wall_s`、`update_wall_s`、两者之和均 ≥1.20×。
- 正确性:整数/离散 tensor 逐元素一致;浮点给出严格容差与误差上界。
- 所有新增/修改代码附测试;协议契约变更同步协议文档。
- 报告含优化前后调用链与架构、profiling 与瓶颈排序、Rust API 和 buffer layout、
  并发拓扑选择、每项优化独立收益、正确性证据、三轮基准对比、剩余瓶颈与推荐配置。
