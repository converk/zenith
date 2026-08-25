# Plan: V17 PPO 原生性能重构

## 阶段总览

1. **冻结基线**(已完成):重新测量标准 512g4e 基线;生成 profiling 与 golden trace。
2. **学习者一次性物化**(已实现):`RolloutBuffer` SoA + `PPOLearner.update` 接入。
3. **Rust 融合决策批**(已完成):`riichienv` core 粗粒度紧凑事实物化 +
   `riichi.encode_v16_batch` 融合 offense/defense/shanten/query row;query 热路径
   消除 decode/canonical JSON 与逐动作 Observation 属性扫描;生产桥接只保留
   该路径,没有 Python runtime 回退或配置分支。SFT/审计保留的
   `analyze_action_queries` 公共接口也委托同一 Rust 编码器,Python 不再保存第二份
   Action Query 业务规则。
4. **紧凑 rollout 返回链路**(已完成):小局内暂存 `Transition` 完成 GAE,worker
   返回前压成 flat+offsets SoA;Ray 只传 29 个大数组/worker;driver 合并后 learner
   与双卡 DDP 直接消费 SoA,不恢复百万级对象;旧对象 fallback 与开关已在验收后
   删除。direct action-id step 依据新 profiling 保留为后续项。
5. **PPO update 深化**:pinned host slabs、non_blocking H2D、DDP static graph /
   fused AdamW / torch.compile 评估。
6. **CPU/Ray 资源 sweep**(已完成):A/B/C/D 三轮微基准 + C/D 标准 GPU 三轮;
   worker BLAS/OpenMP/PyTorch 限为 1,Ray 声明 2 CPU/worker;GPU 证实
   `env_step_threads=2` 无收益,正式值保持 4。T1/T2 actor 拓扑不再是当前主因。
7. **完整测试与三轮基准 + 真实 2048 + 报告**(已完成):最终相对 noso 基线
   rollout 2.55×、update 1.61×、total 1.88×。

worker SoA 与线程资源治理已完成;direct-step、pinned slabs/compile 与动态配额只在
后续 profiling 显示成为主瓶颈时再启动。

## 依赖关系

- 阶段 3 依赖阶段 1(基线+golden)与阶段 2(SoA 通路为 rollout buffer 打底)。
- 阶段 4 依赖阶段 3 的 compact action id 通路。
- 阶段 5/6 依赖阶段 2-4;本轮只完成有实测收益或必要资源治理的子项。
- 阶段 7 依赖全部。

## 关键风险

- Rust 编码器与冻结的 Python 基线及脚本内手算语义 oracle 逐元素一致(尤其动作
  类型分派/终局约定);当前生产与兼容接口统一用 golden trace 防回归。
- DDP 分片在改动 collate 后保持 NCCL 对齐与 early-stop 同步。
- 内存:SoA 扁平 buffer 的宿主峰值内存需与旧 list 对比并记录。
