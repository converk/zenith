# Plan: V17 PPO 原生性能重构

## 阶段总览

1. **冻结基线**(已完成):重新测量标准 512g4e 基线;生成 profiling 与 golden trace。
2. **学习者一次性物化**(已实现):`RolloutBuffer` SoA + `PPOLearner.update` 接入。
3. **Rust 融合决策批**(已完成):`riichienv` core 粗粒度紧凑事实物化 +
   `riichi.encode_v16_batch` 融合 offense/defense/shanten/query row;query 热路径
   消除 decode/canonical JSON 与逐动作 Observation 属性扫描;生产桥接只保留
   该路径,没有 Python runtime 回退或配置分支。
4. **direct action-id step + 紧凑 rollout buffer**:action id 由 Rust 直接应用;
   worker 内 pending 改 SoA;driver→learner 走 Ray 大数组/共享内存。
5. **PPO update 深化**:pinned host slabs、non_blocking H2D、DDP static graph /
   fused AdamW / torch.compile 评估。
6. **并发拓扑 sweep**:T1(少 actor 大 vector)/T2(中)/T3(当前)三组对比。
7. **完整测试与三轮基准 + 报告**(已完成,rollout 1.78× vs 129.374s 基线)。

阶段 4–6 保留为后续可选深化;阶段 3 已使 rollout 验收目标达成,本轮不引入
额外 direct-step/共享内存/DDP 风险。

## 依赖关系

- 阶段 3 依赖阶段 1(基线+golden)与阶段 2(SoA 通路为 rollout buffer 打底)。
- 阶段 4 依赖阶段 3 的 compact action id 通路。
- 阶段 5/6 依赖阶段 2-4。
- 阶段 7 依赖全部。

## 关键风险

- Rust 编码器与 Python oracle 的逐元素一致(尤其动作类型分派/终局约定);
  用 golden trace 回归。
- DDP 分片在改动 collate 后保持 NCCL 对齐与 early-stop 同步。
- 内存:SoA 扁平 buffer 的宿主峰值内存需与旧 list 对比并记录。
