# V18 PPO 训练开关说明

本文档记录 `riichi_ppo_v1/configs/v18_ppo*.yaml` 中训练路径专属开关的语义、
依据与默认值。评测机制常量不在本文范围(见 `evaluation/mechanism.py`)。

## `update_validate_structure`(默认 `true`)

- **作用范围**:learner 的每次模型前向(主 forward 与 SFT reference
  policy-only forward)以及 rollout 推理 actor 的 `_run_full_forward`。
- **语义**:`true` 时模型 forward 内执行全部 GPU 侧结构校验(actor/critic
  `_assert_structure`、长度范围、pair_counts 范围、action query tail 窗口
  canonical 契约、action id 行内唯一性);`false` 时全部跳过。
- **依据**:V18 输入由 Rust 编码器 fail-closed 生成,另有 SFT 契约校验与
  单测覆盖(`test_v18_architecture.py` 的 tail 窗口契约断言等);训练期逐批
  重复校验会引入十余次 GPU→CPU 同步(实测一次 forward 45 个同步点中约
  17 个来自校验),且这些校验同时是 torch.compile fullgraph 的断裂源。
- **数值影响**:无。`false` 仅移除校验分支,capacity/host 标量与算术索引
  的取值语义与 `true` 逐位一致(生产 bf16 autocast 路径经 `torch.equal`
  对照验证;fp32 路径仅存在 ~1e-7 的 GEMM 形状舍入差)。
- **评测/离线路径**:1v3 评测、SFT、其他直接调用 `model(...)` 的路径未传
  该键,保持默认 `true`,行为不变。

## `update_collate_prefetch`(默认 `true`)

learner collate 双缓冲预取线程;`RolloutBuffer.collate` 的
`include_query_rows`/host 容量标量(`shared_capacity`/`critic_total_capacity`)
由 learner 路径消费,worker/inference/测试调用方不受影响。
