# V18 PPO 训练性能优化实施记录（P0-1 / P0-2 / P1-4 评估回退 / P1-5）

**日期**：2026-08-27（基于 HEAD `e71e3a8` + 本次工作区）
**范围**：`audit/reports/v18/design/V18 PPO训练性能优化审查提示词.md` 中的
P0-1、P0-2、P1-5 已实施并验证；P1-4 实测为负优化后回退（P2 观察项）。
**约束**：未改 token schema / 动作空间 / 模型拓扑 / GRP reward；训练语义与数值语义
不变（新单测逐项数值对照）；评测机制常量未动。

## 1. 已实施项与验证

| # | 主题 | 文件 | 关键改动 | 测试 | 结论 |
| --- | --- | --- | --- | --- | --- |
| P0-1 | RolloutBuffer 因子行 uint8 压缩 + gather 单次分配 | `training/rollout_buffer.py` | 因子行值域 < 256（schema 单源），构造时压 uint8（超域 fail-closed）；`_gather_padded` 单次分配目标 dtype 只写合法区；`collate` 直出 int64（数值逐位一致） | `test_factor_flatten_compacts_to_uint8_and_fail_closed`；既有 collate/select/concatenate 测试 | **PASS** |
| P0-2 | learner collate 预取双缓冲 | `training/learner.py` | 主线程预计算 minibatch 计划（RNG 序列与串行一致）→ 后台线程按序 collate（有界队列 + `stop_event` + `put(timeout)` 防死锁）→ `try/finally` 安全停止并合并计时；`update_collate_prefetch` 默认 `True`；learner 侧去掉无用 `query_rows` H2D | `test_prefetch_collate_matches_serial_update` / `_early_stop_does_not_hang` / `_propagates_collate_exception` | **PASS** |
| P1-5 | 推理 host 数据通路 | `training/inference.py` | `collate_request_rows` 直接出 int64（省 `.astype` 全量拷贝）；pinned 缓冲按名复用 + `non_blocking=True` H2D；数值逐位一致 | 既有 `test_inference_dtype` 与 `test_batched_pipeline` 等全绿 | **PASS** |
| P1-4 | DDP `find_unused_parameters` | `training/learner.py` | dummy 0 系数接入 policy 项 + `find_unused_parameters=False` | 单元测试全绿 | **REVERT**（见 §2） |

## 2. P1-4 回退证据

- 方案：bootstrap 期 `loss = value_coef * value_loss + 0.0 * policy_loss.mean()`
  + DDP 全程 `find_unused_parameters=False`。
- A/B 首轮（512 半庄、mb2048、双卡）：`backward 104.2s → 248.4s`
  （105→246ms/step,2.3×）,`update_wall 251.5s → 401.4s`——负优化。
- 根因：dummy 项使 bootstrap 期 backward 多走整条 policy 反传（原本只反传
  critic 路径），成本远高于 find-unused 遍历；且只有前 2 个 update 需要 True,
  策略期收益（原估 5–10ms/步）不足以抵消。
- 处置：恢复 `find_unused_parameters=True` 与原始 bootstrap loss;README/提示词
  记录为 P2 观察项（动态重建 DDP 的 hook 清理风险中,未实施）。

## 3. A/B 性能数据（512 半庄/update,`target_kl=0.0`,`update_epochs=4`,mb2048,双卡,3 轮）

| 轮次 | 基线 update_wall | 优化后 update_wall | Δ | 基线 algorithm_wall | 优化后 algorithm_wall | Δ | 基线 sps | 优化后 sps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1（预热/bootstrap） | 251.5s | **227.2s** | −9.7% | 353.7s | 335.2s | −5.2% | 1428.8 | 1541.7 |
| 2（bootstrap） | 187.3s | **162.0s** | −13.5% | 267.3s | 238.6s | −10.7% | 1452.3 | 1623.7 |
| 3（首个 policy update） | 374.3s | 351.1s | −6.2% | 448.3s | 426.1s | −5.0% | 874.2 | 921.4 |

- 优化后阶段明细（迭代 1）：forward 70.3s、backward 107.7s（≈基线）、
  collate 18.9s（与 GPU 计算重叠,不再加入 update_wall）、H2D 5.7s。
- **数据通路收益**：`rollout/return_array_bytes` 10.2GB → **4.55GB（−55%）**;
  `transition_assembly_s` 2.93s → 1.30s（−56%）。
- 第 3 轮两方案均受机器外部负载影响（backward ~250ms/step）,相对 Δ 仍为 −6.2%；
  结论以第 1/2 轮（−9.7% / −13.5%）为准。

## 4. 回归结果

- `pytest riichi_ppo_v1/tests`：**201 passed, 0 failed**（unit 168 + integration/protocol；
  含新增 4 项预取/uint8 测试；`test_v18_ppo_config` 同步 config 基线
  `update_epochs=2 / minibatch=2048`——该变化为工作区既有 staged 配置）。
- `cargo test --workspace`：141 passed, 0 failed。
- `pytest RiichiEnv/tests`：284 passed, 2 skipped。

## 5. 交付与回滚

- 提示词：`audit/reports/v18/design/V18 PPO训练性能优化审查提示词.md`（覆盖写为
  聚焦四项的实施提示词,含 P1-4 回退记录）。
- 代码：`training/rollout_buffer.py`、`training/learner.py`、`training/inference.py`、
  `tests/unit/test_rollout_buffer.py`、`tests/unit/test_v18_ppo_config.py`。
- 回滚：整体 revert 本主题提交即可（三项优化各自独立,P1-4 已不包含）。
