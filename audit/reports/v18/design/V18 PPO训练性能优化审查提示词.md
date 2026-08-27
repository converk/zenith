# V18 PPO 训练性能优化实施提示词（P0-1 / P0-2 / P1-4 / P1-5）

## 1. 背景与目标

`zenith` 仓库的 V18 PPO 训练链路（当前局面快照输入 + Shared 双向 GQA + 结构化 Actor mask +
Critic 私有信息）已端到端可运行。按 512 半庄/update、minibatch 2048、双卡 DDP 实测：
`update_wall ≈ 187–251s`（其中 model_forward ≈ 48–67s、backward ≈ 78–104s、collate ≈ 15–20s、
H2D ≈ 4–6s、DDP 分片 select+队列 IPC ≈ 26s）,`rollout_wall ≈ 72–99s`（其中
worker 侧 `inference/rpc_wait ≈ 67.7s`、`model_state_prepare ≈ 13.2s`）。

本提示词聚焦四项**经实测方向明确、可独立回滚**的优化，全部保持训练语义与数值语义不变：

- **P0-1**：RolloutBuffer 因子行 dtype 压缩（int32/uint8 → uint8）——把全量 SoA 缓冲
  （实测 ~8–10GB）缩小 ~4×,直接降低 Ray 对象存储、DDP `select` 拷贝与 mp 队列 IPC。
- **P0-2**：learner collate 预取双缓冲线程——实测把每 minibatch ~20ms 的 CPU collate
  与 GPU forward/backward 重叠（上版实验 update_wall 251.5→233.5s,−7.2%）。
- **P1-4**：DDP `find_unused_parameters` 关闭——bootstrap 期以 0 系数 dummy 项把
  policy 路径接入损失图,使全部参数恒收到（零）梯度,从而全程可用
  `find_unused_parameters=False`,消除 PyTorch 官方警告的每步额外 autograd 遍历。
- **P1-5**：推理 actor host 数据通路优化——collate 直接出 int64（省 `astype` 全量拷贝）
  + pinned 缓冲复用与 `non_blocking=True` H2D。

## 2. 硬约束（违反即否决）

- 不修改 V18 token schema、动作空间（241）、模型参数拓扑、GRP reward 定义；
  不重训 SFT/GRP。
- 任何修改不得改变训练逻辑与数值语义：纯重构必须数值等价（`assert_close` 级验证）；
  改变训练语义的参数（minibatch_size、update_epochs、γ/λ、clip、entropy、KL 等）
  只能作为候选建议上报,不得擅自改动正式配置。
- 评测机制常量（1v3:10 进程 × 400 = 4000 半庄、每 5 updates）与 SFT 节奏常量禁止改。
- 性能/训练测试基线固定：`CUDA_DEVICE=0,1`、`learner_gpus=2`、`target_kl=0.0`、
  `update_epochs=4`、`games_per_update=512`、跑 3 轮、第一轮视为预热（AGENTS.md 约定）；
  A/B 需同场景同 seed 同设备。该测试基线与长期训练默认值相互独立。
- 冒烟/测量结束必须清理其产物（`/tmp` 下测量目录与日志;正式
  `checkpoints/train_riichi_v18/ppo` 禁止残留测量数据）。
- 代码注释一律中文；领域常量必须收敛单一来源。
- 所有修改完成并验证后再 commit（用户指定「全部改完再 commit」）。

## 3. 四项优化实现要求

### 3.1 P0-1 RolloutBuffer 因子行 uint8 压缩（`riichi_ppo_v1/training/rollout_buffer.py`）

- 依据：V18 token 因子行字段（segment/kind ≤ 111、action_id ≤ 240、tile_type ≤ 34、
  count 等）均 < 256,实测 fixture 与真实路径 max ≤ 240。
- 实现：
  - 新增 `_compact_factor_flat(flat, name)`：`flat.size==0` 时直接转 uint8;否则
    `if int(flat.max()) > 255: raise ValueError`（fail-closed,防静默回绕）后
    `astype(np.uint8, copy=False)`。
  - `__init__` 中 `actor_factors_flat`、`query_rows_flat`、`query_ids_flat` 经该函数
    压缩;`critic_factors_flat` 本就是 uint8,不动。
  - `_gather_padded` 增加 `dtype` 可选参数：单次分配目标 dtype 输出并只写合法区
    （替代旧「advanced-index 拷贝 + `np.where` 再拷贝 + `.long()` 再拷贝」）,输出值
    与旧实现逐位一致。
  - `collate` 中 `actor_factors`/`query_action_ids`/`critic_factors` 以 `dtype=np.int64`
    gather（模型 embedding 需要 long）;`query_rows` 保持 `dtype=np.int32`（仅离线
    一致性校验用）;空 critic 分支也返回 int64。
- 语义：`collate` 外部 dtype 与旧版一致（int64）,数值逐位一致。
- 测试：`test_rollout_buffer.py::test_factor_flatten_compacts_to_uint8_and_fail_closed`
  （dtype 断言 + 超 255 抛错）。

### 3.2 P0-2 collate 预取双缓冲（`riichi_ppo_v1/training/learner.py`）

- 依据：实测 collate 20.9ms/step × ~1000 = ~21s/update,与 GPU 计算无依赖;
  上版实现（已被回退）因「producer 阻塞 put + consumer 早停不排空 + profiler 非线程安全 +
  RNG 所有权」挂起,本次必须按下列设计修复后再实施。
- 实现：
  - 新增 `_PrefetchAborted`、`_PrefetchState`（有界队列 `Queue(maxsize=2)`、`stop_event`、
    `finished_event`、`errors`、逐批 `times`、`thread`）与 `_prefetch_collate_worker`。
  - producer 只做纯 CPU `transitions.collate`（不触碰 CUDA）;`queue.put(timeout=0.2)`
    循环内反复检查 `stop_event`（防死锁核心）;异常记入 `errors`,finally 置
    `finished_event`。
  - 主线程预计算全部 epoch 的 minibatch 计划（与串行路径同一 `rng` 序列→分桶可复现）,
    再启动 producer;consumer 用 `_prefetch_get`（`get(timeout=0.2)` + `finished_event`
    判定,早停抛 `_PrefetchAborted`）。
  - `update()` 的 epoch 循环包 `try/finally`：finally 调 `_stop_collate_prefetch`
    （set stop_event → 排空队列 → join(5s) → 把 `times` 逐批并入
    `self.profiler.add("update/collate_soa_gather", t)`）。
  - 配置 `update_collate_prefetch`（默认 `True`）;`update()` 内 `host_batch.pop("query_rows", None)`
    后再 `transfer_batch_to_device`（模型 forward 不消费 query_rows,免无用 H2D）。
- 语义：minibatch 顺序、RNG、损失/梯度/指标与串行路径完全一致（新单测逐项对照）。
- 测试：`test_prefetch_collate_matches_serial_update`、`test_prefetch_early_stop_does_not_hang`、
  `test_prefetch_propagates_collate_exception`。

### 3.3 P1-4 DDP `find_unused_parameters` 关闭（**已评估并回退**,见 §3.3.1）

- 意图：消除 PyTorch 官方警告的每步额外 autograd 遍历
  （`find_unused_parameters=True ... extra traversal ...`）。
- 实测结论（512 半庄、mb2048、双卡 DDP,3 轮 A/B 的首轮）：
  - 方案 A（dummy 0 系数接入 policy 项 + `find_unused_parameters=False` 全程）:
    bootstrap 期 backward **104.2s → 248.4s（105→246ms/step,2.3×）**,update_wall
    **251.5s → 401.4s** ——负优化,已回退。根因：dummy 项使 bootstrap 期 backward
    多走整条 policy 反传（原本只反传 critic 路径）,成本远高于 find-unused 遍历。
- **不实施**：方案 B（bootstrap 后重建 `DistributedDataParallel(..., find_unused_parameters=False)`）
  需处理旧 reducer hook 双注册/清理与双 rank 同步,风险中;能否抵消 bootstrap
  期方案 A 的代价以及策略期收益（prompt 原估 backward -5~10ms/步）均未证实,
  列为 **P2 观察项**（仅记录,不动手）。
- 代码保持 `find_unused_parameters=True`（原样）;bootstrap loss 维持
  `value_coef * value_loss_values_.mean()`。

### 3.3.1 回退记录（P1-4）

| 阶段 | 证据 | 结论 |
| --- | --- | --- |
| 实现 | `loss = value_coef * vl.mean() + 0.0 * policy_loss.mean()` + `find_unused_parameters=False` | 单元测试全绿 |
| A/B 首轮 | backward 104.2→248.4s;update_wall 251.5→401.4s;forward 66.8→68.5s(不变) | **REVERT** |
| 保留 | P0-1/P0-2/P1-5 与该项解耦,继续有效 | — |

### 3.4 P1-5 推理 host 数据通路（`riichi_ppo_v1/training/inference.py`）

- 依据：`_run_full_forward` 每 dispatch 先 `collate_request_rows`（int32/uint8 拼装）,
  再 `.astype(np.int64)` 整块拷贝,再同步 `torch.as_tensor(device)` H2D;这些 host 拷贝
  与 GPU forward 串行。
- 实现：
  - `collate_request_rows` 的 `actor_factors`/`query_action_ids`/`critic_factors` 直接以
    int64 分配并写入（赋值时隐式转型）,删除 `_run_full_forward` 的 `.astype(np.int64)`。
  - 新增 `_pinned_capacity_shape` 与 `_host_tensor_to_device`：按名缓存 pinned 缓冲
    （容量 = `inference_max_batch_size × context_tokens`）,`view.copy_(host)` 后在
    `non_blocking=True` 下 H2D;数值逐位一致。
  - `_pinned_pool` 在 actor `__init__` 初始化;非 CUDA 设备走 `torch.from_numpy` 原路径。
- 边界：`profile_cuda_sync` 语义不变;确定性（multinomial RNG、批序）不变。
- 测试：既有 `test_inference_dtype.py`（dtype 契约）与 `test_batched_pipeline.py` 必须全绿;
  不新增 dtype 断言测试（collate 输出已是 int64,旧测试无此断言）。

## 4. 方法论（先证据后动手）

1. **基线**：以 §2 固定基线（512 半庄、3 轮）先量化现状,记录
   `performance.jsonl` 的 `rollout_wall_s`/`update_wall_s`/`update_forward_s` 与
   `ppo/timing/update/*` 阶段明细。
2. **实现**：按 §3 四项依次落地,每项保持语义等价;注释中文;不引入新依赖。
3. **回归**：
   - `conda run -n Mahjong-AI python -m pytest -q riichi_ppo_v1/tests`（当前基线 169 unit +
     integration/protocol,不得减少）;
   - 语义重点：encoding roundtrip、GAE/λ-return、normalized entropy、metrics 统计口径、
     DDP 分片聚合、bucketing 可复现、resume 后 Adam/schedule 连续;
   - Rust 改动：`cargo test --manifest-path RiichiEnv/Cargo.toml --workspace` 与
     `pytest RiichiEnv/tests`（本次无 Rust 改动,复跑确认）;
   - 环境/状态机改动跑 `test_v18_encoding_bridge.py`、`test_bridge_*`、`test_batched_pipeline.py`。
4. **A/B**：同场景同 seed 同设备 3 轮（第一轮预热,报告后两轮统计）;只有热路径明显改善
   （如 ≥5% 或可量化）才保留;无收益或负收益立即回滚该提交并记录。
5. **可回滚**：全部改完并验证后再统一 commit（用户指定）。每个主题在 commit 内保持
   逻辑独立,便于整体 revert。

## 5. 已验证清单（禁止重复实施,除非有新形态/新证据）

| 方案 | 结论 |
| --- | --- |
| 分支梯度裁剪 `foreach=True` + post 范数代数导出 | 无净收益,已 revert |
| `discounted_empirical_returns` 向量化 | 微基准 173→12ms,端到端 <0.1%,已 revert |
| `loss_detail` 惰性化 | 无变化,已 revert |
| `bridge.decode` Python 侧本地 action_id 映射 | decode 缩短,端到端无变化,已 revert |
| 小局结束事件单次解析复用 | ~0.15s/worker,端到端无变化,已 revert |
| 推理批内按长度排序 | padding 未降,已 revert |
| `inference_batch_wait_ms` 8→16 + 排序 | rows/forward 178→287,rollout_wall 持平,已 revert |
| minibatch collate 预取线程（第一版） | 因线程挂起回退;**本提示词 §3.2 以修复设计重新实施** |

## 6. 输出要求

- 实现完成后,在 `audit/reports/v18/report/` 追加性能记录（或更新既有报告段）:
  每项的位置（文件:行）→ 证据（测量数字）→ 影响（量级）→ 方案 → 验证结果
  （基线 vs 修改后对比 + 测试结论 + PASS/REVERT）。
- 同步更新 `audit/reports/v18/report/PROGRESS.md` 的性能记录。
- 最终结论：量化四项的累计收益,明确是否可以开始正式 PPO 训练。
