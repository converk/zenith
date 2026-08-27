# V18 PPO 训练前性能负优化审查提示词

## 背景

`zenith` 仓库的 V18 架构升级（当前局面快照输入 + Shared 双向 GQA + 结构化 Actor mask + Critic 私有信息）已完成，SFT/GRP 已训练，PPO 训练即将正式开始（150 updates × 2048 半庄/update，双卡 DDP learner，12 rollout workers，每 5 updates 保存 checkpoint 并执行 1v3 评测）。

本次任务：在正式训练启动前，对架构升级引入的代码做一次**性能负优化审查**——找出现在比"合理实现"慢的地方，逐项给证据、改、验证。只做审查与已经在审的修复，不改变训练语义。

## 硬约束（宪法与治理，违反即否决）

- 不修改 V18 token schema、动作空间、模型参数拓扑、GRP reward 定义；不重训 SFT/GRP。
- 任何修改不得改变训练逻辑与数值语义（纯重构必须数值等价，已有测试断言的地方用 `assert_close` 级验证）。
- 评测机制常量（1v3：10 进程 × 400 = 4000 半庄、每 5 updates）与 SFT 节奏常量禁止改。
- 修改以"每主题一个 commit、可独立回滚"为单位；训练开始后禁止再重构。
- 冒烟测试结束必须清理其产物（`checkpoints/riichi_ppo_v1_smoke`）。

## 必读参考（先读再动手）

1. `audit/reports/v18/report/PROGRESS.md` —— 已有 PERF 优化记录（如 `StateTokenEmbedding` 与 `_assert_structure` 整批向量化 fwd 74→19ms、策略头整批向量化消除逐行 `torch.nonzero` 同步、`encode_kyoku` 改用 Rust 直接动作映射、按 kind 向量化域统计、`validate_semantics=false` 等）。**已优化的点视为基线，不要重复优化，但要检查同类问题是否在其他路径存在。**
2. `audit/reports/v18/report/V18输入链路复审与性能审查报告.md` 与 `audit/reports/v18/scripts/v18_perf_review.py` —— 上游性能审查方法。
3. `audit/reports/v18/design/V18 ppo训练.md` —— PPO 训练语义（λ-return、分支裁剪、normalized entropy、bucketing、KL guardrail 等，改动不得影响这些语义）。
4. `riichi_ppo_v1/training/profiling.py`（StageProfiler/GpuSampler）与运行输出 `logs/` 中的 `performance.jsonl`（含各阶段耗时统计）—— 用数据定位热路径。
5. 运行约定：Conda 环境 `Mahjong-AI`；性能/训练测试默认 `CUDA_DEVICE=0,1`、`learner_gpus=2`；测试基线 `target_kl=0.0`、`update_epochs=4`。

## 审查范围

- Python：`riichi_ppo_v1/model/`（architecture、dense_embedding、current_state、native_encoding、bridge、validation、semantic_validation）、`riichi_ppo_v1/training/`（worker、inference、learner、rollout_buffer、trajectory、train、metrics、tensorboard）、`riichi_ppo_v1/evaluation/`。
- Rust：`RiichiEnv/riichienv-core/`（observation、mjai_select、状态机）与 PyO3 绑定层。
- 交互：PyO3 边界、Ray actor RPC、numpy↔torch↔Rust 数据路径、DDP collective。
- 明确排除：V16/V17 冷存储代码、SFT 数据预处理（已跑完）、1v3 评测机制本体。

## 八个审查维度

### 1. Python 侧是否在做大量解析与计算

- rollout 每决策 hot path：是否每步逐行/逐对象 Python 循环、重复 JSON 解析、重复编码；可缓存的计算是否每次重算。
- `worker.py`：`PublicStateTracker` 更新、`bridge.last_events` 消费、semantic 业务统计（`record_kyoku` 的逐事件 `json.loads` 与行列扫描）是否每小局全量跑；能否缓存/搁置或向量化。
- `learner.update` 每 minibatch 路径：collate 是否纯 numpy gather；有无 `.item()`/`.tolist()`/`torch.nonzero` 等 GPU→CPU 同步点（历史已消除一处，检查是否残留）；指标聚合（多个 `metric_sample_sums.add_` 张量）能否合并。
- 结构校验（`_assert_structure` 等）是否每次 forward 全跑；能否按配置抽样/关闭（`validate_semantics=false` 是已有先例）。
- 检查 `trajectory.py`/`learner.py` 的 host 侧循环：如 `discounted_empirical_returns` 的逐 transition Python 循环（O(N)）——能否向量化（注意 kyoku 边界 reset）或维持现状并说明理由。

### 2. 可并行的地方是否写成了串行

- 环境侧：`env_step_threads` 是否真正并行；状态机 `MjaiKyokuStateMachineManager` 的逐 env 事件生成是否串行且可批量。
- 推理侧：`RolloutInferenceActor` 跨 worker 批处理是否成为单卡串行瓶颈（队列等待、`inference_max_batch_size`/`inference_batch_wait_ms` 参数是否合理）；采样 `torch.multinomial` 是整批还是逐样本。
- 训练侧：DDP 两 rank 分片负载是否均衡（分片补齐造成多少重复样本/padding）；attention/mask 是否已整批。
- 编排侧：rollout 与 update 严格串行（train.py：collect → update → 权重推送 → 下一轮 collect）。评估 rollout/update 流水线 overlap 的可行性与复杂度（注意 on-policy 语义与模型版本一致性；可作为 P2 观察项，一般不实现）。
- 多卡空闲：rollout 阶段 learner 卡空转、update 阶段推理卡空转——量级多大，是否有低成本改进（如异步权重推送）。
- Ray：`num_workers × envs_per_worker` 吞吐是否与 inference actor 匹配；对象存储传输 `RolloutBuffer` 的序列化/反序列化耗时占比（`performance.jsonl` 的 `rollout/object_store_publish_gap_estimate_s` 等字段）。

### 3. 可放入 Rust 侧却放到了 Python 侧

- 编码链路：`current_state` 快照编码的哪些字段仍在 Python 逐行拼装（对照 `native_encoding.NativeQueryBatch` 已有的批量原生路径）。
- 事件统计与业务指标：`metrics.record_kyoku` 的 JSON 解析与事件扫描在 Python；评估 Rust 侧直接输出结构化小局摘要的可行性与口径保持（测试 `test_metrics.py` 已锁定口径，改动必须全绿）。
- 动作 id 映射、合法动作确定、`HandEvaluator.is_tenpai` 所在侧。
- 原则：**以 profile 数据为准，只迁移 hot path**；不许为了"看起来该进 Rust"而迁冷路径。

### 4. 环境 / 状态机 / PPO 训练代码专项

- 环境：`step_batch` 是否走批量原生路径（`prepare_current_state_batch`）；每 step 的同步点、观察物化成本。
- 状态机：`last_events` 每 kyoku 清空还是每 step 分配；事件缓冲复用；`bridge.sync` 频率与成本。
- rollout：`Transition` 物化 → `RolloutBuffer` SoA 的内存峰值与拷贝次数；`rollout_buffer.concatenate`/`select`/`collate` 的拷贝量。
- update：advantage 归一化、λ-return、MC return、KL 累计、padding fraction 统计的 host 开销；`executed_padded_input_tokens` 等诊断计算是否每步做。
- checkpoint：每 5 updates 的 `save`（全量权重 CPU 拷贝 + torch.save）耗时与磁盘占用；`latest.pt` 每次保存是否必要。

### 5. Rust 侧影响并发性能的实现

- `riichienv-core`：批量 step 是否真正多线程（step_threads 语义）；有无全局锁/`RefCell`/`Mutex` 把并行串行化；RNG 每 env 独立还是共享竞争。
- 状态机：每 env 实例独立；事件缓冲区有无每 step 大分配。
- PyO3 绑定：长任务是否持 GIL；每次跨边界调用的固定开销 vs 批量 API 设计是否合理。
- 有无不必要的克隆/拷贝（Vec/String/JSON 序列化往返）。
- `mjai_select.rs` 等解码路径的查找表与缓存。

### 6. Rust↔Python 交互优化点

- 每决策/每 step 的 PyO3 调用次数与往返次数；能否合并为更少、更大的批量调用（沿 `NativeQueryBatch` 的思路）。
- numpy 传递是否零拷贝（stride/dtype 对齐、`np.asarray` vs 复制）；torch↔numpy 往返次数（worker 环境侧 `envs.step_batch` → torch 每步几次转换）。
- `last_events` 的 JSON 字符串在 metrics/状态跟踪中被反复 `json.loads` 的次数；能否一次解析复用。

### 7. 其他可优化点（提高并行度等）

- GPU/CPU 利用率时间线：对照 `performance.jsonl` 的 `rollout_wall_s`/`update_wall_s`/`algorithm_wall_s`，找出占比最大的阶段。
- patchwork：推理 actor 能否在 update 期间预加载下一轮权重；worker 侧预热/预分配。
- 内存：critic 序列拼接的全 0 初始化后 scatter 的分配；bucketing 窗口（multiplier=8）之后的实际 padding 比例是否还有优化空间（对照 `update/executed_padding_fraction_of_padded_input_tokens`）。
- 日志与指标：`performance.jsonl`/`metrics.jsonl` 的 append 是否阻塞训练循环；TensorBoard 写入频率。

### 8. 每项修改的验证协议（强制）

对每一个拟引入的修改，按顺序执行：

1. **基线测量**：修改前先量化现状（profiler/计时脚本/`performance.jsonl` 字段），记录数字。
2. **修改实现**：保持语义等价；纯重构不得改变任何对外数值。
3. **复测对比**：修改后复测同一场景。只有收益真实（hot path 明显改善，如 ≥10% 或绝对量级可量化）才保留；无收益或负收益立即回滚该提交。
4. **回归测试**：
   - `conda run -n Mahjong-AI python -m pytest riichi_ppo_v1/tests` 全量（当前基线 197 passed，不得减少）；
   - 重点语义测试：encoding roundtrip、GAE/λ-return 目标、normalized entropy、metrics 统计口径（双响/流局听牌/荒牌流局）、DDP 分片聚合、bucketing 可复现、resume 后 Adam/schedule 连续；
   - 环境与状态机改动跑对应 protocol/integration 测试（`test_v18_encoding_bridge.py`、`test_bridge_*` 等）。
5. **可回滚**：每个主题独立 commit；GL 评审时可整体 revert。
6. 所有测量与结果写进报告；未达标项标注 REVERT。

## 输出要求

将审查报告写入 `audit/reports/v18/report/`（命名自描述，如 `V18_PPO训练前性能审查报告.md`），内容包括：

- 审查范围与方法（数据来源、基线数字）。
- 发现清单按严重度分级：
  - **P0 必须修**：显著影响吞吐/延迟且修复成本低；
  - **P1 建议修**：中等收益；
  - **P2 观察项**：低收益或高风险候选，只记录不动手。
- 每项发现给出：位置（文件:行）→ 证据（测量数字）→ 影响（量级）→ 修改方案 → 验证结果（基线与修改后对比 + 测试结论 + PASS/REVERT）。
- 已修复项汇总表（含每个修改的收益与测试结果）。
- 剩余风险与未优化项。
- 最终结论：**是否可以开始正式 PPO 训练**（给出量化依据）。

## 工作方式

- 先读参考文档与现有 profile 数据形成热路径清单，再针对性测量，避免无数据优化。
- 修改与验证以小步为单位；测量方法保持一致（同场景、同 seed、同设备）。
- 如发现必须改动训练核心语义才能优化的点，只记录到 P2 观察项，不擅自实现。
- 报告完成后同步更新 `audit/reports/v18/report/PROGRESS.md` 的性能记录。