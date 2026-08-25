# V17 rollout worker SoA 与 CPU 资源治理设计

> 日期:2026-08-25 · encoding protocol:V16 · training generation:V17

## 1. 问题边界

优化前每个 worker 在小局内生成 `Transition`,完成 GAE 后仍返回
`list[Transition]`。标准 512 约 40 万对象、350 万 ndarray;真实 2048 约 164 万
对象。worker 自报计算完成到 driver 获得结果之间存在 20–90 秒未分段空白。

同机有 24 个物理核/48 逻辑线程,12 worker 各自导入 NumPy/PyTorch/CPU GRP;
默认 BLAS 与 PyTorch 线程池会形成严重超卖。Rust `step_batch` 的线程数是每
worker 共享,不是每桌独占。

## 2. CPU/Ray 资源设计

- driver 创建 `RolloutWorker` 时通过 Ray `runtime_env.env_vars` 注入
  `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`;
  环境在 actor import NumPy/PyTorch 前生效,learner 进程不受影响。
- actor `__init__` 最前执行 `torch.set_num_threads(1)` 与
  `torch.set_num_interop_threads(1)`,再加载 RiichiEnv 与 GRP。
- 每 worker 声明 `num_cpus=2`;12 worker 共预留 24 CPU,仍能在 Ray 检测到的
  48 逻辑 CPU 上并行调度。该声明只治理调度,不做进程 affinity。
- A/B/C/D sweep 后正式保留 `env_step_threads=4`:step=2 在完整 GPU 三轮中
  rollout/update/total 均无收益,不能因单独 step 微基准而修改默认值。

## 3. worker SoA layout

小局进行期间仍保留 `Transition`,因为 reward/GAE 在小局边界回填。`collect()`
返回前一次性压缩,随后不再跨进程传递对象列表。

固定字段按 transition 连续存放:

- lengths:`history/snapshot/query_pair/critic/sequence`
- policy/value:`action/old_logprob/value`
- target:`reward/kyoku_reward/done/advantage`
- `legal_mask[N,241]`

变长字段均为 `flat + offsets[N+1]`:

- history factors/numeric
- snapshot kinds/categorical/numeric
- query rows/action ids
- critic factors

每 shard 共 29 个 ndarray。driver 按 worker id 原顺序 `concatenate`,只拼 flat
数组并重算 offsets;DDP 使用 `select(indices)` 直接产生与旧轮询分片和补齐下标
完全相同的 SoA shard,包括重复 filler index。learner 的 padding/gather、PPO、
SFT KL、DDP 梯度平均均未改变。

## 4. 返回链路 profiling

driver 先 `ray.wait` 等全部 ObjectRef ready,再按原始 worker 顺序 `ray.get`:

1. `worker_ready_s`:worker 执行 + Ray object-store 发布;
2. `object_store_publish_gap_estimate_s`:ready 减最慢 worker `collect_total_s`;
3. `result_get_s`:driver 反序列化/映射;
4. `transition_assembly_s`:driver 合并;
5. worker 额外记录 semantic summary、SoA pack、数组数/字节数、线程、上下文切换、
   games/kyokus/drain 与 collect wall 分布。

旧路径计分轮 `result_get` 21.04/17.69s;SoA 为 0.0021/0.0032s。有效字节量约
1.65–1.69GB/轮,没有删输入;收益来自对象数从约 355.6 万降到 348,而非减少数据。

## 5. oracle 与正确性

专项验收曾用旧返回路径作为临时 oracle;验收完成后已收敛为 SoA 单路径并删除
fallback。当前专项测试固定 seed,直接验证:

- 所有整数、bool、mask 与有效长度逐元素全等;
- float32 字段先显式转换为协议 dtype,`atol=0,rtol=0`,最大误差 0;
- concatenate、任意 select 与 DDP 重复补齐下标全等;
- GAE advantages、empirical returns 与冻结公式一致;
- V16 全 action kind、bridge、golden 与 Rust 编码 oracle 继续通过。

## 6. 后续决策

真实 2048 中 worker max/p50 为 201.82/185.36s,driver wall 206.72s;固定配额尾差
存在,但 games 仅分布在 172–176,且当前首要阶段仍是 inference RPC wait
(122.92s/worker)。简单取消慢 worker 会丢在途小局,本轮不实现动态配额。

Snapshot JSON + critic + action decode 在 2048 合计约 10.63s/worker,只占 rollout
wall 约 5%;worker SoA 已消除最大的 68–91s 返回链路空白。因此暂不继续
Snapshot/Critic/direct-action Rust 融合,避免为次要上界扩大协议状态对齐风险。
