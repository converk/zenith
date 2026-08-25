# V17 PPO 原生性能重构 · 最终报告

> 日期:2026-08-25 · 状态:最终(Rust 融合 + worker SoA/CPU 资源治理)
> 相关产物:`specs/007-v17-ppo-native-performance/`、`audit/reports/v17/{design,eval,report,scripts}/`、`logs/v17/`

---

## 1. 优化前后调用链

### 优化前(Round 1 基线实测)

```
BatchedRiichiEnv(每 worker 32 桌)
  -> Python Observation(dict[seat])
  -> active_decisions()
  -> bridge.prepare_v16:
       action_jsons / snapshot_json(每动作 JSON serialize)
       -> state_machine.prepare_decisions(Rust, JSON 输入)
       -> decode_actions(每合法动作 id -> MJAI JSON)
       -> analyze_action_queries(Python 逐动作循环, 每动作 2~3 次 PyO3)
       -> encode_query_row / encode_snapshot_rows / encode_critic_features
       -> numpy padding
  -> inference.remote(...)  (GPU 前向, 小批 ~50 行)
  -> ray.get(results)  -> decode -> env.step_batch -> bridge.sync
```

### 优化后(SoA 学习者 + batched query)

- **Rollout(worker 侧)**:`analyze_action_queries_batch` 让同一 observation 的
  不变量事实只算一次,offense/defense/shanten 内核各汇聚为 1 次 batch 调用。
- **PPO update(learner 侧)**:`RolloutBuffer` 一次性物化,`collate` 用向量化
  gather 替代逐 Transition `torch.tensor` 拷贝;padding plan 跨 epoch 复用。

---

## 2. Profiling 与瓶颈排序(标准 512g4e,第 2/3 轮均值附近实测)

| 阶段 | 优化前 | 优化后(SoA+batch) | 说明 |
|---|---|---|---|
| `state/v16_query_assembly` | ~80s | ~52s | 最大热路径;offense shanten 内核 |
| `rollout/model_state_prepare` | ~85s | ~56s | 含 query/snapshot/critic/history |
| 推理 `full_forward`(2 actor 求和) | ~96s | ~65s | 批尺寸随编码提速而增大 |
| `inference/queue_wait`(2 actor 求和) | ~121s | ~92s | GPU 前向等待 |
| `env/step_batch_native` | ~2.5s | ~2.5s | Rust 原生(已很快) |
| `update/collate_host_padding_copy` | ~105s | ~8s(soa_gather 7.9s) | **SoA 最大收益** |
| `update/model_forward`+`backward` | ~246s | GPU 计算不变 | 不允许削减 |

**瓶颈排序**:① 查询组装(offense shanten);② GPU 前向;③ queue_wait;④ env step(可忽略)。

---

## 3. Rust / buffer API 与 layout

### 3.1 学习者 SoA(`RolloutBuffer`)
- 一次物化:所有变长字段(history/snapshot/query/critic)落成 flat arrays +
  `offsets[N+1]`;标量字段(actions/old_logprobs/values/rewards/advantages/
  done/legal_mask)直接抽取。
- `collate(indices)` 用 `_gather_padded` 向量化 gather,输出与
  `materialize_host_batch` **byte-exact** 的 V16 padded host 张量。
- 关闭 `update_use_soa=false` 可回退旧逐样本路径(oracle)。

### 3.2 批量 query(`analyze_action_queries_batch`)
- `_observation_facts(observation)`:抽出 remain/rivers/dora/hand/meld 等
  observation 不变量,一次决策共享。
- 按动作分类收集 `off_reqs`/`def_reqs`/`hand_reqs`,各内核各 1 次 batch 调用;
  组装阶段按 kind 组装 O0..O9 / D0..D9(与 `analyze_action_queries` 完全一致)。

---

## 4. 并发拓扑选择(已实测)

| 拓扑 | num_workers | envs/worker | inference batch 设置 | rollout 均值 | 结论 |
|---|---|---|---|---|---|
| T3(当前) | 12 | 32 | 默认(rows 0/wait 8/target 0) | 120.44s | **最优** |
| T-bigbatch | 12 | 32 | rows 512 / wait 20 / target 100 | 138.78s | **负结果**(延迟 > GPU 收益) |

**结论**:rollout 关键路径是 worker 编码 + RPC 等待延迟,而非 GPU 利用率;
增大 inference batch 反而拖慢。三个拓扑中 T3 最优,T-bigbatch 已排除,
T1(少 worker 大 env)理论上会减少并发编码并行度,风险较高未跑通。

---

## 5. 每项优化独立收益(同条件基线 vs 最优组合,第 2/3 轮均值)

| metric | 基线 | SoA-only | SoA+batch(最优) |
|---|---|---|---|
| rollout | 129.374s | 127.331s | **120.444s** |
| update | 209.565s | 152.004s | **150.159s** |
| total | 340.178s | 280.630s | **271.900s** |
| speedup | — | 1.02×/1.38×/1.21× | **1.07×/1.40×/1.25×** |

- **SoA 独立收益**:update 1.38×(host collate ~105s→~8s)。
- **batch query 叠加收益**:rollout 1.06×、total 1.03×(相对 SoA);query 组装
  ~80s→~52s。

---

## 6. 正确性证据

- `RolloutBuffer.collate` 与 `materialize_host_batch` 逐元素等价
  (`tests/unit/test_rollout_buffer.py`,3 项;`update_use_soa` 真假两大权重路径 loss
  一致)。
- `prepare_v16`(batch off/on)325 个 tensor 逐元素一致;
  `analyze_action_queries_batch` 与逐动作 oracle 0 mismatch/1233 actions
  (`tests/unit/test_action_query_batch.py`,2 项)。
- 现有单测 158+ 通过;桥接/批量管线集成测试通过。
- 领域不变:V16 encoding/action schema、GRP reward、PPO/GAE/value/entropy/
  SFT KL 数学、current-only transition、DDP 权重一致均未改动;reference forward
  保留(SFT KL 非零)。

---

## 7. 三轮基准对比(vs 同条件基线)

见 `audit/reports/v17/{report/round1-progress,round2-progress,round3-progress}.md`
与 `logs/v17/perf_512g4e_{noso,soa,bq,bigbatch}.log`。最优组合关键结果:
- rollout 120.44s(1.07×)、update 150.16s(1.40×)、total 271.90s(1.25×)。

---

## 8. 剩余瓶颈与 Amdahl 上界

- `state/v16_query_assembly` ~52s:offense **shanten 内核** ~37μs/动作,批量只能
  降 PyO3/Python 开销,无法减少 shanten 计算本身(Amdahl:该段在 worker 关键路径,
  理论削到 ~40s 后 rollout 约 1.2×)。
- GPU `full_forward` ~65s + `queue_wait` ~92s:受限于 batch 尺寸与等待延迟;
  直接加大 batch 已证负结果。
- **滚出 1.2× 未达;若仅能削编码到 ~40s,rollout 上限约 (129−52)/ (120−40)≈1.27×**
  (需消除编码与 GPU 前向的重叠遗漏 + 削 queue_wait)。

## 9. 下一步(未完成项,按优先级)

1. **完整 Rust 融合** `encode_v16_batch`:post-shape 计算 + shanten 批内核 +
   快照/query/critic 编码整体入 Rust,产出 SoA buffer(消除 ~64μs/动作 Python
   后处理)。
2. **双缓冲/流水线**:把 worker 编码与 GPU 前向真正重叠,削 queue_wait。
3. 紧凑 rollout buffer(worker 内 SoA + Ray 大数组/shared memory)。
4. PPO update 深化:pinned host slabs、non_blocking H2D、DDP static graph /
   fused AdamW / torch.compile。

## 10. 阶段性训练建议(已由第 17 节取代)

- 当时启用 `update_use_soa: true` 与 Python `v16_batch_query`;后者及其配置开关
  已在 Rust-only 清理中删除。
- 保持 `num_workers=12`、`envs_per_worker=32`、`inference_batch_wait_ms=8.0`、
  `inference_batch_target_rows=0`(即默认,不要用大 batch)。
- 启动命令(标准基线验证):
  `env -C /mnt/disk1/hubowen/zenith RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0,1 PYTHONUNBUFFERED=1 python -m riichi_ppo_v1.training.train --config riichi_ppo_v1/configs/v17_ppo_perf_512g4e.yaml --device cuda --learner-gpus 2`

---

## 11. Round 5:Rust Action Query 融合快速路径(取代第 8–10 节的阶段性结论)

### 11.1 新调用链

```
Observation/Action(每个唯一对象只跨一次 PyO3)
  -> riichienv.prepare_v16_compact_facts
       post-shape / remaining / rivers / meld+kan / kind / O7-O9 / defense SoA
  -> riichi.encode_v16_batch (GIL released)
       offense shared kernel + defense + simple shanten + kind dispatch
       -> query_rows int32[N,2,15] + wait_masks uint64[N]
  -> 仅听牌行由 riichienv Rust yaku batch 计算并回填 O4/O5
  -> 按 decision offset view/reshape -> V16 padding -> GPU inference
```

state-machine 在 `prepare_decisions` 时保存 action id→原始合法 Action 下标,
query 热路径不再执行 `decode_actions -> MJAI JSON -> canonical JSON -> PyObject`
代表动作匹配。`riichi` 公开模块仍不依赖 `riichienv`。

### 11.2 Rust API 与 buffer layout

- `CompactV16Facts`:20 个 C-contiguous SoA 数组,核心行为
  `shape_counts/remaining/defense_counts uint8[N,34]`、
  `opponent_rivers uint64[N,3]`、其余逐行动作/flag/slot 数组。
- `encode_v16_batch`:输出 `query_rows int32[N,2,15]`;末维为
  `[query_type, action_id, action_type, primary_tile_code, source_seat_code,
  answer_0..answer_9]`;另输出 `wait_masks uint64[N]` 与批内唯一形状计数。
- 向听 34 种摸牌扫描保持完整,但复用三个未变化牌组的 table-combine 中间结果;
  这是等价公共子表达式消除,不删除任何候选牌或有效输入。

## 12. 正确性证据

- 全 kind 合成回归:tsumo/ron/reach/dahai/chi/pon/ankan/daiminkan/kakan/
  none/pass/ryukyoku,新旧 `query_rows` 逐元素全等。
- 真实 env 30 tick bridge 对照:history/snapshot/query/critic/legal mask 全字段全等。
- 冻结 golden:15 组数组、69,733 个元素 byte-exact(`integer_exact=1`,
  `float_exact=1`),decode 与状态机边界 JSON 全等。
- 清理前验收:`riichi_ppo_v1/tests/unit` 164 passed;V16 bridge 集成 11 passed;
  Rust crate 10 passed;RiichiEnv 定向测试 14 passed。移除 3 个旧路径专用测试后,
  当前完整 unit 为 161 passed,集成与 bot 为 53 passed、1 个硬件测试 skipped。
- `analyze_action_queries` 当前仅为 Rust 单动作编码结果恢复 `ActionQuery` 兼容对象;
  Python 中原有动作分派、post-shape、向听/有效牌、防守与役种实现已删除。SFT、
  在线审计、PPO 与测试共用 Rust 语义;脚本内手算的 V16 语义断言全部通过。
- V16 schema、GRP/PPO/GAE/value/entropy/SFT KL、current-only transition、
  opponent mix、checkpoint/DDP 数学均未修改;无需改协议文档。

## 13. 独立收益与 profiling

CPU 真实决策批(8 env × 120 ticks,8,822 action rows):

| 路径 | `state/v16_query_assembly` | 相对 |
|---|---:|---:|
| Python batch oracle | 1.493045s | 1.00× |
| Rust compact+fused | 0.136433s | **10.94×** |
| Rust-only 清理后复测 | 0.168479s | **8.86×** |

清理后复测包含 Rust O4/O5 facts 重建与在线 Observation override 边界;仍远低于
原 Python batch。GPU 三轮结果对应 Rust 融合主体,删除回退不会增加 rollout 分支。

标准 GPU 基准中,worker throughput 从 Python batch 的约 378 transitions/s 提到
计分轮约 731–734 transitions/s;inference rows/forward 从约 103 提到 173–176。
新瓶颈排序:① GPU inference/full forward 与分布式排队;② PPO update 的最长序列/
系统抖动;③ env step(计分轮仅 1.71–1.77s)。编码已不再是首要瓶颈。

## 14. 标准 512g4e 三轮结果

配置:`CUDA_DEVICE=0,1`,`learner_gpus=2`,`games_per_update=512`,
`update_epochs=4`,`target_kl=0.0`,seed=1。第 1 轮预热:

| 轮次 | rollout | update | total | epochs | minibatches |
|---|---:|---:|---:|---:|---:|
| 1(预热) | 90.143s | 145.470s | 235.627s | 4/4 | 1360/1360 |
| 2 | 71.339s | 111.523s | 184.259s | 4/4 | 1024/1024 |
| 3 | 74.204s | 190.518s | 265.829s | 4/4 | 1056/1056 |
| 2/3 均值 | **72.772s** | **151.021s** | **225.044s** | 4/4 | 全部执行 |

相对同条件 `noso` 基线 129.374/209.566/340.178s:
**rollout 1.78× / update 1.39× / total 1.51×**。相对此前最优
SoA+Python-batch 120.444/150.159/271.900s:Rust 融合的独立增益为
**rollout 1.66× / update 0.99× / total 1.21×**。第三轮 update 190.518s
为系统抖动,已原样计入均值,未剔除。

## 15. 真实 2048 配置验证

`games_per_update=2048`,`update_epochs=2`,`target_kl=0.01`:exit 0;
2,092 games、1,640,549 transitions、2/2 epochs、2,140/2,140 minibatches、
3,281,098 executed samples,`early_stop=False`。耗时 rollout 275.312s、
update 287.917s、total 563.269s。相对旧同配置 464.087/276.544/740.673s:
rollout **1.69×**、total **1.32×**;新轮 transitions 多 1.77%,update 的 4.1%
差异与样本量/系统抖动一致,所有训练计算完整执行。

## 16. 剩余瓶颈与后续项

- GPU inference/full-forward 与 queue wait 已成为 rollout 主瓶颈;此前 big-batch
  sweep 已证实简单增大等待窗口会变慢,不推荐重试。
- history 继续由 Rust state-machine 提供;snapshot/critic 仍沿用既有批路径,
  没有伪称进入同一个 `encode_v16_batch`;听牌 yaku facts 与计算已迁入 Rust。
  全字段已通过 oracle,且 profiling 中不再是主导项。
- worker SoA/Ray 大数组已在第 18–25 节完成。direct action-id step、pinned
  slabs/DDP static graph 仍可深化,但新 profiling 显示它们不是当前主瓶颈。
- 流水线/双缓冲未合入:融合路径已把 rollout 推到 1.78×,继续改异步状态机会增加
  current-only transition 与 action/state 对齐风险,本轮选择在正确性边界停下。

## 17. 推荐正式配置与启动命令

使用自包含 `riichi_ppo_v1/configs/v17_ppo.yaml`;Action Query 现在无运行时开关,
固定使用 Rust 融合路径。保持 `update_use_soa` 默认 true,以及现有
12 worker × 32 env / 双 inference actor 拓扑;worker 使用
`rollout_worker_num_cpus=2`、`rollout_worker_cpu_threads=1`、
`worker_return_soa=true`,并保留 GPU 验证胜出的 `env_step_threads=4`:

```
env -C /mnt/disk1/hubowen/zenith \
  RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0,1 PYTHONUNBUFFERED=1 \
  /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python \
  -m riichi_ppo_v1.training.train \
  --config riichi_ppo_v1/configs/v17_ppo.yaml \
  --device cuda --learner-gpus 2 \
  2>&1 | tee logs/v17/v17_ppo.log
```

本轮只做验证,未启动正式长期训练。

---

## 18. Round 6:CPU 线程/Ray 资源与 worker SoA

新数据流:

```
worker 小局内 Transition(仅本进程,用于 reward/GAE 回填)
  -> 最终 GAE
  -> RolloutBuffer(flat arrays + offsets,29 ndarray/shard)
  -> Ray object store
driver
  -> 按 worker id 合并 SoA(不恢复 Transition)
  -> DDP round-robin select(含原有 filler index)
  -> learner 直接 collate SoA
```

Ray actor 的 `runtime_env.env_vars` 在 NumPy/PyTorch import 前只给 rollout worker
设置 `OPENBLAS/OMP/MKL/NUMEXPR=1`;actor 初始化最前设置 PyTorch intra/interop=1。
learner 没有被全局限流。每 worker 声明 2 CPU,12 worker 在 24 物理核/48 逻辑
线程机器上全部同时调度。

## 19. A/B/C/D CPU 三轮 sweep

12 进程 × 32 env × 500 ticks,每进程另跑 1000 次同结构 GRP forward:

| 组合 | step threads | worker CPU limit | table-step/s | GRP max | 总 wall | 最大线程数 | 非自愿切换 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A current | 4 | 无 | 280,136 | 3.336s | 5.706s | 52 | 27,583 |
| B step2 | 2 | 无 | 280,267 | 3.460s | 5.840s | 52 | 31,399 |
| C limit1 | 4 | 1 | 277,539 | **0.407s** | **2.596s** | 1 | 601 |
| D limit1+step2 | 2 | 1 | 274,074 | 0.374s | 2.626s | 1 | **400** |

线程限流令 GRP 尾时延约 8.2×、非自愿切换约 45.9× 改善;环境完整路径中
step=2 没有 CPU 收益。Ray actor 本身另有基础服务线程,运行时 `torch_threads` 与
`torch_interop_threads` 均实测为 1。

## 20. C/D 标准 GPU 三轮:step=2 负结果

相同 seed=1;第 1 轮预热,不剔除第 3 轮 update 抖动:

| 组合 | 计分轮 rollout | rollout 均值 | update 均值 | total 均值 | env step 均值 |
|---|---|---:|---:|---:|---:|
| C limit1,step4 | 72.148 / 73.384 | **72.766** | **149.939** | **223.978** | **1.803** |
| D limit1,step2 | 77.336 / 68.454 | 72.895 | 150.565 | 224.754 | 2.035 |

D 相对 C 分别慢 0.18%/0.42%/0.35%,所以回退 step=2 配置并保留正式默认 4。
C 相对 Round 5 A(72.772/151.021/225.044)端到端基本持平:线程限流是必要资源
治理,但 GPU 闭环收益只有 0–0.7%,不虚报为主要加速来源。

## 21. worker SoA 返回链路收益

计分轮 worker 计算分布几乎不变(C vs SoA p50 45.37→45.11s),说明采样计算没有
被删除。变化发生在返回边界:

| metric(计分轮均值) | C:旧对象 | C+worker SoA | 变化 |
|---|---:|---:|---:|
| Transition objects | 395,093 | 0 | 全部在 worker 内压缩 |
| ndarray count/global | 3,555,833 | 348(29/worker) | **10,218× 更少** |
| 有效数组字节/global | ~1.648GB | ~1.694GB | 信息未减少 |
| worker SoA pack p50 | 0 | 0.443s | 新增一次性成本 |
| semantic summary/worker | 0.0357s | 0.0375s | 不变 |
| object-store publish gap | 2.535s | 0.115s | **22.1×** |
| driver `ray.get` | 19.366s | 0.0026s | **约 7,350×** |
| driver merge | 1.266s | 0.486s | 2.61× |
| rollout wall | 72.766s | **50.684s** | **1.44×** |

旧路径的 `ray.get` 才是 worker mean/max 与 driver wall 巨大空白的主因;字节量并未
下降,收益来自少量大数组取代数百万 Python/ndarray 对象。

## 22. 标准 512g4e 最终三轮

配置:`CUDA_DEVICE=0,1`,`learner_gpus=2`,`games_per_update=512`,
`update_epochs=4`,`target_kl=0.0`,seed=1:

| 轮次 | transitions | games | rollout | update | total | epochs | minibatches | samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1(预热) | 514,702 | 553 | 71.548 | 125.113 | 197.313 | 4/4 | 1344/1344 | 2,058,808 |
| 2 | 394,384 | 553 | 53.180 | 91.733 | 145.443 | 4/4 | 1032/1032 | 1,577,536 |
| 3 | 393,523 | 554 | 48.188 | 168.465 | 217.106 | 4/4 | 1032/1032 | 1,574,092 |
| 2/3 均值 | 393,954 | 553.5 | **50.684** | **130.099** | **181.274** | 4/4 | 全部 | 1,575,814 |

相对 Round 5 A:rollout **1.44×**、update **1.16×**、total **1.24×**。相对最初
同条件 noso 129.374/209.566/340.178s:rollout **2.55×**、update **1.61×**、
total **1.88×**。第三轮 update=168.465s 原样计入。

主要 stage(计分轮均值):env step 1.796s/worker、Rust query 6.298s/worker、
RPC wait 29.214s/worker;inference 两 actor 求和 host collate 4.260s、H2D 2.596s、
full forward 41.338s、queue wait 100.813s。GPU utilization mean 68.04%、max 100%。

## 23. 真实 2048 最终验证

直接使用更新后的 `v17_ppo.yaml`,仅把 iterations 限为 1 且写入独立性能
checkpoint:2078 games、24,668 kyokus、1,610,326 transitions、2/2 epochs、
2100/2100 minibatches、3,220,652 executed samples,`early_stop=False`。

| metric | 旧 Rust-only | worker SoA 最终 | speedup |
|---|---:|---:|---:|
| rollout | 275.312s | **206.725s** | **1.33×** |
| update | 287.917s | **203.300s** | **1.42×** |
| total | 563.269s | **412.045s** | **1.37×** |

worker rollout min/p50/p90/max=175.19/185.36/200.30/201.82s;games/worker
min/p50/p90/max=172/173/174.9/176;drain steps 79–95。SoA pack p50/p90=
2.147/2.318s,driver get=1.9ms,合并=2.014s。返回仍为 348 arrays,每 worker
平均约 577MB,没有减少有效输入。GPU utilization mean/max=50.48%/100%。

旧轮与新轮都超过 2048 games 目标;新轮因异步 inference 服务顺序改变而轨迹长度
不同,但没有主动减少 games、GRP、transitions 或训练计算。

## 24. 正确性与计数验证

- SoA round-trip/concatenate/select/DDP 重复 filler:离散数组与有效长度逐元素全等;
  float32 先显式转协议 dtype,`atol=0,rtol=0`,最大误差 0。
- GAE advantages、empirical returns、相同模型权重与 shuffle seed 的 PPO
  loss/policy/value/entropy/KL 与旧路径一致。
- 全 action kind(tsumo/ron/reach/dahai/chi/pon/ankan/daiminkan/kakan/pass/
  none/ryukyoku)、V16 bridge/golden/oracle 沿用并通过。
- 最终测试:`riichi_ppo_v1/tests/unit` 167 passed;integration/protocol 34 passed;
  `riichienv-state-machine` Rust 10 passed;真实 Ray 冒烟通过并清理产物。
- 标准与 2048 的 GRP 调用与 kyoku 边界一致;最终标准计分轮分别约 6025/6017
  global GRP calls,2048 为 24,630;epochs/minibatches/executed samples 无减少。

## 25. 后续 profiling 决策

2048 中 Snapshot JSON + critic + action decode 合计约 10.63s/worker,只占 rollout
wall 约 5.1%;inference RPC wait 为 122.92s/worker。返回链路大空白已消除,所以本轮
不继续 Snapshot/Critic/direct-action Rust 融合。

固定配额仍造成 worker max/p50 约 1.09×,但每 worker games 仅 172–176。动态接管
需要跨 worker 协调完整半庄与小局收口,收益上界尚不足以覆盖语义风险;不采用达到
全局目标就取消慢 worker的错误方案。下一优先级应重新 profile inference 调度或
learner pinned/shared-memory 传输,而不是减少计算或再次放大 inference wait 窗口。
