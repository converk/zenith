# V17 PPO 原生性能重构 · 最终报告

> 日期:2026-08-25 · 状态:最终(Rust 融合后 rollout/update/total 均达到验收)
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

## 10. 推荐正式训练配置(结合当前最优)

- 启用 `update_use_soa: true`(默认)与 `v16_batch_query: true`(默认)。
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
  -> 仅听牌行一次 riichienv yaku batch 回填 O4/O5
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
- `riichi_ppo_v1/tests/unit`:164 passed;V16 bridge 集成 11 passed;Rust crate
  10 passed;RiichiEnv 定向测试 14 passed。
- V16 schema、GRP/PPO/GAE/value/entropy/SFT KL、current-only transition、
  opponent mix、checkpoint/DDP 数学均未修改;无需改协议文档。

## 13. 独立收益与 profiling

CPU 真实决策批(8 env × 120 ticks,8,822 action rows):

| 路径 | `state/v16_query_assembly` | 相对 |
|---|---:|---:|
| Python batch oracle | 1.493045s | 1.00× |
| Rust compact+fused | 0.136433s | **10.94×** |

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
- history 继续由 Rust state-machine 提供;snapshot/critic 与少量听牌 yaku 回填
  仍沿用既有批路径,没有伪称进入同一个 `encode_v16_batch`。它们已通过全字段
  oracle,且 profiling 中不再是主导项;若继续融合应单列后续任务与收益基准。
- direct action-id step、worker pending SoA/Ray 大数组、pinned slabs/DDP static
  graph 仍可深化,但不再是本次 ≥1.20× 验收的阻塞项。
- 流水线/双缓冲未合入:融合路径已把 rollout 推到 1.78×,继续改异步状态机会增加
  current-only transition 与 action/state 对齐风险,本轮选择在正确性边界停下。

## 17. 推荐正式配置与启动命令

使用自包含 `riichi_ppo_v1/configs/v17_ppo.yaml`,显式保持
`v16_batch_query: true`,`v16_rust_encoding: true`,`update_use_soa` 默认 true,
以及现有 12 worker × 32 env / 双 inference actor 拓扑:

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
