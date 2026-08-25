# V17 PPO 原生性能重构设计(V17 Native Performance)

> 版本:0.1(设计草图)
> 关联产物:`specs/007-v17-ppo-native-performance/`
> 目标:在不改变训练语义与计算量的前提下,让 `rollout_wall_s`、`update_wall_s`
> 及「rollout+update」总时间至少加速 1.20×(争取 1.30×)。

---

## 1. 当前架构与热路径

### 1.1 事件流(Rollout)

```
BatchedRiichiEnv(Rust 批量 env, 每 worker 1 个, 32+ 桌)
  -> Python Observation(dict[seat -> Observation], 每桌 4 个)
  -> active_decisions(): 扫描每个 seat 的 legal_actions()
  -> Bridge.prepare_v16(decisions):
       - action_jsons_and_decision_flag(): 每个合法动作 -> MJAI JSON 字符串
       - snapshot_json(): 每决策 -> snapshot JSON 字符串
       - state_machine.prepare_decisions(): 一次 Rust 调用, 产 history factors + legal mask
       - state_machine.decode_actions(): 每个合法 action id -> MJAI JSON
       - analyze_action_queries(): 每个动作 1 对 Offense/Defense Query(Python 循环,
         内部再调 2~3 次 PyO3: riichi.analyze_offense_v16 / analyze_defense_v16 /
         riichienv.analyze_offense_v16 / riichi.analyze_hands)
       - encode_query_row / encode_snapshot_rows / encode_critic_features(Python + PyO3)
       - numpy padding -> V16PreparedBatch
  -> InferenceActor.infer.remote(...) (async)
  -> ray.get(results)
  -> Bridge.decode(): action id -> MJAI -> Observation.select_action_from_mjai -> Action
  -> env.step_batch(actions) (Rust)
  -> bridge.sync(observations) -> state_machine.apply_events_batch (Rust)
  -> kyoku 结算: GRP boundary reward + finish_kyoku_gae -> list[Transition]
```

### 1.2 瓶颈排序(旧 profiling 线索, 需实测确认)

针对实际 2048 局配置的旧线索(rollout ≈ 408s / update ≈ 456s):

| 阶段 | 旧线索 | 性质 |
|---|---|---|
| `rollout/model_state_prepare` | ~212s | V16 编码 + JSON + 逐动作 PyO3 |
| `state/v16_query_assembly` | ~200s | **逐动作 Python 循环 + 2~3 次 PyO3/动作** |
| `inference/rpc_wait` | ~81s | GPU 前向与跨 worker 批处理 |
| `env/step_batch_native` | ~6s | Rust 原生步进(已很快) |
| `update/host padding/copy` | ~105s | **逐 Transition `torch.tensor` 拷贝** |
| `update/forward`+`backward` | ~246s | GPU 计算(不可削减, 禁止降计算量) |

结论:两大可优化支点是 **(A) rollout 的 V16 query/snapshot/critic 编码** 与
**(B) learner 的逐样本 host collate / H2D**。

---

## 2. 优化方案总览

### 2.1 学习者一次性物化(SoA buffer)——已实现

- 新文件 `riichi_ppo_v1/training/rollout_buffer.py`:`RolloutBuffer` 在 worker GAE
  完成后一次性把 `list[Transition]` 抽成扁平 SoA(变长字段用 offset 索引),
  `collate(indices)` 用向量化 gather 替代逐样本 `torch.tensor` 拷贝。
- worker、driver、learner 与 DDP 固定走 RolloutBuffer 单路径;旧逐对象 fallback
  仅用于验收对照,现已删除。
- 实测:host 拷贝从 ~37.9ms/512行 降到 ~2.2ms/512行(≈17×);当前测试直接
  校验全部字段、padding、分片与冻结 GAE/returns 公式。
- 收益区间:标准 512局/4epoch 下 `update/host padding copy` 约 100s → ~6s。

### 2.2 Rust 融合快速路径(待实现, 主目标)

把 (1.1) 的 Python 编码热路径合并为「带类型的 Rust 批量编码」:

```
compact action ids
  -> Rust 批量 step/reset(BatchedRiichiEnv 已具备)
  -> typed event/state update(state_machine 已具备)
  -> active decision 扫描 + legal mask(Rust)
  -> V16 history/snapshot/query/critic 编码(Rust, 一次性产出 SoA buffer)
  -> 连续 compact batch(0/短 PyO3 边界)
```

具体子任务:

1. **消除高频 MJAI JSON serialize/parse**:`prepare_decisions` / `decode_actions`
   的输入从 `Vec<String>`(JSON)改为结构化 Rust 数据(或把观察者事实直接以
   compact array 传入),避免每次决策的 `action.to_mjai()` + `json.dumps` +
   `serde_json::from_str`。
2. **消除 Observation PyObject / dict/list / 字符串**:`BatchedRiichiEnv` 暴露
   一个「紧凑观察快照」接口(hands/melds/discards/scores/dora 等以 flat array),
   供 Rust 编码器直接消费,不再在 Python 侧遍历 `Observation` 属性。
3. **把 `analyze_action_queries` / query row / snapshot / critic feature 迁移或
   等价实现到 Rust**:新增 `riichienv-state-machine` 的批量函数(如
   `encode_v16_batch`),输入整份 batch 的 compact 事实,输出
   history/snapshot/query/critic/legal_mask。
4. **inference 返回 action id 后由 Rust 直接应用**:`bridge.decode` 已是 Rust
   侧 `decode_actions`;再让 `select_action_from_mjai` 退出 Python,直接用
   compact action id 驱动 `env.step_batch`。
5. **每个 batch tick 只保留少量粗粒度 PyO3 调用**:把每动作 2~3 次 PyO3 汇聚为
   每 batch 1~2 次(见 3.2 中间方案)。
6. **Rust 计算释放 GIL + 优先持久线程池**:`py.detach` + `std::thread`/rayon 池。
7. **返回连续 SoA NumPy/共享内存 buffer**:明确 dtype/offset/length/所有权。

### 2.3 并发拓扑

- 48 核是 CPU 线程预算,不是逻辑环境数上限。逻辑环境可 >48 以撑大 inference batch。
- 建议比较 3 组拓扑(见 4.2)。

---

## 3. V16 编码层实现选择

### 3.1 方案 A:完整 Rust 编码(最终目标)

在 `riichienv-state-machine` 增加 `encode_v16_batch`:

```
encode_v16_batch(
  env_compact: Vec<CompactTableState>,  // 每桌面事实(公开+本家)
  decisions: Vec<(env_index, seat_id, action_ids)>,
  walls: Vec<[...]>,
) -> (history_factors, history_numeric, snapshot_kinds/cat/num,
      query_rows, query_action_ids, legal_mask, critic_factors/lens)
```

内部复用现有 `analyze_offense_v16` / `analyze_defense_v16` / `analyze_hands` /
`public_opponent_summary` 的 Rust 内核,但避免逐行 Python 往返。

### 3.2 方案 B:PyO3 批量汇聚(中间步, 低风险)

不重写动作类型分派逻辑,而是把「每动作 1 行调用」改为「每 batch 1 次调用」:
- 先在 Python 侧收集所有动作的 compact 输入(手牌计数/剩余/河掩码/meld);
- 一次性调用 `riichi.analyze_offense_v16` / `analyze_defense_v16` /
  `analyze_hands` / `public_opponent_summary`(一次数组行);
- 在 Python 侧做纯编码/组装。

收益:PyO3 调用次数从 O(动作数)降到 O(1)/batch;但仍保留 Python observation
遍历。作为方案 A 之前的可交付中间态。

### 3.3 数据通路(compact rollout buffer)

- worker 内:每桌 pending 从 `list[Transition]` 改为「按 seat 追加的变长 SoA」;
  完成小局时 GAE 直接写入 flat advantage/returns。
- driver→learner:不再用 `list[Transition]`,改用 Ray object store 中少量大数组
  (flat values + offsets/lengths),或 worker 直接向 learner rank 交付 shard。
- buffer 至少含:变长输入、legal mask、action、old logprob/value、reward、
  advantage、return、env/seat/边界、policy namespace。

---

## 4. 测量与基准

### 4.1 标准基准

- `CUDA_DEVICE=0,1`、`learner_gpus=2`、`target_kl=0.0`、`update_epochs=4`、
  `games_per_update=512`、相同 seed、3 轮(第 1 轮预热,报 2/3 及均值)。
- 另用 `v17_ppo.yaml` 实际完整参数(2048 局、2 epochs、target_kl=0.01)验证一次。

### 4.2 拓扑组合(至少 3 组)

| 拓扑 | num_workers | envs_per_worker | env_step_threads | 说明 |
|---|---|---|---|---|
| T1 少 actor 大 vector | 4 | 96 | 8 | 少 Ray actor, 大推理批次 |
| T2 中 actor 中 env | 8 | 48 | 4 | 中间 |
| T3 当前类似 | 12 | 32 | 4 | 现状 |

---

## 5. 正确性验证(冻结基线)

- 优化前冻结旧路径为 oracle,固定 seed 生成 golden trace(见
  `scripts/freeze_golden_baseline.py`),比较:action opportunities、legal
  id/mask、history/snapshot/query/critic tensor、action id↔env action、
  state-machine events、end_kyoku/end_game、scores/ranks、transitions、
  rewards/advantages/returns、done/GAE boundary。
- 整数/离散 tensor 逐元素一致;浮点给出严格容差与误差上界。
- 保持:V16 编码/action schema、GRP reward、rank utility、PPO/GAE/value/
  entropy/SFT KL 数学、current-only transition、opponent mix 语义、
  checkpoint exact resume、DDP 梯度平均及 rank 间权重一致。

---

## 6. 交付物

- `specs/007-v17-ppo-native-performance/`(spec/plan/tasks)
- `audit/reports/v17/design/`(本文件)
- `audit/reports/v17/eval/`(三轮基准输出)
- `audit/reports/v17/report/`(最终报告 + PROGRESS)
- `audit/reports/v17/scripts/`(运行/验证脚本)
- `logs/v17/`(运行日志)

---

## 7. Rust 融合落地设计(2026-08-25)

最终实现采用两个独立 Rust crate 的单向粗粒度边界,保持公开模块 `riichi` 不依赖
`riichienv`:

1. `riichienv.prepare_v16_compact_facts(observations, observation_indices,
   actions, action_ids, boundary_overrides...) -> CompactV16Facts`:每个唯一
   Observation 只提取一次,
   在 core 内完成 post-shape、remaining、river、meld/kan、kind/O7/O8/O9 与
   defense facts,返回 20 个 C-contiguous SoA NumPy 数组。
2. `riichi.encode_v16_batch(...) -> V16BatchEncoding`:释放 GIL,复用
   `offense_row_v16`、`defense_row` 与 `shanten`,直接写
   `query_rows int32[N,2,15]` 和 `wait_masks uint64[N]`;附带
   `unique_offense_rows/unique_shanten_rows` 审计计数。
3. 只有 `wait_masks != 0` 的少量听牌行由
   `riichienv.analyze_v16_yaku_batch` 在 Rust 内重建 post-hand 并批量计算 O4/O5。
   Python 只把连续结果写回 query buffer;依赖方向不反转,全部 action 的 query 行
   仍逐一保留。

`query_rows` 的末维严格保持 V16 15 列:
`[query_type, action_id, action_type, primary_tile+1, source_seat+1,
answer_0..answer_9]`;每动作固定 offense/defense 两行。其余输入的 dtype/shape
由两个 PyO3 API 做严格校验,非法 shape/count/kind 立即报错。

state-machine 的 pending request 同时保存固定 action id→原始合法 Action 下标,
`bridge.prepare_v16` 不再执行 `decode_actions` 后的 JSON parse/canonical/match。
生产桥接只保留 Rust 融合路径,不再暴露 Python batch/逐动作回退开关。逐动作实现
也不再由 Python 重复计算:SFT、审计与历史调用方仍可使用
`analyze_action_queries`,但该接口只把一个动作交给 Rust 融合编码器并恢复兼容对象。
动作分派、post-shape、向听、有效牌、防守与役种均以 Rust 为唯一语义来源。
