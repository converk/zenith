# V18 PPO 性能优化第二轮执行提示词

> 本文档是自包含执行提示词,供新会话直接投喂执行。执行前无需重新做瓶颈分析,
> 所有测量数据、修改点(file:line)、验证标准、A/B 协议与回档规程均已内置。
> 执行会话**可以使用 sub-agents** 分担工作(分工与硬约束见 §9「子代理使用」;
> GPU 基准必须由主会话串行执行)。
> **总约束:不改变业务语义与训练语义**——所有改动只允许是执行路径/数据通路/
> 校验时机/调度方式的优化,任何触及损失函数、优势估计、采样分布、评测机制
> 语义的改动一律禁止。

## 1. 背景与实测瓶颈(2026-08-28,run2 = lr 9e-5/8e-5 第二轮,配置 `riichi_ppo_v1/configs/v18_ppo.yaml`)

生产配置一次迭代(`games_per_update=2048`,12 worker × 32 env,DDP 双卡,
`update_epochs=2`,`minibatch=1536`,梯度累积 10)实测:

| 指标(iteration 14) | 数值 | 说明 |
| --- | --- | --- |
| `iteration/algorithm_wall_s` | 825.2(逐轮 825~900) | rollout + update |
| `rollout/wall_s` | 234.7(28%) | 其中 worker 等推理 RPC **168.5s(72%)** |
| `update/wall_s` | 587.5(72%) | 见下行分解 |
| update forward / backward(每 rank) | 194.4s / 316.4s,共 868 minibatch | **224ms fwd + 364ms bwd / 批** |
| update 未计时开销 | ~67s | 分片 select 拷贝+mp.Queue pickle(~2.3GB/rank)、4 次 Python 循环 MC return、权重推送 |
| 模型执行效率 | fwd ≈11.6 TFLOPS / bwd ≈8.2 TFLOPS | L20 bf16 峰值 ~119.5,**约 10% MFU** |
| rollout 推理 padding | 25.8%(`padding_fraction_of_padded_tokens`,rank0) | update 侧分桶后仅 0.6% |
| rollout 推理批 | 每 rank 3742 次 forward,均值 178 行 / 22.5ms | 批大小 ≈ 进流速率 ×(wait+forward)自平衡 |
| worker 单步 CPU | model_state_prepare 25.7s + critic 编码 8.4s + env 5.9s(每 worker 每轮) | 大多被 RPC 等待掩盖 |

已证实的具体病灶(均已定位到行):

1. **前向内嵌 8~10 个 GPU→CPU 同步点**:`architecture.py:397`
   (`shared_lengths.max().item()`)、`:489`(`critic_total_lengths.max().item()`)、
   `:384-387/:425-426/:475`(`torch.any(...)` → bool 同步)、`:430-432`
   (`int(pair_counts.sum())`)、`:454-457`(重复 action-id `torch.any`)、
   `:284-303`(`_assert_structure` 内 4 次 `bool((...).any())`)、
   `dense_embedding.py:206-219`(argsort + `starts/ends/kind_values .tolist()`
   3 次同步 + 逐类别 Python 循环)。这些同步同时是 2026-08-27 torch.compile
   负优化(graph break)的根因。
2. **每个 minibatch 多跑一次完整 SFT reference 前向**:`learner.py:932-944`,
   `sft_kl_coef` 全程 >0(0.0025→0.0005),冻结模型每批一次 policy-only 前向,
   约占 update forward 的 ~40%。
3. **每 minibatch 4 次无条件诊断同步**:`learner.py:1003-1008` 的 `loss_detail`
   用 `float(...)` 取 4 个均值,仅在报错时才需要。
4. **梯度累积未用 `no_sync()`**:`learner.py:1011-1025`,accumulation=10 却每批
   allreduce;`:1040-1051` 另有每批一次 `all_reduce(MIN)` + `finite.item()` 同步。
5. **每批一次 Python 逐条 MC return 循环 ×4 次/update**:`learner.py:232-245`
   (每 rank update 内 1 次 + driver 聚合内再算),O(1.4M) Python 循环。
6. **SDPA 自定义 bool mask**:flash 后端被硬性排除(实测 "No available
   kernel");当前走 mem_efficient(正确)。实测(本机,B1536/H16/L134/D16,
   bool (B,1,L,L) mask):mem_efficient fwd+bwd **15.1ms** vs math 后端
   **78.8ms(5.2×)**;mask 构建本身仅 0.4ms。**风险**:mem_efficient 有隐含
   约束,任何回退到 math 都是 5 倍悬崖且无报错。
7. **推理批未按长度排序**:批内 padding 到最大长度,25.8% 推理计算是 padding。
8. **eval 结果缓存按文件名命中**:`train.py:482-490`,不校验 checkpoint 内容,
   已发生 run1/run2 复用同一份旧 JSON 导致评测记录污染(用户 2026-08-28 19:59
   手动 rerun 的 0.2695/0.2860 才是 u005/u010 的真实值)。

## 2. 回档点、预检与回滚规程

### 2.1 Git 回档点(执行会话第一步必须完成)

- 分支:`V16`;基线 HEAD:`b9af8a3debf94c4778f37b8adadc7b892877d13d`
  ("audit(progress): 记录 V18 PPO 第二轮(9e-5/8e-5)stop/归档并修正缓存评测
  记录",2026-08-28 20:43),工作区干净。
- 第一步:`git tag perf-v18-round2-baseline b9af8a3`。
- **单主题回滚**:每个任务一个独立 commit(见 §5-§7 各任务的提交要求);
  回滚用 `git revert <sha>`。涉及 Rust/wheel 的任务 revert 后必须重装扩展:
  `cd RiichiEnv && CONDA_PREFIX=/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI bash scripts/install_conda_extension.sh`。
- **全量回档**:`git reset --hard perf-v18-round2-baseline`(同样注意 wheel);
  `checkpoints/`、`datasets/`、`logs/`、`audit/` 产物不受 git reset 影响。
- 提交纪律:每主题一个 commit、测试通过、可独立回滚;message 沿用
  `perf(ppo): ...` 惯例;禁止多主题混合 commit。

### 2.2 执行预检(每次开始 GPU 工作前)

1. `ps aux | grep riichi_ppo_v1.training.train` 应为空(第二轮已于 2026-08-28
   20:43 stop/归档;若维护者又启动了新训练,则本轮只允许写代码/测试/文档,
   禁止跑 A/B、禁止重建 wheel——「长训练运行期间禁止重构」教训)。
2. `nvidia-smi`:本机 5 卡(0/1 L20 46GB、2 T400、3/4 L20)。**只用物理 0/1**
   (`CUDA_DEVICE=0,1`);3/4 常被其他任务占用,不要占用也不要报告为空闲异常。
3. 有 ray 残留会话时 `ray stop --force`;Python 一律用 `Mahjong-AI` conda 环境。

## 3. 范围裁定

### 3.1 纳入(本文档 §5-§7 的 A1-A5、B1-B4、C1-C5)

全部为代码路径优化;其中 C 组为进攻型,逐项 A/B 门控(§8)。

### 3.2 明确排除(执行会话不得擅自扩权)

- **不动 `v18_ppo.yaml` 任何既有配置值**(minibatch 1536、accumulation 10、
  `envs_per_worker=32`、`inference_batch_wait_ms=16` 等全部保持;维护者已决定
  配置类 A/B 不纳入本轮)。唯一允许的配置变更是 B1 需要的**新增开关键**
  (§6 B1),不属于调参改动。
- 不做 eval 与训练重叠、不做 rollout/update 流水重叠(staleness-1)、
  不做分片 /dev/shm 共享内存传输(留待下轮)。
- 不改 1v3 评测机制常量(`evaluation/mechanism.py`)、不改损失/优势/采样语义、
  不改 V18 模型拓扑与 checkpoint 契约(`ModelConfig` 不动,参数量不变)。
- 不触碰 `checkpoints/`、`datasets/`、`logs/v16|v17`、`audit/reports/v16|v17`
  等归档资产。

## 4. 语义等价性标准(三级)

| 级别 | 适用 | 验收方式 |
| --- | --- | --- |
| T1 逐位一致 | 纯数据通路/校验时机类(A2/A4/A5/B1/B3/C2) | 新增单测断言 `np.array_equal` / `torch.equal`(或逐位一致张量对照);既有测试全绿 |
| T2 容差一致 | 批组成/后端调度类(B2/B4) | 固定输入上新旧路径逐样本对照:logits/logprob/value `max abs diff ≤ 1e-4`(fp32 输出;若实测可达逐位一致则收紧为逐位断言);KL 派生量 `≤ 1e-5` |
| T3 A/B 门控 | 进攻型(C1-C4) | §8 协议,收益达标且数值/稳定性无损才保留,否则 revert 并记录放弃原因 |

新增单测放 `riichi_ppo_v1/tests/unit/`,命名延续现有风格
(`test_rollout_buffer.py`、`test_learner_ddp.py`、`test_v18_dense_embedding.py`
等已有文件内追加或新建)。验收线:执行开始时先在基线 HEAD 跑一次
`conda run --no-capture-output -n Mahjong-AI python -m pytest -q riichi_ppo_v1/tests`
记录基线,此后每个 commit 后必须不低于基线(目标全绿)。

## 5. 阶段 A(P0 低风险速赢,5 个 commit,合并做一次 A/B)

### A1 修复 eval 缓存污染(正确性,必须最先做)

- 位置:`train.py:482-504`(`run_1v3_evaluation` 的缓存命中分支)。
- 改法:缓存命中时校验 `checkpoint_path` 的 sha256 与 JSON 内记录一致
  (首次运行时把 `checkpoint_sha256` 写入 summary JSON;旧格式无该字段一律视为
  未命中并重跑覆盖)。93MB 文件哈希 ~0.3s,只在评测日发生。
- 验证:新增单测(tmp 目录伪造小 checkpoint + 旧格式/同哈希/异哈希三例)。
- 提交:`fix(eval): 1v3 缓存按 checkpoint 内容校验,防跨 run 复用旧结果`。

### A2 `loss_detail` 延迟求值

- 位置:`learner.py:1003-1008`。改法:改为仅在 `raise`(1009-1010 与 1047-1051
  的错误分支)时构造 detail 字符串;消除每 minibatch 4 次 `float()` GPU 同步。
- 级别 T1(无数值变化);不需要新单测,既有套件回归即可。
- 提交:`perf(ppo): loss 诊断字符串延迟构造,消除每 minibatch 4 次 GPU 同步`。

### A3 梯度累积 `no_sync()` + 有限性检查降频

- 位置:`learner.py:1011-1025`(backward)与 `:1040-1051`(all_reduce MIN)。
- 改法:
  1. 非组末 minibatch 用 `with self.model_ddp.no_sync():` 包裹 `backward()`
     (单卡/无 DDP 路径不受影响);
  2. `all_reduce(MIN)` 有限性检查从每 minibatch 降到每累积组一次(与
     optimizer step 同节拍;两 rank 的组边界由 padded minibatch 数严格对齐,
     collective 次数天然一致——这点必须写注释说明);
  3. `finite.item()` 同步随检查降频自然减少(868→87 次/update)。
- 等价性:T2。在 `test_learner_ddp.py` 增加双 rank(gloo, CPU)小模型对照:
  no_sync 累积路径与逐步 allreduce 路径最终参数 `allclose(atol=1e-6)`。
- 提交:`perf(ppo): 梯度累积 no_sync 与有限性检查按累积组节拍同步`。

### A4 `discounted_empirical_returns` 向量化

- 位置:`learner.py:232-245`。当前是 O(N) Python 循环,每次 update 全仓共执行
  4 次(每 rank update 内 1 次 + `learner_ddp.py` 聚合路径再算),N≈1.4M。
- 改法:分段(以 `done` 切局)后用 **float64 reversed `np.add.accumulate`**
  实现后缀和,最后 `astype(float32)`。关键约束:
  - gamma=1.0(生产值)时与旧循环的浮点加法顺序完全一致,必须 `np.array_equal`
    (旧循环用 Python float64 累加、逐步截断存储;float64 中间量 + 末端一次性
    cast 与之逐位一致);
  - gamma≠1 的 Horner 依赖无法用 accumulate 逐位复现,**保留旧循环路径**并按
    gamma 分派(通用性:不硬编码,按参数判断);
  - 分段边界用 `np.where(done)`。
- 验证:新增单测——随机 buffer 上 gamma=1.0 断言 `np.array_equal`,gamma≠1
  断言与旧实现 `np.array_equal`(直接对照保留的旧循环)。
- 提交:`perf(ppo): MC return 分段向量化(gamma=1 逐位一致)`。

### A5 learner collate 跳过 `query_rows`

- 位置:`rollout_buffer.py:298-382`(`collate`)与 `learner.py:909-911`
  (collate 后 `pop("query_rows")`——该字段 learner 不消费,纯浪费)。
- 改法:`collate` 增加 `include_query_rows: bool = True`(默认值保持其他调用方
  兼容),learner 预取路径传 False。
- 级别 T1;收益主要是预取线程 CPU 与内存(本身与 GPU 重叠,墙钟收益 0~10s),
  属卫生项。
- 提交:`perf(ppo): learner collate 跳过未消费的 query_rows`。

**阶段 A 完成后**:全套件回归 → §8 A/B → 预期 `update/wall_s` −35~70s。

## 6. 阶段 B(P1 核心代码路径,4 个主题,B1/B2 各自 A/B)

### B1 前向去同步化(compile 的前置)

- 位置:`architecture.py:384-387/397/425-432/454-457/475/489` 与
  `dense_embedding.py` 不在本项(留给 C2)。
- 改法:
  1. **host 侧 capacity**:`collate`(`rollout_buffer.py:298-382`)在 numpy 里算好
     `shared_capacity = int((行内 segment==SEGMENT_SHARED).sum(-1).max())`
     (padding 为 0,不污染计数)与
     `critic_total_capacity = int(max(shared_len + critic_len + 1))`,
     以 `int | None = None` 可选参数传入 `forward`;为 None 时保留现 `.item()`
     回退(测试/其他调用方零改动)。
  2. **校验开关**:forward 增加关键字参数 `validate_structure: bool = True`;
     为 False 时跳过 `:384-387/:425-426/:454-457/:475` 的 GPU 侧校验与
     `_assert_structure`(`:388` actor 调用与 `:473` critic 调用)。learner 与
     inference actor 按配置 `update_validate_structure`(新键,默认 True 保持
     现行为)传入;`v18_ppo.yaml` **新增**(不改动任何既有键)
     `update_validate_structure: false`,注释说明依据:输入由 Rust 编码器
     fail-closed 生成 + SFT 契约校验 + 单测覆盖,训练期跳过仅移除重复检查。
  3. **可选(仅在契约可证时做)**:`:430` 的 `torch.nonzero(query_mask)` 是
     最后一个同步点。若能用真实数据断言「action query 行恒为每行序列尾部的
     连续 2×pair_count 行」(canonical 契约),则改为算术索引
     (`col = lengths - 2*pair_counts + arange` 聚合)消除同步;先写断言测试,
     断言不过就保留 nonzero 并在注释记录。
- 级别 T1(同值不同来源 + 移除校验,无数值变化);验证:开关两态在合法输入上
  输出 `torch.equal` 的单测 + 既有 architecture/dense 测试全绿。
- 提交:`perf(ppo): 前向 capacity host 化与训练期结构校验开关`。
- A/B:预期 update forward/backward 各 −10~20%,`update/wall_s` −60~110s。

### B2 SFT reference 每 update 预计算一次

- 位置:`learner.py:932-944`。当前每 minibatch 对冻结 reference 跑一次
  policy-only 前向(约 40% forward)。
- 改法:`update()` 内、epoch 循环前,对本 rank 分片用大批量(如 8192 行)
  `torch.inference_mode()` 一次性算完 reference 的 `policy_logits`
  (含 legal mask 的 -inf masking,与现路径完全同构),以 fp32 张量
  `[N_shard, 241]`(~665MB,GPU 46GB 充裕)驻留至 update 结束;
  minibatch 内按 `indices` gather 后交给既有 `categorical_kl_values`。
- 等价性:T2:随机抽 ≥3 个真实 minibatch,新路径 KL 与现路径逐样本对照
  `max abs diff ≤ 1e-5`;若实测逐位一致则收紧断言。若大批量与 minibatch 批
  差异超容差(不应发生),回退为按 1536 行分块预计算(批组成与 update 完全
  一致,必然逐位一致)。
- 注意:`policy_only=True` 路径不产生 value;bootstrap 期(`sft_kl_coef=0`)
  跳过预计算。
- 提交:`perf(ppo): SFT reference logits 每 update 一次预计算`。
- A/B:预期 `update/wall_s` 再 −40~70s。

### B3 SDPA 后端锁定(防 5 倍回退悬崖,防守型)

- 位置:`architecture.py:102-104`(`GQA.forward` 的 SDPA 调用)。
- 改法:用 `torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)`
  包裹该调用(或 learner/inference 初始化时
  `torch.backends.cuda.enable_math_sdp(False)` + 启动探针断言 mem_efficient
  对 bool mask 可用)。锁定后若约束不满足会 fail-fast 报错而非静默 5 倍变慢。
- 级别 T1(后端未变,无数值变化);验证:单测在 CUDA 上断言 flash-only 上下文
  对该 mask 报错、efficient-only 正常(无 CUDA 跳过)。
- 提交:`perf(ppo): 锁定结构化 mask 的 SDPA mem_efficient 后端`。

### B4 推理批按长度排序(padding 25.8% → ~5-8%)

- 位置:`inference.py:470-493`(`_infer_many`)与 `:59-112`
  (`collate_request_rows`)。
- 改法:在 `_infer_many` 内对每个 `(namespace, greedy)` 的 rows 列表先按该行
  `actor_lengths`(次序 `critic_lengths`)降序排序,再按 `max_batch` 切块;
  输出路由 `assign_batch_outputs`(`:115-130`)本就按 group 顺序回填,排序
  天然正确,无需改路由。
- 等价性:T2:构造固定请求集,排序前后每行 action/logprob/value 对照
  (行间独立计算,预期逐位一致;若 bf16 内核引入差异则按 ≤1e-4 容差断言),
  另加「每个请求行收到自己的结果」的路由断言。
- 提交:`perf(ppo): 推理批内按序列长度排序,消除跨局 padding`。
- A/B:看 `rollout/wall_s`(预期 −25~40s)与
  `rollout/inference_actor/rank*/inference/padding_fraction_of_padded_tokens`。

## 7. 阶段 C(P2 进攻型,逐项 A/B 门控;任何一项不达标即 revert 并记录)

### C1 torch.compile 重试(必须晚于 B1)

- 背景:2026-08-27 全模型 compile 在 DDP 下灾难性退化(8s/步,IndexPutBackward0
  图断裂);B1 落地后 `nonzero/item` 图断裂源已除,critic 组装路径的 IndexPut
  仍在——**只编译三个 Decoder 子模块**
  (`public_backbone/actor_backbone/critic_backbone`),
  `torch.compile(module, dynamic=False, fullgraph=True)`,不编译整个模型。
- 接入:`PPOLearner.__init__` 与 inference actor 按新配置键 `torch_compile`
  (默认 false)启用;失败模式(fallback eager)必须自动降级并打日志。
- 验证:固定批输入 compile vs eager logits `allclose(rtol=1e-2, atol=2e-2)`
  (bf16 融合重排容差)+ §8 A/B。
- 放弃判据:fullgraph 编译失败且回退 fullgraph=False 仍无收益;A/B 变慢;
  或 DDP 下出现同步/稳定性异常。三者任一即 revert 并在 PROGRESS.md 记录。
- 预期:update 再 −15~30%。

### C2 StateTokenEmbedding 重写(消除 argsort/tolist/动态分派)

- 位置:`dense_embedding.py:190-245`。当前 forward:`torch.argsort(stable)` →
  `starts/ends` → `starts.tolist()/ends.tolist()/kind_values.tolist()`(3 次同步)
  → 按动态 kind 值的 Python 循环。此层约占 forward ~24%。
- 改法(方向,实现可自由发挥但受约束):类别集合是**编译期静态**的
  (`CATEGORY_SCHEMAS`)——改为按静态 kind 键表迭代,段边界用
  `torch.searchsorted(sorted_kind, static_kind_list)` 得 GPU 张量切片,
  `index_select`/`index_copy_` 完成聚散,全程无 `.tolist()`/`.item()`;
  槽位元数据(列下标、表引用、slot 布局)在 `__init__` 预计算为常量结构。
- 级别 **T1 逐位一致**(同数学、同顺序);验证:新增单测对照旧实现
  (随机张量 + 真实 fixture)逐位一致,`test_v18_dense_embedding.py` 全绿。
- A/B:预期 fwd −10~20%。

### C3 推理 CUDA Graph(bucket 固定形状)

- 位置:`inference.py:495-595`(`_run_full_forward`)。当前每 forward
  ~5-6ms 固定开销(launch+同步)。
- 改法方向:namespace=`rollout` 的 current 模型专用(其余 namespace 保持 eager);
  按 (batch_bucket × L_bucket) 固定形状 capture,batch 桶 {128, 256, 512}、
  L 桶在 {112, 144, 176, 208, 256} 就近向上取;不足桶的行用合法 dummy 行补齐
  (输出丢弃);**只 capture 到 policy_logits 为止,sampling 留在图外**;
  静态输入缓冲 + pinned 拷贝。eval/评测路径不动。
- 验证:同形状 graph vs eager logits 逐位一致(`torch.equal`)断言;
  microbench 吞吐 ≥ +25% 才进入 A/B。
- 放弃判据:任一形状逐位不一致、OOM、或 A/B 无收益。复杂度最高,排在 C1/C2
  之后,允许放弃。
- 预期:rollout 推理吞吐 +40~60%,`rollout/wall_s` 再 −20~40s
  (受 worker 侧 RPC 往返制约,收益上限见 §10)。

### C4 FlexAttention(可选,C1 成功后的追加)

- `torch.nn.attention.flex_attention`(torch 2.7.1 已内置)把
  `_actor_structured_layout`/`_bidirectional_layout`/`_critic_layout`
  (`architecture.py:124-177`)的 mask 语义改写为 `mask_mod` 函数,编译成
  融合 Triton kernel,并利用 action-pair 块稀疏跳过。
- 级别 T2(kernel 数学不同,rtol=1e-2/atol=2e-2)+ A/B;放弃判据:编译失败、
  无收益、或与 C1 的 compile 组合不稳定。预期 attention 份额再压 50%+
  (整体 update 约 −5~7%),优先级最低。

### C5(可选尾巴)Rust 下沉 `critic_feature_encode`

- `rollout/timing/state/critic_feature_encode` ≈8.4s/worker/rollout
  (经 rg `critic_feature_encode` 定位于 bridge 链路)。PyO3 化并要求
  `np.array_equal` 逐位一致 + `cargo test --workspace` 全绿 + wheel 重装。
- 仅在 B4/C3 落地后 worker CPU 成为新约束时才有意义,允许跳过。

**执行顺序**:A(5 commits)→ A/B → B1 → A/B → B2 → A/B → B3 → B4 → A/B →
C1 → C2 → C3 → C4/C5(视前序结果)。A/B 期间同样禁止并行改代码
(「长训练运行期间禁止重构」适用于 perftest 运行窗)。

## 8. A/B 性能协议(每主题/每阶段)

1. **基准配置**:新建自包含 `riichi_ppo_v1/configs/v18_ppo_perftest.yaml`
   (只新增文件;以 `v18_ppo.yaml` 全文为底本复制后仅改以下键):
   `iterations: 3`、`total_updates: 3`、`games_per_update: 512`、
   `update_epochs: 4`、`target_kl: 0.0`、`eval1v3_enabled: false`、
   `checkpoint_dir: checkpoints/train_riichi_v18/ppo_perftest`。
   其余键(含 minibatch 1536 / accumulation 10 / 12 worker / 32 env /
   wait 16ms)与生产配置逐键一致;`test_config_loading` 相关约定:自包含、
   不 overlay。
2. **运行命令**(任意目录,AGENTS.md 标准启动):

   ```bash
   env -C /mnt/disk1/hubowen/zenith RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0,1 \
     PYTHONUNBUFFERED=1 python -m riichi_ppo_v1.training.train \
     --config riichi_ppo_v1/configs/v18_ppo_perftest.yaml --device cuda \
     --learner-gpus 2 2>&1 | tee logs/v18/ppo_perf2_<主题>_<date>.log
   ```

3. **轮次**:每主题跑 3 轮,第 1 轮视为预热单独记录,报告第 2/3 轮统计
   (AGENTS.md 约定)。基线 = 未修改 HEAD(`b9af8a3`)先跑一轮 3 轮记录,
   避免机器负载漂移误判(2026-08-27 第 3 轮曾受外部负载影响)。
4. **对比指标**(读 `checkpoints/train_riichi_v18/ppo_perftest/performance.jsonl`):
   `iteration/algorithm_wall_s`、`rollout/wall_s`、`update/wall_s`、
   `iteration/sps`、`ppo/timing/update/model_forward/total_s`、
   `ppo/timing/update/backward/total_s`、
   `ppo/timing/update/collate_h2d/total_s`、
   `rollout/timing/inference/rpc_wait/worker_mean_total_s`、
   `rollout/inference_actor/rank0/inference/full_forward_rows_mean`、
   `rollout/inference_actor/rank0/inference/padding_fraction_of_padded_tokens`、
   `rollout/inference_actor/rank0/timing/inference/full_forward/mean_ms`。
   **读数陷阱**:`timing/*/total_s` 是跨 rank 取 max、`/count` 是跨 rank 求和,
   每 rank 每 minibatch 耗时 = total_s ÷ (count ÷ 2);聚合后的
   `full_forward_rows_mean` 是两 rank 之和(约为单 rank 2 倍)。
5. **清理**:每轮 A/B 结束删除本轮产生的
   `checkpoints/train_riichi_v18/ppo_perftest/`(临时基准产物,非训练
   checkpoint;`logs/v18/` 的 perftest 日志保留作证据)。
6. **判定**:阶段 A/B 主题以「指标不劣化 + 预期收益方向兑现」保留;C 组按
   §7 各自放弃判据。

## 9. 全局工程约定(执行全程有效)

### 子代理(sub-agents)使用

- **允许并鼓励使用 sub-agents** 分担执行工作量,推荐分工:
  - *Explore(只读)*:全仓引用检查(`rg`)、修改点与调用方定位、既有测试
    盘点、配置键/指标键核对——不产生任何写入,适合并行派发;
  - *general-purpose*:运行 pytest 套件、解析 `performance.jsonl` 汇总 A/B
    指标、起草 PROGRESS.md 条目与实施记录报告。
- 硬约束:
  1. **GPU 独占**:pytest 套件(单测/integration,均为 CPU 路径)可交由
     sub-agent 运行;除此之外任何 CUDA 工作——perftest A/B、冒烟、模型前向
     验证——只能由主会话**串行**执行,严禁派发给 sub-agent 并行运行。双卡被
     两个进程争抢会把 3 轮基准数据全部污染(本机 3/4 卡属其他任务,同样不可
     占用);
  2. **版本控制不下放**:`git commit / revert / tag` 只能由主会话执行,
     §2.1 的回滚纪律不委派给 sub-agent;
  3. **结论须复核**:sub-agent 返回的行号、测试结果、指标数字必须由主会话
     抽查关键证据后方可写入 commit 或 PROGRESS.md;
  4. sub-agent 同样遵守「训练/perftest 运行窗内禁止改代码、禁止重建 wheel」
     (§2.2 预检第 1 条)。

- 代码注释一律中文;新增配置键必须有注释说明依据与默认值语义。
- 通用性:不硬编码版本号、checkpoint 名、路径、对手模型、批大小常量;
  新开关一律走配置项,默认值不改变未配置方的历史行为。
- 常量单源:新增桶/容量等常量进 `encoding_protocol.py` 或对应模块常量区,
  禁止散落。
- 删除任何代码前先 `rg` 全仓引用检查;本轮预期**零删除**(只有修改与新增)。
- 不修改 `KyokuEventTupleProtocol.md` 等协议文档(无协议变更);若 B1 的
  `update_validate_structure` 落地,在 `riichi_ppo_v1/docs/` 对应训练文档
  补一段开关说明。
- 进度记录:每完成一个主题,在 `audit/reports/v18/report/PROGRESS.md` 追加
  条目(主题、commit、A/B 数据、回滚点)。
- 全部完成后写实施记录报告
  `audit/reports/v18/report/V18_PPO性能优化第二轮实施记录.md`
  (格式参照 `V18_PPO训练性能优化实施记录.md`:改动表、A/B 表、放弃项、
  回归结果、回滚说明)。

## 10. 验收清单

1. `conda run --no-capture-output -n Mahjong-AI python -m pytest -q riichi_ppo_v1/tests`
   不低于基线(目标全绿);涉及 Rust 的 C5 另跑
   `env LD_LIBRARY_PATH=$CONDA_PREFIX/lib conda run --no-capture-output -n Mahjong-AI cargo test --manifest-path RiichiEnv/Cargo.toml --workspace`。
2. 每主题 commit 独立可 revert;`git log` 无混合主题提交。
3. 生产配置 `v18_ppo.yaml` 除 B1 明确允许的**新增**键外逐键与基线一致
   (`git diff b9af8a3 -- riichi_ppo_v1/configs/v18_ppo.yaml` 自查)。
4. 性能对照表(基线 vs 最终,512 局 perftest 3 轮取 2/3 轮):

   | 指标 | 基线 | 最终 | Δ |
   | --- | --- | --- | --- |
   | algorithm_wall_s / iteration | ~?(基线实测) | | |
   | update/wall_s | | | |
   | rollout/wall_s | | | |
   | update fwd / bwd 每-minibatch ms | 224 / 364(生产规模参考) | | |
   | inference padding_fraction | 0.258 | | |
   | sps | | | |

5. 预期总收益(本范围,供校对,非验收线):update −135~250s、
   rollout −25~80s,`algorithm_wall_s` 825 → 约 **495~665s**;
   若 C 组全部放弃,保底约 600~710s。收益兑现不足 50% 时,在实施记录中
   逐项归因(而非继续加项)。
6. 明确遗留(写入实施记录「后续轮次」):配置类 A/B(minibatch 3072+accum 5、
   envs_per_worker 64、wait_ms 扫描)、eval 与训练重叠、staleness-1 流水重叠、
   分片共享内存传输。
