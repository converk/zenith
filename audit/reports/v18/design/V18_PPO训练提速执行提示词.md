# V18 PPO 训练提速执行提示词

> 本文档是交给执行会话的任务提示词。任务:在不改变训练超参数的前提下,
> 把 V18 PPO 每个 update 的墙钟时间(稳态约 731s)显著降低。
> **维护者 2026-09-01 批准 Tier 2 全部方向:训练超参数与模型输入语义
> (V18 输入协议、Actor 公共信息边界、Critic 私有输入)不得改动;训练期
> 语义(浮点归约顺序、数据陈旧度、minibatch 组织等)允许修改。**
> 以下所有耗时数据
> 来自真实训练运行 `logs/v18/v18_ppo_r5_20260829.log`(150 updates)与
> `checkpoints/train_riichi_v18/ppo/performance.jsonl`(分段计时),行号基于
> 当前 HEAD。开始前请先读 `AGENTS.md`(最高治理文档)、本文、上述日志与
> `audit/reports/v18/report/V18输入链路复审与性能审查报告.md`,以及已有的
> `audit/reports/v18/scripts/v18_perf_review.py`。

---

## 一、硬约束(不可违反)

1. **训练超参数一律不得改动**:gamma、gae_lambda、ppo_clip、
   gradient_accumulation_steps=10、全部学习率/熵系数/KL/SFT-KL/value 系数、
   target_kl、seed;正式训练与最终验证口径下 `games_per_update=2048`、
   `update_epochs=2` 同样不得改。优化过程中 `v18_ppo.yaml` 保持一字不动
   (交付收尾的最后一笔 commit 除外,见第七节);性能测试一律使用独立自
   包含配置(见第五节),其中**只允许**改工作负载与编排键:
   games_per_update、update_epochs、iterations/total_updates、resume、
   eval/checkpoint/日志输出重定向,以及本任务新增的性能开关键(用于
   关/开消融);PPO 超参与
   rollout/推理拓扑(num_workers、envs_per_worker、inference_*)必须与
   正式配置逐项相同。
   **唯一例外**:`minibatch_size` 最多允许增大到 **2048**(显存上限;注意这会把
   等效有效 batch 从 1536×10×2=30720 变为 40960,报告必须明确标注这一影响;
   1536 时 learner 峰值 allocated ≈22GB / reserved ≈38GB,46GB 卡)。
2. **数值语义三级标注**,每项优化必须声明属于哪一级:
   - ★ 逐位不变(纯通路/纯调度改动);
   - △ 数学等价但浮点归约顺序改变(如 torch.compile);
   - ✗ 训练语义变更(如 rollout 数据滞后一拍)。
   **维护者已批准(2026-09-01):△ 与 ✗ 级允许实施且默认开启。批准边界是
   "模型输入语义不得改"——V18 输入协议、Actor 公共信息边界、Critic 私有
   输入必须逐位不变(编码搬迁/复算必须 bit-identical,同约束 3);训练期
   语义(浮点归约顺序、数据滞后一拍、minibatch 组织等)可以修改。△/✗ 项
   仍须实现为可关闭的配置开关,报告中给出"关/开"消融对照与开启态的数值
   分布对比。**
3. 协议契约不得触碰:Actor 公共信息边界、Critic 私有输入、V18 输入协议、
   评测机制常量(`evaluation/mechanism.py`)。
4. 现有 checkpoint 一律只读;性能测试不得写入
   `checkpoints/train_riichi_v18/ppo/` 正式目录(见测试协议的重定向要求)。
5. 每项优化独立 commit、附带对应测试、`riichi_ppo_v1/tests` 全通过;代码注释
   中文;不硬编码版本号/路径;删除代码前 `rg` 全仓引用检查。

## 二、实测基线:一个 update 的时间花在哪(必读)

稳态数据(iteration 5–150 均值,双 L20、12 worker × 32 env):

| 阶段 | 耗时 | 占比 |
|---|---|---|
| algorithm_wall_s(总计) | ≈731–760s | 100% |
| rollout_wall_s(收集) | ≈223–241s | 30% |
| update_wall_s(learner 更新) | ≈494–520s | 70% |
| sps / transitions | ≈1800 / ≈1.33M | — |

**两阶段完全串行**(`train.py:541-588`:collect → 拼接 → update → 广播权重)。
注意:`minibatches=1748` 是双 rank 求和口径,每 rank 每 update 实际 874 个
minibatch,单次 forward ≈139.6ms、backward ≈318ms。

### rollout 内部(每 worker 每 update,来自 performance.jsonl)

| 项 | 耗时 | 说明 |
|---|---|---|
| **inference/rpc_wait** | **169.4s(78%)** | 同步等待推理 RPC,`worker.py:666-682` |
| model_state_prepare | 25.8s(12%) | 编码:current_state_assembly 12.9s(Rust,持 GIL+深拷贝)+ critic_feature_encode 8.8s(纯 Python,`bridge.py:223-245`)+ 合法动作 JSON 链 3.6s |
| env_step_batch | 5.8s(2.7%) | Rust 步进,4 线程已释放 GIL,**不是瓶颈** |
| event_sync / transition_materialize / decode | 2.1 / 2.5 / 1.0s | |
| 未插桩杂项 | ≈7.4s | GRP CPU GRU 前向(22.4k 次/update)+ 事件 JSON 双重解析 |

推理 actor 侧(2 actor,`inference.py`):**7706 次 dispatch 全部由 16ms 凑批
超时触发(100% 睡满,从未触发行数上限 512)**;rows/dispatch≈349;每请求仅
~32 行;**GPU 推理期占空比仅 ≈31%**;单 asyncio 事件循环内串行执行
host_collate/H2D/forward(`inference.py:418-473`),计算期间新 RPC 只能排队。

### update 内部(每 update,每 rank)

| 项 | 耗时 | 代码 |
|---|---|---|
| **backward(含 DDP)** | **276.7s** | `learner.py:1159-1166` |
| forward | 121.9s | `learner.py:1068-1091`(已 autocast bf16) |
| **SFT reference 预计算** | **53.3s** | `learner.py:685-730`,chunk 内 collate+H2D 串行 |
| collate_h2d | 15.7s | `learner.py:385-396`,**逐字段 pageable `.to(device)`,无 pin 无 non_blocking** |
| collate_soa_gather | 45.3s | 已被 prefetch 线程重叠,0 墙钟 |
| 优化器+裁剪 | <0.7s | fused AdamW 已启用 |
| DDP allreduce | <2s | `no_sync` 已压到每 10 步 1 次 |
| 未计时 gap | ≈42s | driver 分片 IPC(~2.4GB×2 pickle,`learner_ddp.py:379-461`)、metric 累加、权重广播 |

**关键判断:模型仅 ~6M 参数,GPU 全迭代 util 70%、功率 215W/峰 290W——
backward 277s 是 kernel-launch/小算子/同步开销主导,不是算力饱和。** 具体
开销来源:
- 嵌入层 per-category/per-slot Python 循环(`dense_embedding.py:179-212,
  271-290`),一次 forward 数百个小 kernel,backward 翻倍;
- critic 装配布尔掩码索引触发 nonzero→GPU→CPU 同步(`architecture.py:560-572`);
- 每次前向构建 [B,T,T]≈18.6M 元素 mask(`architecture.py:156-188`)。

### 已排除的嫌疑(不要浪费时间)

- `find_unused_parameters=True`:仅 bootstrap 前 2 个 update 生效,
  `learner.py:600-624` 在首个非 bootstrap update 已自动重建为 False(日志
  警告只在 iteration 1 出现即证据);
- fused AdamW 已启用(`learner.py:498-519`),optimizer_step 0.12s;
- DDP 通信(<2s)、collate(已重叠)、env 步进(5.8s)、Ray 数据回传
  (result_get_s=0.002s)都不是瓶颈。

## 三、对维护者候选方向的核实结论(先纠偏)

1. **Rust 并行**:`env_step_threads=4` 是 Rust scoped 线程且已释放 GIL,
   env 步进仅 5.8s——"步进并行"不是瓶颈。真正可做的是:动作/事件链去 JSON、
   编码去深拷贝、删冗余计算、critic 特征下沉 Rust、release profile 调优。
   注意 state-machine 的 `thread::scope` 并行段是 no-op(`table.rs:23-35`),
   JSON 解析实际都在 GIL 内串行。
2. **worker×env 并发**:worker 内 4 线程只在 step_batch 生效且无关痛痒;
   真正的问题是**齐步推进**(每步一个 RPC 只带 ~32 行)+ **16ms 凑批窗 100%
   睡满** + 单 asyncio 推理 actor 串行。env 一级公民化是结构性大改,先做
   中间档(见 Tier 1/2)。另注意:12 worker 峰值 RSS ≈29GB/worker
   (≈348GB/503GB),扩 envs_per_worker 前必须先做内存瘦身。
3. **JSON→tensor**:模型输入与编码输出已经是 Rust→numpy,JSON 集中在动作
   注册/解码链(同一动作 5 重序列化转换)与事件同步通道,直接耗时约 4–10s/
   worker——是中等收益项,不是最大头。
4. **GPU 推理**:占空比 31%、批永远凑不满、窗口全额睡满——这里确实有收益。
5. **维护者未提的最大头**:**learner update 阶段占 69%**,其中 backward 277s
   是 kernel 开销主导——这是本任务最重要的方向。

## 四、优化方案清单(按优先级)

### Tier 0:零/低风险纯通路(先做,每项独立 commit)

1. **H2D 通路优化 ★**:pinned memory + non_blocking + dtype 压缩。
   `learner.py:385-396` 逐字段 pageable `.to(device)`;照抄推理侧现成实现
   `inference.py:218-246`(pinned 池+non_blocking)。同时 collate 输出的
   factor 是 int64(值全 <256,`rollout_buffer.py:318-342`)——host 侧改
   uint8/int32 传输、GPU 侧 `.long()`,每 minibatch 传输量 ~55MB→~10MB。
   预期 15.7s→3–4s。
2. **推理凑批"到齐即 flush" ★**:`inference_batch_target_workers: 6`
   (`inference.py:241-247`,-1 显式禁用了按 worker 数触发,当前 100% 批次
   睡满 14.6ms 窗口)或行数阈值。预期每 dispatch 省 10–14ms,rollout −6~10%。
   可先用 CLI `--inference-batch-wait-ms` 做 A/B 验证方向。
3. **Rust release profile ★**:全仓 Cargo.toml 均未定义 `[profile.release]`
   (默认 opt-level=3、lto=off、codegen-units=16、无 target-cpu)。加
   `lto="thin"`、`codegen-units=1`,构建时 `RUSTFLAGS="-C target-cpu=native"`
   (构建机=运行机)。Rust CPU 段(step+编码 ~19s/worker)预期提速 10–30%。
   **注意改后必须重装并验证加载的是新 .so(见第六节 maturin 坑)。**
4. **critic 装配布尔索引改算术 scatter ★**:`architecture.py:560-572` 三处
   布尔掩码索引(内部 nonzero 强制 GPU→CPU 同步),同文件 476-487 已有算术
   索引示范。写位置与写值完全相同,数值不变,forward 与 backward 排水都受益。
5. **(可选,维护者已允许)minibatch 1536→2048**:每 rank 每 epoch
   874→655 个 minibatch,kernel 开销场景直接砍 25% 调用次数。单独作为一个
   测量变体,报告标注有效 batch 变化与显存峰值。

### Tier 1:中等改造,数值不变

6. **SFT reference 预计算提速**:`learner.py:685-730`,chunk 8192 的
   collate+H2D 在 stage 内串行;交给已有的 prefetch 线程模式或放大 chunk。
   预期 53.3s→30–35s。
7. **推理 actor 事件循环解耦**:`inference.py:418-473` 在单 asyncio 协程内
   同步执行 collate/H2D/forward,阻塞事件循环。把计算移到后台线程+CUDA
   stream,使 RPC 接收/凑批与计算重叠。
8. **Worker 内双缓冲流水线**:`worker.py:632-781` 顺序执行编码→RPC 等待→
   步进;把 32 env 拆两组,组 A 等推理时编码组 B(每步可重叠 ~7ms CPU)。
   注意 `opponent_mix.enabled=false` 时每步只有单一请求,现有"先提交全部
   再等待"形同虚设(`worker.py:666-682`)。保持决策顺序与 seed 语义。
9. **Rust 编码/交互链清理**(合计 ~15–20s/worker + GC 压力大降):
   - 动作 5 重 JSON 转换收敛:`bridge.py:71-87` 的 Python json 往返 +
     `manager.rs:103-138` 的再解析,`prepare_decisions` 改收 `Vec<Action>`
     直接推导 241 维 id;decode 改用已有 `Observation.find_action(action_id)`
     (`python.rs:141-143`,红五消歧已完备);
   - 合并两处 `Vec<Observation>` 深克隆(`current_state_encoding.rs:955-988`
     与 `encoding_facts.rs:361-678`),`prepare_current_state_batch` 加
     `py.detach`;
   - 删除每 Observation 的冗余 HandEvaluator 向听计算(~540 万次/update,
     V18 编码不消费);
   - critic 特征下沉 Rust(`bridge.py:223-245`,纯 Python 8.8s;critic 需要
     四家手牌,这是每步物化 128 个 Observation 的结构性原因);
   - `_push_mjai_event` 停写 rollout 从不消费的全局 mjai_log。
10. **RPC 载荷紧凑化 ★**:`worker.py:539-554` 发 worker 级 padded int32
    (~0.5–1MB/请求 × 42K 请求),改 uint8 factor + flat/offset 紧凑格式
    (同 RolloutBuffer),object store 流量 20–40GB→~8GB,推理侧 collate
    减半。
11. **driver 侧 42s gap**:`learner_ddp.py:379-461` 的 multiprocessing Queue
    pickle(~2.4GB×2)改共享内存/一次性传递;metric 累加
    (`learner.py:1237-1302`)纳入计时并治理;权重广播与首个 reference
    precompute 重叠。
12. **Transition 物化免拷贝**:`worker.py:581-606` 每行 `.copy()` 切片
    ~17KB×1.33M≈23GB memcpy;Rust assemble 直接输出 flat+offset 变长布局。
    释放 worker RSS(~29GB/worker)后,方向 2 的扩 env 才有空间。

### Tier 2:高收益,已获维护者批准(2026-09-01)

批准范围:训练期语义可以修改;模型输入语义不得修改(边界同约束 2/3)。
以下各项实现为**可关闭的配置开关、默认开启**;交付时把开关键显式写入
`v18_ppo.yaml` 并注明批准日期(见第七节的收尾 commit)。

13. **torch.compile learner(△ 非逐位一致,已获批)**:backward 277s 的
    根本解法。SFT 已有先例(`sft/trainer.py:359-362` 的 torch_compile
    开关),PPO learner(`learner.py:480`)未编译。动态形状对策:分桶已把
    桶内 padding 压到 0.72%(`bucket_window_multiplier=8`),再把桶内长度
    pad 到 8/16 的倍数 + `dynamic=True` 限制重编译次数;在 DDP wrapper
    之下编译原始模块(SFT 的做法)。预估 forward+backward(398s)可砍
    40–60%。浮点归约顺序改变、训练轨迹会漂移——这正是获批的 △ 级影响,
    按约束 2 做消融与分布对比;编译 warmup 落在测试的第 1 个 update,
    报告重编译计数。
14. **rollout/update 流水线重叠(✗ 语义变更,已获批)**:`train.py:541-588`
    严格串行,重叠后总墙钟 ≈ max(update, rollout)(731s→约 510s,−30%,
    与 13/14 叠加后按流水线节拍收敛)。实现要点:
    - 数据滞后一拍:rollout k+1 与 update k 并行,采样权重比 update k 的
      起点旧一拍;PPO ratio 仍自洽——old_logprob 由推理 actor 在采样时
      返回,即真实采样策略,逐行成立;
    - **权重广播时机二选一**:(a) 逐轮冻结(推荐)——每轮 rollout 全程
      使用固定权重,新权重推迟到该轮结束后才广播,陈旧度均匀、推理侧无
      热切换;(b) mid-rollout refresh——推理 actor 在 rollout 中途热切换
      新权重,依赖行级 logprob 自洽,陈旧度不均,实现与验证都更重,
      不要求;
    - 显存与算力争用实测:learner(峰值 allocated ≈22GB / reserved
      ≈38GB)与推理 actor 同卡共存,全量验证必须报告两卡峰值显存;
    - 指标口径:重叠后 algorithm_wall_s ≠ rollout_wall_s +
      update_wall_s,报告给出每个 update 的流水线时间线(rollout/update
      各自起止);
    - driver 重构集中在 `train.py:541-588`;复核 `update_timeout_s` 语义。
15. **env 一级公民架构(方向 2 的完全体,在批准范围内)**:Rust
    `step_batch` 已支持部分动作 map 原地等待(`env.rs:116-144`),锁死点在
    Python 编排层:同步 for 循环、`batch_index = env_index*4 + seat_id` 与
    状态机槽位绑定(`bridge.py:36-37`)、collect 的"齐步半庄"停止条件。
    纯编排重构、不动模型输入语义,但前置依赖重:先完成 12(worker 内存
    瘦身)与 7(推理解耦)。**决策规则:13/14 落地后若关键路径仍在
    rollout 侧,实施本项;否则交付设计提案即可,报告写明判断依据。**

## 五、性能测试协议(维护者决定,优先于 AGENTS.md 默认基线)

分两级:优化过程中的**快速缩放测试**(每项优化跑一次,便宜)与收尾时的
**全量正式验证**(只跑一次)。**不使用** AGENTS.md 的默认性能基线
(target_kl=0.0/update_epochs=4/512 半庄)——本任务要测真实工作负载。

### 5.1 快速缩放测试(优化过程中)

1. **口径**:以 `v18_ppo.yaml` 为底本写一份自包含性能测试配置(建议放
   `audit/reports/v18/scripts/v18_ppo_perf_scaled.yaml`,参照
   `v17_ppo_resume.yaml` 的完整副本做法),只改以下键:
   - `resume: checkpoints/train_riichi_v18/ppo/checkpoint_00150.pt`、
     `init_model: null`(避开 iteration 1–2 的 critic bootstrap 路径,测
     稳态 learner 代码;fresh 起跑的前 2 个 update 全是 bootstrap,测不到);
   - `iterations: 152`、`total_updates: 152`(resume 后 learner 从
     checkpoint 恢复 iteration=150 续跑,正好 2 个 update。注意
     `iterations` 是**绝对总数不是增量**——CLI 的 `--iterations` 也一样,
     主循环是 `range(learner.iteration, config["iterations"])`,
     `train.py:541`);
   - `games_per_update: 1024`、`update_epochs: 1`(工作负载缩减);
   - `checkpoint_dir` 重定向到 `checkpoints/train_riichi_v18/ppo_perf/` 下
     专用子目录,`eval1v3_enabled: false`;
   - **其余一切(PPO 全部超参、num_workers=12、envs_per_worker=32、
     inference_*、learner_gpus=2、seed)与正式配置逐项相同**。
2. **为什么是 2 个 update**:第 1 个 update 吸收 Ray 启动、模型加载、
   CUDA/cudnn/编译预热;**第 2 个 update(日志 iteration=152)是唯一测量
   点**。(若该项测试涉及 torch.compile,重编译可能延入第 2 个 update,
   需同时报告编译计数。)
3. **预算**:缩放后预计单 update rollout ≈110s + update ≈150–190s ≈
   4.5–5min,整跑含启动约 10–11min。若基线的第 2 个 update 超过 5min,
   把半庄降到 768(epoch 保持 1)并在报告记录;**不要**通过减 worker/
   env/推理参数压时间——那会改变 rollout 计时结构,结果不可信。
4. **对比规则**:缩放口径数字**不得**与 r5 日志(2048×2)直接比较,只在
   同口径内做 before/after、按比例外推。固定开销(driver 侧 gap 的固定
   部分、每 update 一次性成本)在缩放口径占比略高,报告需标注。
5. **节奏**:开工先测一次缩放基线;此后**每项优化只跑一次**;结果异常
   (提升 <5% 或方向相反)时复测基线排除机器状态漂移;每个 Tier 收尾
   复测一次基线。跑完 `ray stop --force`。
6. **命令模板**(iterations 已写进配置,无需 CLI 覆盖):
   ```
   env -C /mnt/disk1/hubowen/zenith RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0,1 \
     PYTHONUNBUFFERED=1 python -m riichi_ppo_v1.training.train \
     --config audit/reports/v18/scripts/v18_ppo_perf_scaled.yaml \
     --device cuda --learner-gpus 2 \
     2>&1 | tee logs/v18/perf_scaled_<优化名>_<日期>.log
   ```

### 5.2 全量正式验证(所有优化完成后,只跑一次)

用正式口径(games_per_update=2048、update_epochs=2)做端到端验证:再写
一份自包含 resume 配置(同样从 checkpoint_00150 续、`iterations: 152`、
输出重定向、eval 关闭),跑 2 个 update,以第 2 个 update(iteration=152)
对照 r5 稳态数字(731/223/505)。这一跑是报告的**头条数字**。

- **按交付态运行**:Tier 2 开关默认开启(含 torch.compile 与流水线
  重叠);2 个 update 的窗口足够让重叠在第 2 个 update 生效、让编译
  warmup 留在第 1 个。
- 重叠启用后 algorithm_wall_s 不再等于 rollout_wall_s 与 update_wall_s
  之和,报告以流水线时间线(rollout/update 各自起止)呈现;两卡显存
  峰值(learner 与推理 actor 同卡共存)必须记录。
- 收尾 sanity:entropy/approx_kl/value_loss 与 r5 尾部同量级、GPU 峰值
  显存无恶化。必要时可延长到 3–5 个 update 看稳定性,但 2 个是底线。

### 5.3 必报指标与数值验证(两级测试通用)

- **必报指标**(前后对比表):日志行的 algorithm_wall_s、rollout_wall_s、
  env_step_s、update_wall_s、update_forward_s、sps、
  worker_transitions_per_s;performance.jsonl 的
  `ppo/timing/update/*`(model_forward、backward、reference_precompute、
  collate_h2d、collate_soa_gather、optimizer_step)、
  `rollout/inference_actor/inference/*`(dispatches、dispatch_timeout、
  dispatch_rows_mean、queue_wait、full_forward)、GPU 峰值显存
  allocated/reserved。
- **数值验证**:★ 项尽量做到同口径同种子第 2 个 update 后关键指标
  (loss、entropy、approx_kl、value_loss)逐位或近逐位一致;△/✗ 项
  (已获批、默认开启)对比分布量级——允许轨迹漂移,不允许系统性偏移
  (对照 r5 尾部同 update 数的指标带);每个 Tier 2 开关做一次"关/开"
  消融(缩放口径即可,单独归因各自的收益与数值影响)。
- 交付前 `riichi_ppo_v1/tests` 全部通过;测试产生的临时 checkpoint/
  中间产物清理,perf 日志、测试配置与报告留档。

## 六、工程与运维注意(含关键坑)

- **maturin 坑(重要)**:`riichi` 模块的 editable 安装指向已不存在的
  `file:///mnt/disk1/hubowen/zenith/riichi`,当前实际加载的是 site-packages
  里的旧 `.so`。**凡改 `RiichiEnv/` 或 `riichienv-state-machine/`,必须在
  Mahjong-AI 环境重新 maturin develop/install,并用
  `python -c "import riichi, riichienv; print(riichi.__file__)"` 之类方式
  确认加载的是新编译产物**——否则测的是旧代码,结论全部作废。
- 所有 Python 命令用 Conda 环境 `Mahjong-AI`;`CUDA_DEVICE=0,1`
  (映射物理 GPU 0/1);Ray 测试结束 `ray stop --force`。
- 日志全部写入 `logs/v18/`;测试配置自包含,不得 overlay 继承。
- 机器:48 核 / 503GB RAM / 4× L20 46GB(正式训练用其中 2 张)。
- 不允许在长训练运行期间重构/移动模块(AGENTS.md 教训);本任务每次只跑
  2 个 update 的短测试,不存在此约束冲突,但同理不要在跑测试时并行改动
  被测代码。

## 七、交付物

1. 优化代码 + 对应测试(每项独立 commit、可独立回滚);
2. 性能测试自包含配置(缩放口径与全量验证两份)与运行命令留档于
   `audit/reports/v18/scripts/`;
3. `audit/reports/v18/report/V18_PPO训练提速优化报告.md`:每项优化的
   before/after 分段计时对比表(缩放口径)、**5.2 全量正式验证的最终数字
   (2048×2,交付态,对照 r5 稳态 731/223/505)**、总收益、语义影响标注
   (★/△/✗)、未做项与后续建议;
4. `audit/reports/v18/report/PROGRESS.md` 追加记录;
5. Tier 2 各项以可关闭开关交付、**默认开启**(维护者 2026-09-01 批准,
   边界见约束 2);报告单独章节记录每项的语义影响、"关/开"消融结果与
   关闭方法;
6. 全部验证通过后的最后一笔 commit:把 Tier 2 开关键显式写入
   `v18_ppo.yaml`(自包含、注明维护者批准日期)——这是任务中唯一允许
   修改该文件的时机,此前一律不动。

## 八、建议执行顺序与预期

先建立缩放口径基线(2 updates,测第 2 个)→ Tier 0 逐项(1→4,每项一次
缩放测试并记录)→ 阶段性汇总 → Tier 1 按性价比(建议 6→7→9→10→8→11→12)
→ Tier 2 实施(建议 13→14,13 可在模型侧代码改动(4/9)落定后提前;15 按
其决策规则定实施或只交提案)→ 5.2 全量正式验证(2048×2,交付态)定稿
头条数字 → 收尾 commit 写入 `v18_ppo.yaml`。

预期:仅 Tier 0+1(不动任何训练语义)稳态约 731s → **550–600s**;Tier 2
已获批——叠加 torch.compile 与流水线重叠(可组合 minibatch=2048)后目标
**350–450s**,其中重叠生效后总墙钟 ≈ max(rollout, update)。(预期以
全量口径为准;缩放口径只用于快速迭代与相对比较。)
