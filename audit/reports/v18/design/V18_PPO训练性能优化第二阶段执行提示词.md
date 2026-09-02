# V18 PPO 训练性能优化(第二阶段)执行提示词

> 本文档是交给执行会话的任务提示词。任务:在第一阶段(2026-09-01/02,每 update
> 746.0s → 454.6s,−39.1%)基础上继续压缩 V18 PPO 稳态墙钟。当前关键路径在
> **learner update 侧(451.3s)**,rollout 生成(≈180s)已被流水线重叠完全吸收。
> 维护者批准边界与第一阶段相同(2026-09-01):**模型输入语义不得改;训练期
> 语义(浮点归约顺序、数据滞后一拍、minibatch 组织、调度/编排)允许修改**;
> △/✗ 级实现为可关闭开关并给出消融。
> 以下所有耗时数据来自交付态全量验证(2048×2,commit `40ed96b` 时期,
> iteration=152)与缩放口径各阶段运行,详见
> `audit/reports/v18/report/V18_PPO训练提速优化报告.md`(第一阶段报告,必读)。
> 开始前请先读:`AGENTS.md`(最高治理文档)、本文、第一阶段报告,以及
> `audit/reports/v18/scripts/` 下已提交的 5 份性能测试配置。

---

## 〇、当前状态(必读)

- 分支 `V16`,起点 HEAD `d4b9e86`(含第一阶段 14 个 commit 与 2.14 簿记修正
  `788dd01`:重叠编排已保证每份 rollout 恰好被消费一次)。
- Tier 2 开关 `torch_compile: true`、`rollout_update_overlap: true` 已显式写入
  `riichi_ppo_v1/configs/v18_ppo.yaml`(交付态默认开启);新增开关同规范。
- 第一阶段实施的存量优化(H2D pinned+uint8、推理凑批 quorum、Rust release
  profile、critic 算术 scatter、reference 预计算流水线、推理事件循环解耦、
  Rust GIL 释放 + 全局 mjai_log 停写、RPC uint8、torch.compile、重叠)全部
  在代码中,不要回退。

### 各阶段时延(全量口径 2048×2,交付态,iteration=152)

| 阶段 | 时延 | 说明 |
|---|---|---|
| **每 update 总墙钟** | **454.6s** | ≈ max(update, rollout 生成) |
| update:model_forward | 99.3s | torch.compile 已生效 |
| update:backward(含 DDP) | 162.2s | torch.compile 已生效 |
| update:reference_precompute | 75.4s | **参考模型未编译**;与并行推理 actor 争 GPU |
| update:collate_h2d | 3.1s | 第一阶段已优化 |
| update:optimizer_step | ≈0.1s | fused AdamW |
| update:未插计 gap | **≈111s** | **无任何计时归属**,本阶段首要目标 |
| rollout 生成(纯) | ≈180s | 被重叠完全吸收,优化它零收益 |
| learner 峰值显存 allocated/reserved | 21.9GB / 38.4GB | 46GB 卡,余量充足 |

缩放口径(1024×1)对照:交付态 159.4s;无重叠消融 rollout 88.6s + update
115.3s(fwd 22.1 / bwd 37.4 / ref ≈25.5);机器噪声带 ±4s。

### 关键判断

1. 关键路径 = update 451.3s,其中**未插计 ~111s(≈25%)完全没有计时归属**,
   是最大的黑盒;**backward 162.2s 是最大插计项**;**reference 75.4s 未编译**
   是最便宜的确定收益。
2. 未插计段的疑似构成(第一阶段观察):collate 预取线程被 12 个 rollout
   worker(24 核编码)抢 CPU 后变慢、主线程在预取队列上干等;driver 侧
   shard select 拷贝(~2.5GB×2)+ multiprocessing pickle + 全量 buffer 指标
   聚合 + 权重广播同样与 rollout 编码争 CPU/内存带宽。
3. rollout 侧不要再投入(重叠吸收,零墙钟收益)。

---

## 一、硬约束(不可违反)

1. **训练超参数一律不得改动**:gamma、gae_lambda、ppo_clip、
   gradient_accumulation_steps=10、全部学习率/熵系数/KL/SFT-KL/value 系数、
   target_kl、seed;正式口径 `games_per_update=2048`、`update_epochs=2` 不得改。
   **唯一例外**:`minibatch_size` 最多 2048(须在报告披露等效有效 batch
   1536×10×2=30720 → 2048×10×2=40960 的变化与显存峰值)。
2. **模型输入语义逐位不变**:V18 输入协议、Actor 公共信息边界、Critic 私有
   输入、评测机制常量(`evaluation/mechanism.py`)不得触碰;新增 padding/
   静态 shape 不得改变任何有效 token 的数值(见项 E 的逐位校验要求)。
3. **数值语义三级标注**(★ 逐位不变 / △ 数学等价但浮点归约顺序改变 /
   ✗ 训练语义变更)。维护者已批准 △/✗ 级实施;每项实现为**可关闭开关**,
   给出关/开消融;若某项默认开启后不稳(数值异常/显存/死锁),允许以默认
   关闭交付并在报告注明理由,由维护者复核。
4. `v18_ppo.yaml` 在收尾 commit 前一律不动(与第一阶段相同,收尾时把本任务
   新增开关键显式写入并注明批准日期);性能测试一律用独立自包含配置,其中
   只允许改工作负载/编排键与本任务新增的性能开关键;PPO 超参与 rollout/
   推理拓扑必须与正式配置逐项相同。
5. **现有 checkpoint 只读**;性能测试输出重定向到
   `checkpoints/train_riichi_v18/ppo_perf/<专用子目录>/`,`eval1v3_enabled: false`。
6. 每项优化独立 commit、附带对应测试、`riichi_ppo_v1/tests` 全绿;代码注释
   中文;不硬编码版本号/路径/种子;删除代码前 `rg` 全仓引用检查。
7. **证据留存(第一阶段教训,硬性要求)**:每次成功验证运行结束后,**先**把
   `performance.jsonl` 中被测迭代行(以及关键日志行)复制到
   `audit/reports/v18/report/perf_evidence/`(随报告入库留档),**再**清理
   `ppo_perf/` 临时目录;**禁止用重复运行覆盖成功日志**——重复确认一律用新
   日志文件名;成功运行的原始 jsonl/日志在报告引用前不得删除。

---

## 二、优化项清单(按建议执行顺序)

### A0. 缩放基线复测(开工第一步,必做)

用现有 `audit/reports/v18/scripts/v18_ppo_perf_scaled.yaml` 在当前 HEAD 上
重跑一次缩放基线(2 updates,测第 2 个),作为本会话所有 before/after 的
机器状态基准(第一阶段数字采于 2026-09-01,机器状态可能漂移)。预期 ≈159s
(±噪声);若偏离 >10%,先排查机器负载再开工。

### A. minibatch 2048 编译态重测(便宜,先做)

- 背景:第一阶段的负收益结论(−3.9s,+7GB 显存)测于 **eager 模式**;
  torch.compile 已摊薄 launch 开销,经济学变了,需要重测。
- 做法:配置已存在(`v18_ppo_perf_scaled_mb2048.yaml`,需确认与当前 HEAD
  兼容),跑一次缩放口径,报告 fwd/bwd/显存峰值与有效 batch 披露。
- 验收:若 update 墙钟下降 ≥5% 且显存峰值 allocated ≤35GB,可纳入交付态
  (单独开关或直接改 scaled 配置对比);否则记录不采纳。
- 语义:✗(有效 batch 变化,须披露);纯测量,无代码改动。

### B. update 未插计 ~111s:先插计,后治理(预期 −40~80s,风险低)

**B1 插计(★ 纯计时)**,把以下位置全部纳入 StageProfiler/计时,跑一次缩放
口径拿到分布(全量口径抽查一次即可):
- `learner.py::_prefetch_get`:主线程在预取队列上的**等待时长**(新增
  `update/collate_wait` 计时;这是「collate 饥饿」假设的直接证据);
- `learner.py` 预取线程:`_prefetch_collate_worker` 的线程 busy 时长与队列
  满阻塞时长(现有 `collate_soa_gather` 只记线程内 collate 时间);
- `learner_ddp.py::update`:`learner_shard_indices`+`transitions.select`(每
  rank ~2.5GB gather)、命令队列 `put`(pickle 传输)、`_recv`(等待子进程);
- `train.py`:权重广播 `update_weights` 的时长(当前包含在 update_wall 内,
  无独立计时);
- `learner_ddp.py::aggregate_learner_metrics`:host 侧全量数学
  (`rollout_target_metrics`/`rollout_update_targets`/`discounted_empirical_returns`
  在 driver 上对全量 buffer 重复计算了多次)。

**B2 治理**(按插计结果择施,均为 ★ 或低风险编排改动):
- collate 预取队列加深(当前 `maxsize=2`,`learner.py` 两处:训练预取与
  reference 生产者)或双 collate 线程;队列深度做成配置键(如
  `update_collate_prefetch_depth`),默认值经消融确定;
- `aggregate_learner_metrics` 的全量数学复用 shard 准备阶段已算出的
  advantages/returns(注意:必须**复用同一结果**,不得重排浮点求和顺序,
  否则从 ★ 降级为 △,需标注);
- shard IPC 改共享内存(`/dev/shm` 或 `multiprocessing.shared_memory` 传递
  SoA 数组,只传元数据);实现时保持「每 rank 数据严格一致」与
  `update_timeout_s` 语义;
- **CPU 亲和性隔离**:给 driver 与两个 DDP 子进程
  (`learner_ddp.py::_learner_worker`,经 `os.sched_setaffinity`)预留专属核
  (如 48 核机器的高位段),与 12 个 rollout worker 的 24 核隔离;做成配置
  键(如 `learner_cpu_affinity`),默认不设置(行为不变),消融后定默认。

**验收**:缩放口径 update 墙钟相对 A0 基线下降,且未插计 gap 显著缩小;逐项
记录每个子改动的单独贡献。

### C. reference_precompute 编译 + chunk 调优(预期 −20~30s,风险低)

- `PPOLearner` 的 `self.reference_model` 未编译(2.13 只编译了训练模型):
  同样 `torch.compile` 包装(注意 `load_state_dict`/`.eval()` 与包装顺序,
  权重读写沿用 `_state_dict_source` 的思路);开关
  `torch_compile_reference`(默认跟随 `torch_compile`)。
- chunk 大小 `update_reference_precompute_batch_size`(默认 8192)在 16384 /
  32768 上消融(显存余量 46GB 足够),选优。
- 注意动态 shape 重编译:与训练模型相同的 shared_capacity guard 会触发
  warmup 重编译(第一阶段已见 recompile_limit(8),仅热身期);若做了项 E
  的桶静态 shape 则一并消失。
- 语义 △(归约顺序);显式写入报告。

### D. CUDA graphs + 桶静态 shape(期望 −80~130s,风险最高,第二阶段攻关)

- 原理:compile 默认模式后剩余大头仍是小算子/launch 开销(GPU 功率 215W/
  峰值 290W 未饱和);`mode="reduce-overhead"`(CUDA graphs)可基本消灭
  launch 开销,前提是**静态 shape**。
- 实现:把 `bucketed_minibatches`/`collate` 改为**按桶固定容量 pad**
  (桶窗口 `bucket_window_multiplier=8` 内按长度排序,桶容量=桶内最大长度;
  新增的 padding 列全是 0/无效区,现有 mask/`kind_row_plan`/shared_capacity
  逻辑对超长 padding 列已有正确语义)。**必须逐位校验**:同一批数据在
  「batch 内 max pad」与「桶容量 pad」两种 collate 下,模型输出(logits/
  value)应逐位一致或仅 △ 级差异(不同 kernel tiling 的低比特差),写单测
  断言并明确标注等级。
- 在此之上对训练模型启用 `torch.compile(mode="reduce-overhead")`;开关
  `torch_compile_mode`(`""`=默认模式 | `reduce-overhead`),消融定默认。
- **已知风险,必须逐一验证**:DDP 与 CUDA graphs 的兼容性(static-buffer
  约束)、`no_sync` 梯度累积交互、显存峰值(graph 池 + 38.4GB 现状)、
  eager 回退路径;先单卡实验再上双卡 DDP;数值与显存任一不稳则降级为
  默认关闭交付并注明。
- 注意:`no_sync`/累积组边界/`planned_minibatches` 对齐逻辑(两 rank 一致)
  是 DDP 正确性的命门,任何编排改动都要复跑
  `tests/unit/test_learner_ddp.py` 与双卡冒烟。

### 明确不做

- rollout 侧一切优化(重叠吸收,零收益);
- 训练超参、数值路径降精度(bf16→fp8/int8 属于数值语义级变更,未获批;
  当前瓶颈也不在计算精度);
- 1v3 评测机制常量与协议契约。

---

## 三、性能测试协议(与第一阶段相同,优先于 AGENTS.md 默认基线)

### 3.1 快速缩放测试(每项优化跑一次)

- 配置:`audit/reports/v18/scripts/v18_ppo_perf_scaled.yaml`(已提交,自包含;
  resume `checkpoint_00150.pt`、`iterations: 152`、`games_per_update: 1024`、
  `update_epochs: 1`、输出重定向 `ppo_perf/scaled`、评测关闭;其余键与正式
  配置逐项相同)。**不要另起口径**;新开关的消融配置以它为底本另存独立
  yaml(参照 `_ablation_no_compile.yaml` 的做法)。
- 口径:第 1 个 update 吸收启动/编译热身,**第 2 个 update(iteration=152)
  是唯一测量点**;涉及编译/重编译的项需同时报告编译与重编译计数
  (dynamo 警告行)。
- 命令模板(**必须用 Mahjong-AI 环境的显式 python 路径**,base 环境无 torch):

  ```
  env -C /mnt/disk1/hubowen/zenith RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0,1 \
    PYTHONUNBUFFERED=1 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python \
    -m riichi_ppo_v1.training.train \
    --config audit/reports/v18/scripts/<配置名>.yaml --device cuda --learner-gpus 2 \
    2>&1 | tee logs/v18/perf_scaled_<项名>_<日期>.log
  ```

- 预算:整跑 ≈9–11 分钟;对比只做同口径 before/after(对照 A0 复测基线),
  噪声带 ±4s,提升 <5% 或方向相反时复测基线排除漂移;每轮跑完
  `ray stop --force` 并确认 `nvidia-smi` 无残留显存。

### 3.2 全量正式验证(收尾只跑一次)

- 配置:`audit/reports/v18/scripts/v18_ppo_perf_fullscale_verify.yaml`
  (2048×2,resume checkpoint_00150,iterations 152,交付态开关全开)。
- 头条数字 = iteration=152 的 `algorithm_wall_s`,对照第一阶段交付态
  **454.6s**(而非 r5 的 746.0s);整跑 ≈20–25 分钟(iteration=151 含编译
  热身 ≈900s 量级,属预期)。
- 必查:两卡显存峰值(learner 与推理 actor 同卡共存)、
  entropy/approx_kl/value_loss/clipfrac 与第一阶段交付态同量级
  (0.174 / 1.15e-3 / 0.271 / 0.0115 附近)、重编译仅出现在热身 update。
- 数值验证:★ 项尽量逐位或近逐位一致;△/✗ 项对照分布量级,不允许系统性
  偏移。

### 3.3 必报指标(与第一阶段一致)

日志行:algorithm_wall_s、rollout_wall_s、update_wall_s、update_forward_s、
sps、worker_transitions_per_s;performance.jsonl 的 `ppo/timing/update/*`
(新增的插计段一并上报)、`rollout/inference_actor/inference/*`、GPU 峰值
显存 allocated/reserved;新增开关的编译/重编译计数。

### 3.4 证据留存(硬性要求,见一.7)

每轮成功运行:`tee` 的日志留在 `logs/v18/`(命名
`perf_scaled_<项名>_<日期>.log` / `perf_fullscale_<日期>.log`);
**performance.jsonl 的被测迭代行复制到
`audit/reports/v18/report/perf_evidence/<运行名>.jsonl` 入库留档后**,
才允许删除 `ppo_perf/` 临时目录。

---

## 四、工程与运维注意

- **maturin 坑**:本任务预计**不涉及 Rust**;若确需改 `RiichiEnv/`,必须
  `RUSTFLAGS="-C target-cpu=native" bash RiichiEnv/scripts/install_conda_extension.sh`
  重装,并用 `riichi.__file__` + site-packages `.so` mtime 验证加载新产物,
  否则结论作废(第一阶段有完整操作记录)。
- 所有 Python 命令用 Conda 环境 `Mahjong-AI`(建议显式
  `/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python`);`CUDA_DEVICE=0,1`
  映射物理 GPU 0/1;后台有一个无关的 `riichi-lab-bot ranked` 进程常驻
  cuda:0(~420MB,只读 checkpoint),属机器状态背景,记录即可。
- 单测在 CPU 上构造 `PPOLearner` 时必须显式 `torch_compile=False`
  (CPU 编译极慢,第一阶段 resume 测试曾 91s);涉及 DDP/编排的改动必须
  复跑 `tests/unit/test_learner_ddp.py` 与 integration 全量。
- 长测试期间禁止重构/移动模块(AGENTS.md 教训);短测试同理不要并行改动
  被测代码。
- 冒烟/临时产物用完即删;日志只写 `logs/v18/`。

---

## 五、交付物

1. 优化代码 + 对应测试(每项独立 commit、可独立回滚;commit message 中文,
   沿用 `perf(v18-ppo):` / `fix(v18-ppo):` / `test(v18-ppo):` / `docs(v18-ppo):`
   前缀风格);
2. 新增消融配置留档于 `audit/reports/v18/scripts/`;证据文件留档于
   `audit/reports/v18/report/perf_evidence/`;
3. `audit/reports/v18/report/V18_PPO训练性能优化第二阶段报告.md`:每项的
   before/after 分段计时(缩放口径)、全量正式验证头条数字(2048×2 交付态,
   对照第一阶段 454.6s)、未插计 gap 的插计分布表、语义影响标注(★/△/✗)、
   每个开关的关/开消融与关闭方法、未做项与后续建议;
4. `audit/reports/v18/report/PROGRESS.md` 追加记录;
5. 收尾 commit:把本任务新开关键显式写入 `v18_ppo.yaml`(自包含、注明维护者
   批准日期)——这是本任务唯一允许修改该文件的时机。

## 六、建议执行顺序与预期

A0 基线复测 → A(mb2048 编译态重测)→ B1 插计 → C(reference 编译+chunk)
→ B2 治理(按插计结果)→ 每个 Tier 收尾复测一次基线 → E(CUDA graphs,
若 B/C 后 update 仍显著高于 rollout 生成段则值得投入;先单卡验证再 DDP)
→ 全量正式验证定稿头条数字 → 报告/PROGRESS → 收尾 commit。

预期:B+C+D(不含 E)把全量口径 454.6s 压到 **≈350–380s**;E 成功再压到
**≈300–330s**。(预期以全量口径为准;缩放口径只用于快速迭代与相对比较。)
