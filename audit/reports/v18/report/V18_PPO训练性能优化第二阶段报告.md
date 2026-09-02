# V18 PPO 训练性能优化第二阶段报告

**日期**:2026-09-02
**执行依据**:`audit/reports/v18/design/V18_PPO训练性能优化第二阶段执行提示词.md`(维护者批准边界与第一阶段相同,2026-09-01 延续;△/✗ 级训练期语义变更获准,实现为可关闭开关并给消融)
**起点**:分支 `V16`,HEAD `cb1bdfc`(第一阶段 14 个 commit + 2.14 簿记修正 `788dd01`)
**基线对照**:第一阶段交付态全量验证 2048×2 `algorithm_wall_s = 454.6s`(iteration=152,commit `40ed96b` 时期)

---

## 一、总结论(TL;DR)

| 口径 | 优化前(第一阶段交付态) | 优化后(第二阶段交付态) | 收益 |
|---|---|---|---|
| **全量正式验证(2048×2,交付态)** | **454.6s / update** | **325.8s / update** | **−128.8s(−28.3%)** |
| 缩放口径(1024×1,迭代过程) | 147.5s(A0 复测) | 98.6s(B2 shm 档) | −33.1% |
| sps(全量) | 3144 | 4461 | +41.9% |

全量口径分项对照(iteration=152,第二阶段交付态 vs 第一阶段交付态):

| 指标 | 第一阶段 | 第二阶段 | 说明 |
|---|---|---|---|
| algorithm_wall_s | 454.6 | **325.8** | rollout 与 update 重叠,总墙钟 ≈ max(update, rollout) |
| update_wall_s | 451.3 | 322.6 | 其中 fwd 99.3→78.3、bwd 162.2→106.4、ref 75.4→23.1 |
| reference_precompute | 75.4 | **23.1** | reference 编译 + 更小占比 |
| learner 峰值显存 allocated/reserved | 21.9GB / 38.4GB | **28.1GB / 31.5GB** | allocated 因 mb2048 上升,reserved 下降;GPU0 总量峰值 34.1GB(含推理 actor),46GB 卡余量充足 |
| 数值 sanity(entropy/approx_kl/value_loss/clipfrac) | 0.17416 / 1.15e-3 / 0.2707 / 0.0115 | **0.17428 / 0.00093 / 0.2775 / 0.0094** | 同量级、无系统性偏移 |
| 重编译 | 仅 warmup(151) | 仅 warmup(151) | 稳态零重编译 |

> 注:第一阶段交付态(454.6s)含 A0 之前的所有 Tier 0/1/2 优化;本阶段在其上继续叠加 A(mb2048)、B1(插计,纯测量)、C(reference 编译)、B2(shard 共享内存),并评估 D(CUDA graphs)后决定不投入。

---

## 二、实施项明细(每项独立 commit、附带测试、独立回滚)

### A0. 缩放基线复测(必做,开工第一步)

用已提交的 `v18_ppo_perf_scaled.yaml`(1536×1)在当前 HEAD 复测:
**iteration=152 `algorithm_wall_s = 147.5s`**(第一阶段 159.4s,偏差 −7.5%,在 ±10% 容差内,机器状态良好且略快)。后续所有 before/after 以本会话当日重测为准。
证据:`logs/v18/perf_scaled_a0_baseline_20260902.log` + `perf_evidence/perf_scaled_a0_baseline_20260902.jsonl`。

### A. minibatch 2048 编译态重测(✗ 有效 batch 变化,须披露)→ **采纳**

- 背景:第一阶段在 eager 模式下测得 mb2048 负收益(−3.9s,+7GB);torch.compile 摊薄 launch 开销后经济学变了,需重测。
- 结果(缩放口径 2048×1,iteration=152,对照 A0 1536×1):

  | 指标 | mb1536(A0) | mb2048 | 变化 |
  |---|---|---|---|
  | algorithm_wall_s | 147.5 | **117.0** | −30.6s(−20.8%) |
  | update_wall_s | 145.8 | 115.3 | −30.5s |
  | model_forward | 29.2 | 19.8 | −32% |
  | backward | 56.0 | 25.3 | −55% |
  | reference_precompute | 28.3 | 28.3 | 持平 |
  | 显存峰值 allocated | — | **27.9GB** | ≤35GB 验收线通过 |

- 有效 batch 披露:**1536×10×2=30720 → 2048×10×2=40960**(维护者批准边界内唯一超参例外)。显存峰值 allocated 27.9GB(reserved 29.1GB),46GB 卡余量充足。
- 语义 ✗(有效 batch 变化),纯测量、无代码改动;消融配置留档 `v18_ppo_perf_scaled_mb2048.yaml`。
- **采纳为交付态**:后续各项对照均以 mb2048 为新基准。

### B1. update 未插计段插计(★ 纯计时,commit `880754c`)

按提示词把五处未插计段全部纳入 StageProfiler/计时:
- `learner.py`:`update/collate_wait`(主线程预取等待,每 minibatch)、`update/collate_put_block`(预取线程 put 反压,整 update 汇总)、`update/learner_wall`(rank 侧 update 总墙钟,单条);
- `learner_ddp.py`:`update/shard_select_s`(shard 索引+select gather+全量 advantage/return 数学)、`update/shard_put_s`(命令队列入队)、`update/learner_recv_wait_s`(等待两 rank 结果)、`update/aggregate_metrics_s`(全量指标汇总);
- `train.py`:`update/weights_broadcast_s`(权重广播,原埋在 update_wall 内)。

**插计分布(缩放口径 mb2048,iteration=152,B1 运行)**:

| 段 | 时延 | 归属/解读 |
|---|---|---|
| model_forward | 23.8s | GPU(已编译) |
| backward | 38.7s | GPU(当日机器负载偏高,见机器噪声说明) |
| reference_precompute | 28.3s | GPU(编译前) |
| collate_h2d + 其他插计 | ~2.0s | — |
| **collate_wait(主线程预取等待)** | **0.21s(mean 0.6ms/批)** | **「collate 饥饿」假设被证伪** |
| **collate_put_block(线程反压)** | **78.6s** | 预取线程因队列满而等待=供给远快于 GPU 消费,管线健康 |
| shard_select_s | 4.0s | driver,全量 gather 拷贝 |
| shard_put_s | 3.4s | driver,pickle 入队 |
| learner_recv_wait_s | 120.9s | = shard 传输尾差(6.8s)+ learner_wall(114.2s) |
| aggregate_metrics_s | 0.12s | driver(向量化后极小) |
| weights_broadcast_s | 0.43s | driver |
| learner_wall − Σ消费侧插计 | ≈21.8s | rank 侧 GPU 排水/同步点(异步 enqueue 计时不捕获真实 GPU 执行) |

**结论(决定 B2 取舍)**:
1. collate 饥饿不成立(collate_wait≈0),「队列加深/双 collate 线程/CPU 亲和性」**无收益,不做**(put 反压 78.6s 说明供给充足);
2. `aggregate_learner_metrics` 的全量数学复用仅 0.12s,**不值得做**;
3. driver 侧可治理项 = shard pickle+管道传输 ≈10s(**B2 目标**);
4. rank 侧残余 ≈21.8s 属 GPU 排水(同步点),压缩方向是 GPU 工作本身(**C/D 目标**)。

### C. reference 前向编译 `torch_compile_reference`(△,默认跟随 torch_compile;commits `2056388` `7bb4641` `b2d5641`)→ **采纳**

- 实现:`forward_actor` 扩为 policy-only 唯一消费入口并作为**独立 code object** 编译(与训练模型 forward 的 dynamo 缓存槽互不挤占);reference 模块本体不包装,state_dict 键与 eager 一致,checkpoint 契约不变。新配置键 `torch_compile_reference` 默认跟随 `torch_compile`。
- **关键坑(记录证据)**:torch 2.7 inductor 无法 lower `empty(pin_memory=True)`,参数无梯度(冻结模型)时编译必现 NotImplementedError。首版用 `torch._dynamo.disable` 隔离行表 pinned H2D,但边界切出的 resume 帧与训练编译共享 dynamo 缓存槽,把训练图挤出缓存回退 eager(fwd 19.8→34.8s、bwd 25.3→51.8s 实测回退)。修正为:调用方把行表逐类上传为 CUDA 张量,token_embedding 合并走纯 GPU cat 分支(numpy 分支原样保留给训练路径),训练编译图结构与第一阶段一致。同窗口对照:

  | 指标 | eager reference | 编译 reference |
  |---|---|---|
  | reference_precompute | 28.5s | **11.8s(−16.7s,−58%)** |
  | algorithm_wall_s | 122.2s | **114.4s(−7.8s,−6.4%)** |

  fwd/bwd 差异在当日机器噪声带内(B1 零改动运行 bwd 亦达 38.7s,见下「机器噪声」说明)。
- chunk 消融(`update_reference_precompute_batch_size`):16384 档 reference_precompute 12.5s vs 8192 档 11.8s(噪声内,编译态下大批量无额外收益)→ **维持默认 8192**;32768 档在真实拓扑(learner 与推理 actor 同卡)下编译图 5.63GiB 中间缓冲 OOM(37.2GB 处),不采纳,配置留档 `v18_ppo_perf_scaled_refchunk32768.yaml` 供更大显存环境复测。
- 语义 △(bf16 内核选择随图形状变化;编译态 vs eager logits 单测:−inf 位逐位一致,有限位 ≤4e-3,与既有 B2 预计算记录同量级);SFT-KL 偏差按 `sft_kl_coef ≤ 0.0025` 有界(≤1.4e-7 量级,沿用第一阶段 T2 判定)。

### B2. shard 共享内存 IPC `learner_shard_transport`(★ 数组逐位一致,默认 `shm`;commits `db2a419` `f66516f`)→ **采纳**

- 按 B1 插计结果择施(唯一被证据支持的可治理项):driver 把 select 分片的 SoA 数组写入 `/dev/shm`(每 rank 一块,select gather 之外唯一一次拷贝),命令队列只传字段布局元数据;learner 以 `np.ndarray(buffer=...)` 零拷贝视图重建 RolloutBuffer,替代 pickle 路径的「pickle+管道 memcpy+unpickle」三次整块拷贝(2.5GB/分片)。
- 生命周期:driver `close()+unlink()`(finally,含异常路径);worker 仅持映射,update 结束即 close;`update_timeout_s` 与错误传播语义不变。启动时 `sweep_stale_shard_blocks()` 清理异常崩溃遗留的 tmpfs 块(存活 pid 的块绝不动)。
- 缩放口径(iteration=152,shm vs 同日 pickle 档):

  | 指标 | pickle | shm |
  |---|---|---|
  | algorithm_wall_s | 114.4 | **98.6(−15.8s)** |
  | update_wall_s | 112.7 | **96.9(−15.7s)** |
  | shard_put_s | 3.54 | **0.05** |
  | 传输尾差(recv_wait−learner_wall) | 6.81 | **0.04** |
  | shard_select_s(含 shm 写入) | 3.94 | 7.69(+3.75,一次 memcpy 代价) |
  | 显存峰值 | — | 28.0GB(持平) |

  driver 侧确定性节省 ≈6.5s(put 3.5 + 尾差 6.8 − shm 写 3.8),其余为机器噪声内的有利漂移。
- 每 rank 数据严格一致:单测 writer→view 往返与 select 分片**逐字段逐位一致**;替身 worker 端到端 shm/pickle 行数一致。
- 关/开消融:置 `learner_shard_transport: pickle` 回退历史路径(消融配置 `v18_ppo_perf_scaled_ablation_shard_pickle.yaml`;显式 shm 确认档 `v18_ppo_perf_scaled_shm.yaml`)。

### D. CUDA graphs + 桶静态 shape → **不投入**(证据留档)

按提示词「先单卡验证再 DDP」的门槛做单卡探针(静态 shape B=2048×T=144,真实模型 fwd+bwd):

- 默认编译 373ms vs eager 783ms(2.1×,与生产一致);
- **`mode="reduce-overhead"`(CUDA graphs)直接报错** `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`,位置在 `architecture.py` 的 `torch_dynamo_resume_in_forward_at_544/600`(模型 forward 的 resume 帧结构);即使对输出做 `.clone()` workaround 仍失败——属**结构性不兼容**,需要重构 forward 的图断点结构,且 DDP + no_sync + cudagraph 的已知交互风险更高。
- 桶静态 shape 的唯一动机是 CUDA graphs 的前提;默认编译模式稳态**零重编译**(仅 warmup 8 槽),静态化无稳态收益,反而增加 padding 计算量。
- 关键路径判断:A/C/B2 落地后缩放 update ≈97s ≈ rollout 生成段(~90s),流水线已近平衡;全量 update 325.8s vs rollout ~180s,进一步压 update 的边际收益递减,D 预期收益(原 −80~130s 按 454.6s 口径估算)大幅缩水且风险最高。
- 结论:不实施;探针脚本为 /tmp 临时产物已删除,决策依据记入本报告与 PROGRESS。

### 机器噪声说明(当日 bwd 波动带)

同日多次运行 bwd 分项波动大(pickle 档 25.3~51.8s、B1 零改动运行 38.7s),与 GPU 功率/频率/利用率无差异(GPU0 功率均值 253~255W、SM 频率 2520MHz 恒定),属 CPU 负载漂移(load average 5→9)影响 host 侧 enqueue 节奏。故跨运行对比一律采用**同窗口紧邻配对**(C 用 eager/compile 前后紧邻两跑,B2 用当日 pickle/shm 两跑),并保留每跑的插计段内自洽分解;绝对提升以全量正式验证单一干净运行为准。

---

## 三、全量正式验证(报告头条数字)

配置:`audit/reports/v18/scripts/v18_ppo_perf_fullscale_verify.yaml`(自包含,resume checkpoint_00150,iterations 152,2048×2,交付态开关全开:torch_compile / rollout_update_overlap / torch_compile_reference / learner_shard_transport=shm / minibatch_size 2048;评测关闭)。
运行:`logs/v18/perf_fullscale_stage2_20260902.log`(2026-09-02 18:15–18:39,完整跑完 2 个 update)。

- **iteration=152(交付态稳态):`algorithm_wall_s = 325.8s`**(第一阶段 454.6s,**−128.8s,−28.3%**);sps **4461**(第一阶段 3144,+41.9%);
- 分项:update_wall 322.6s = shard_select 14.1 + shard_put 0.09 + learner_recv_wait 307.0 + aggregate 0.21 + broadcast 0.22;rank 侧 learner_wall 306.9 = fwd 78.3 + bwd 106.4 + reference_precompute 23.1 + collate_h2d 2.4 + 其他插计 ~2.9;未插计残余 ≈93.9s(GPU 排水/同步点,非黑盒——enqueue 异步计时不捕获真实 GPU 执行,见 B1 分析);
- 传输尾差 recv_wait − learner_wall = **0.08s**(B2 生效);
- collate_wait 0.6s(饥饿仍为 0)、collate_put_block 261.6s(反压,健康);
- 两卡显存:learner 峰值 allocated 28.1GB / reserved 31.5GB(46GB 卡),GPU0 总量峰值 34.1GB(含推理 actor 与常驻 bot ~0.4GB),**无 OOM,评测分片余量充足**;
- sanity:entropy 0.17428(第一阶段 0.17416)/ approx_kl 0.00093(1.15e-3)/ value_loss 0.2775(0.2707)/ clipfrac 0.0094(0.0115)——同量级、无系统性偏移(允许轨迹漂移,不允许分布偏移);
- iteration=151 含编译 warmup(1007.8s),dynamo 重编译(forward_actor 与 forward 各 8 槽)仅出现在热身期,**iteration=152 稳态零重编译**。

---

## 四、语义影响与开关(全部可关闭)

| 开关 | 默认 | 语义等级 | 影响 | 关闭方法 |
|---|---|---|---|---|
| `minibatch_size` | 1536→**2048** | ✗ 有效 batch 30720→40960 | 见第二节 A 披露 | 配置改回 1536(累加保持 10) |
| `torch_compile`(第一阶段) | true | △ | 浮点归约顺序改变、轨迹漂移 | 置 false |
| `rollout_update_overlap`(第一阶段) | true | ✗ 数据滞后一拍 | 流水线重叠 | 置 false |
| `torch_compile_reference`(本阶段) | true(跟随 torch_compile) | △ | reference logits 内核形状级差异(−inf 位逐位一致) | 置 false |
| `learner_shard_transport`(本阶段) | **shm** | ★ 数组逐位一致 | /dev/shm 传递分片,无数值影响 | 置 pickle |
| `update_reference_precompute_batch_size`(既有) | 8192 | △ | 16384 无额外收益,32768 OOM 不采纳 | — |

数值验证:★ 项(B2)单测逐字段逐位一致;△/✗ 项对照分布量级,不允许系统性偏移(见第三节 sanity)。

---

## 五、测试与证据留存

- `riichi_ppo_v1/tests`:231 passed(新增 B1 插计键断言、C 编译对照与构造契约、B2 往返/端到端/非法配置拒绝共 7 项);临时 checkpoint/日志已清理。
- 证据文件(已入库 `audit/reports/v18/report/perf_evidence/`,每次成功验证后先留档再清理 `ppo_perf/`):
  `perf_scaled_a0_baseline_20260902.jsonl`、`perf_scaled_a_mb2048_20260902.jsonl`、
  `perf_scaled_b1_instrument_20260902.jsonl`、`perf_scaled_c_refcompile_20260902.jsonl`(v1 回归问题版)、
  `perf_scaled_c_refcompile_v2_20260902.jsonl`、`perf_scaled_c_refeager_20260902.jsonl`、
  `perf_scaled_c_refchunk16384_20260902.jsonl`、`perf_scaled_b2_shm_20260902.jsonl`、
  `perf_fullscale_stage2_20260902.jsonl`。
- 运行日志:`logs/v18/perf_scaled_*.log`、`logs/v18/perf_fullscale_stage2_20260902.log`(共 10 份,均完整留存)。

---

## 六、未做项与后续建议

1. **D(CUDA graphs)不实施**:单卡探针即因模型 forward 的 resume 帧结构与 cudagraph 静态输出缓冲不兼容而报错(输出 clone workaround 亦无效),需重构 forward 图断点结构;且稳态零重编译使桶静态 shape 无额外收益。若未来延长训练/换 torch 版本,CUDA graphs 方向可重估,届时优先处理 DDP + no_sync 交互。
2. **B2 未做子项(按插计证据排除)**:collate 队列加深/双线程/CPU 亲和性(collate_wait≈0,反压 261.6s 证明供给充足);aggregate 全量数学复用(0.21s);权重广播优化(0.22s)。均收益低于噪声带,不做。
3. **rollout 侧零投入**(重叠吸收):全量 rollout 生成 ~180s 被流水线吸收,优化零墙钟收益。
4. **未插计残余 ≈93.9s 的进一步归因**:rank 侧残余为 GPU 排水(同步点)。若需要,可将 `profile_cuda_sync` 打开一次(每 stage 边界 sync)以获得真实 GPU 执行时间分布,但仅用于分析,不影响交付态。
5. **warmup 期(1007.8s)**:含两套 dynamo 编译(训练 forward + reference forward_actor)。若未来在意冷启动,可考虑 checkpoint 缓存 inductor 缓存(实验性),或保持现状(长训练摊销后可忽略)。

## 七、交付物清单

1. 优化代码 + 测试:6 个独立 commit(`880754c` B1、`2056388` C、`7bb4641` C 修正、`b2d5641` C 消融配置、`db2a419` B2、`f66516f` B2 采纳),每个可独立回滚;
2. 消融配置留档:`audit/reports/v18/scripts/` 下 `v18_ppo_perf_scaled_mb2048.yaml`(既有)、`_ablation_ref_eager.yaml`、`_refchunk16384.yaml`、`_refchunk32768.yaml`、`v18_ppo_perf_scaled_shm.yaml`、`_ablation_shard_pickle.yaml`;`v18_ppo_perf_fullscale_verify.yaml` 更新为交付态(mb2048);
3. 证据文件:`audit/reports/v18/report/perf_evidence/`(9 份);
4. 本报告 + `PROGRESS.md` 追加记录;
5. 收尾 commit:`v18_ppo.yaml` 显式写入本阶段新开关键与 minibatch 2048(注明批准边界),并同步 `test_v18_ppo_config` 的有效 batch 断言。
