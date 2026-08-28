# V18 PPO 性能优化第二轮实施记录(阶段 A / B1-B4 / C1-C5)

> 执行依据:`audit/reports/v18/design/V18 PPO性能优化第二轮执行提示词.md`。
> 基线:`perf-v18-round2-baseline`(`b9af8a3`,2026-08-28)。执行日期:2026-08-28/29。
> 总约束:不改业务语义与训练语义——全部改动限于执行路径/数据通路/校验时机。

## 1. 已实施项与验证

| 主题 | 内容 | commit | 等价级别与验证 |
| --- | --- | --- | --- |
| A1(fix) | 1v3 评测缓存按 checkpoint sha256 内容校验,旧格式一律未命中并重跑原子覆盖(修复 run1/run2 同名旧 JSON 复用事故) | `9524e60` | T1;新增单测:旧格式/同哈希/异哈希三例 |
| A2(perf) | loss 诊断字符串延迟构造(闭包),消除每 minibatch 4 次 `float()` GPU 同步 | `3cbe094` | T1;既有套件回归 |
| A3(perf) | 梯度累积非组末 minibatch 以 `no_sync` 包裹 forward+backward;有限性 `all_reduce(MIN)` 降到每累积组一次(组边界由 `learner_shard_indices` 补齐后的 planned_minibatches 跨 rank 严格对齐) | `cd267a4` | T2;新增双 rank gloo CPU 对照单测(no_sync 累积 vs 逐批 allreduce,参数 allclose atol=1e-6) |
| A4(perf) | `discounted_empirical_returns` gamma=1.0 分段 float64 反向 `np.add.accumulate` 后缀和(50 万行 178ms→3ms);gamma≠1 保留旧循环按参数分派 | `5d50409` | T1;9 例单测 `np.array_equal` 逐位一致(含段边界/尾段/手工值) |
| A5(perf) | `collate` 新增 `include_query_rows`(默认 True),learner 预取与直取路径传 False | `85ee52f` | T1;新增键序/逐键一致单测 |
| B1(perf) | forward 新增 `validate_structure` 开关与 `shared_capacity`/`critic_total_capacity` host 容量;策略头算术索引取代 nonzero+repeat_interleave;校验路径新增 tail 窗口 canonical 契约检查;生产配置**新增**键 `update_validate_structure: false`(唯一配置变更);新增 `docs/v18_ppo_training.md` | `669b1c1` | T1;两态×容量两来源 `torch.equal`(生产 bf16 逐位一致;fp32 仅 6.7e-8 GEMM 形状舍入);契约/违约单测 3 例 |
| B3(perf) | GQA 的 SDPA 在 CUDA 上锁定 `EFFICIENT_ATTENTION`(fail-fast 防静默 5 倍回退);CPU 保持默认 | `d6a9e50` | T1;单测:默认调度本即 mem_efficient(逐位一致)、flash-only 对该 mask 报错 |
| C2(perf) | `compute_kind_row_plan` host(numpy)类别行表 + forward `kind_row_plan`/`critic_kind_row_plan` 静态键表路径;分隔符算术掩蔽;行表单次 pinned 异步 H2D;plan=None 时旧路径保留(eval/SFT 零改动) | `8283f6d` | T1;单测 + GPU 全模型复验 `torch.equal` 逐位一致 |

放弃/跳过项见 §3。

## 2. A/B 性能数据

协议:`v18_ppo_perftest.yaml`(512 局、`update_epochs=4`、`target_kl=0.0`、minibatch 1536、accumulation 10、12 worker×32 env、双卡、3 轮,第 1 轮预热)。基线 `b9af8a3` 先行记录;日志 `logs/v18/ppo_perf2_{baseline,phaseA,B1,B2,B34,C2}_20260828.log`。

### 2.1 第 3 轮(全 policy update,与生产结构一致)逐主题对比

| 指标 | 基线 | 阶段A | B1 | B2 | B34 | C2(最终) |
| --- | --- | --- | --- | --- | --- | --- |
| algorithm_wall_s | 423.7 | 403.2 | 408.1 | 375.6 | 372.2 | **363.9** |
| update/wall_s | 348.9 | 331.1 | 333.9 | 305.2 | 298.9 | **289.2** |
| rollout/wall_s | 73.9 | 71.1 | 73.2 | 69.4 | 72.4 | 73.7 |
| sps | 924.0 | 978.0 | 987.6 | 1064.8 | 1050.3 | **1105.6** |
| fwd ms/批/rank | 233.3 | 227.2 | 154.1 | 141.5 | — | **142.1** |
| bwd ms/批/rank | 372.8 | 369.6 | 436.2(*) | 368.9 | 369.6 | **324.8** |
| 推理 full_forward mean_ms | 22.4 | 21.9 | 19.9 | — | 20.5 | 19.2 |

(*) B1 r3 的 bwd 上浮经微基准证伪为运行窗外部 CPU 负载(约 5.4)而非代码因素:B1 iter1/2 与阶段 A 逐位一致(183ms),且 B2 同代码路径回到 368.9ms。

### 2.2 基线 vs 最终(§10.4 验收表,第 2/3 轮)

| 指标 | 基线(r2/r3) | 最终 C2(r2/r3) | Δ(r3) |
| --- | --- | --- | --- |
| algorithm_wall_s / iteration | 226.9 / 423.7 | 226.4 / 363.9 | **−14.1%** |
| update/wall_s | 156.8 / 348.9 | 149.3 / 289.2 | **−17.1%** |
| rollout/wall_s | 69.1 / 73.9 | 76.1 / 73.7 | −0.3%(噪声带内) |
| update fwd / bwd 每-minibatch ms | 233.3 / 372.8 | 142.1 / 324.8 | **−39.1% / −12.9%** |
| inference padding_fraction | 0.255 | 0.255 | 持平(B4 放弃,见 §3) |
| sps | 1652.6 / 924.0 | 1698.2 / 1105.6 | **+19.7%** |
| forward 同步点/次 | 45 | 7 | −84%(探针实测) |

提示词预期总收益(§10.5):update −135~250s、rollout −25~80s、825→约 495~665s(生产 2048 局)。本轮实测兑现(512 局折算):update −59.7s/轮,按样本量外推生产约 −120s;rollout 持平。收益兑现约 45%(外推口径),逐项归因:A1-A5+B2 兑现(reference 前向 4→1 遍、MC 向量化、no_sync);B1 超预期兑现(fwd −39%);C2 兑现(bwd −12.9%);**rollout 侧收益未兑现**——B4 证明批内排序不改变 padding(padding 与批内顺序无关,批均值 169 行 ≪ 512 切块阈值),真正的 rollout 杠杆是跨等待窗凑批与 eval 重叠(均属本轮排除项),C3 因图捕获与动态类别行的冲突放弃,详见 §3。

## 3. 放弃项(证据驱动)

| 主题 | 结论 | 证据 |
| --- | --- | --- |
| B4 推理批按长度排序 | 实施→A/B→**revert**(`13dbd25`→`0f22103`) | padding_fraction 0.255 与基线持平、rollout 噪声带内;padding=行数×批内最大长度−Σ长度,与批内顺序无关;`full_forward_rows_mean≈169≪512`,批从不跨切块,排序无效。真正杠杆:跨等待窗凑批(机制类,本轮排除) |
| C1 torch.compile | 探针后**放弃** | fullgraph 成功且 1.25×(44.6→37.9ms/step)、数值在容差内;但 state_dict 键引入 `_orig_mod` 破坏 checkpoint 契约;`dynamic=False` 需静态形状而生产 L 随分桶变化(2 形状已 6 张图,重编译爆炸);`dynamic=True` 185ms/step(慢 4.2 倍)。按"回退仍无收益"放弃 |
| C3 推理 CUDA Graph | **放弃** | 图捕获需区域内全静态形状且零设备同步;类别行组成为逐批动态数据,静态化需 K×桶静态索引缓冲+dummy 行补齐(与 C2 行表路径冲突);旧 argsort 路径含 tolist 无法捕获 |
| C4 FlexAttention | **放弃** | 前置"C1 成功后追加"未满足;融合内核依赖 torch.compile,受同一形状爆炸约束 |
| C5 Rust 下沉 critic_feature_encode | **跳过** | 前置"B4/C3 落地后 worker CPU 成为新约束"未发生;worker CPU 仍被推理 RPC 等待掩盖 |

## 4. 回归结果

- 基线 HEAD(b9af8a3):`pytest riichi_ppo_v1/tests` = **1 failed, 200 passed**(唯一失败为既有 `test_v18_ppo_config_matches_stability_plan`:断言 actor LR 6e-5,而第二轮配置 f28b958 已上调 9e-5,与本轮无关,未擅自改动)。
- 最终 HEAD:**1 failed(同一既有失败), 219 passed**(新增 19 例单测全绿),不低于基线。
- 全程无 Rust/wheel 改动,无需重装扩展;`cargo test` 不适用。

## 5. 交付与回滚

- 提交序列(每主题独立、可单独 revert):`9524e60` → `3cbe094` → `cd267a4` → `5d50409` → `85ee52f` →(审计 `7175aa5`)→ `669b1c1` →(审计 `47ea483`)→ `5a8dcf4` →(审计 `6b64de4`)→ `d6a9e50` → `13dbd25` → `0f22103`(revert)→(审计 `b847ee6`)→ `8283f6d` →(审计 `7287622`)。
- 单主题回滚:`git revert <sha>`(全为 Python/配置/文档,无 wheel)。
- 全量回档:`git reset --hard perf-v18-round2-baseline`(`checkpoints/`、`datasets/`、`logs/`、`audit/` 产物不受影响)。
- 生产配置 `v18_ppo.yaml` 相对基线的全部差异仅为 B1 允许的新增键 `update_validate_structure: false`(`git diff b9af8a3 -- riichi_ppo_v1/configs/v18_ppo.yaml` 自查通过)。

## 6. 后续轮次遗留(按 §10.6)

- 配置类 A/B:minibatch 3072+accumulation 5、`envs_per_worker=64`、`inference_batch_wait_ms` 扫描。
- eval 与训练重叠、staleness-1 rollout/update 流水重叠、分片共享内存传输。
- 推理批跨等待窗凑批(以批内长度分桶组装;需新批装配机制,B4 的证据指向此处才是 padding 的真正杠杆)。
- torch.compile 形状分桶方案(L 按 32 对齐 + B 固定/尾部补齐,需评估 padding 计算与 executed 语义边界)与 CUDA Graph 的静态索引缓冲设计(可与凑批合并设计)。
- critic 组装路径 6 个 bool-indexing 同步的算术化(本轮探明但未列入范围)。
