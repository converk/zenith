# V18 PPO 训练提速优化报告

**日期**:2026-09-01 至 2026-09-02
**执行依据**:`audit/reports/v18/design/V18_PPO训练提速执行提示词.md`(维护者 2026-09-01 批准 Tier 2 全部方向)
**基线数据**:`logs/v18/v18_ppo_r5_20260829.log` + `checkpoints/train_riichi_v18/ppo/performance.jsonl`(150 updates,iteration 5–150 均值)
**测试协议**:提示词第五节(缩放口径 1024×1×2 updates 测第 2 个;全量验证 2048×2×2 updates 测第 2 个;不用 AGENTS.md 默认基线口径)

---

## 一、总结论(TL;DR)

| 口径 | 优化前 | 优化后 | 收益 |
|---|---|---|---|
| **全量正式验证(2048×2,交付态)** | **746.0s / update**(r5 稳态均值) | **454.6s / update** | **−39.1%** |
| 缩放口径(1024×1,迭代过程) | 285.7s | 159.4s | −44.2% |
| sps(全量) | ≈1793 | ≈3144 | +75% |

全量口径分项对照(iteration=152,交付态 vs r5 稳态均值):

| 指标 | r5 稳态(优化前) | 交付态(优化后) | 说明 |
|---|---|---|---|
| algorithm_wall_s | 746.0 | **454.6** | rollout 与 update 重叠,总墙钟 ≈ max(update, rollout) |
| rollout_wall_s | 230.6(串行占用) | 454.6(含与 update 并行等待) | 重叠态该指标=流水线墙钟,非纯生成时间 |
| update_wall_s | 512.5 | 451.3(其中 fwd 99.3 + bwd 162.2) | forward+backward 由 399.7s → 261.5s(−35%) |
| reference_precompute | 53.2 | 75.4 | 与 rollout 争用 GPU(重叠态),墙钟被流水线吸收 |
| collate_h2d | 15.8 | 3.1 | pinned+non_blocking+uint8 |
| learner 峰值显存 allocated/reserved | ≈22GB / ≈38GB | 21.9GB / 38.4GB | 无恶化 |
| 数值 sanity(entropy/approx_kl/value_loss) | 0.1745 / 6.2e-4 / 0.281 | 0.1742 / 1.15e-3 / 0.271 | 同量级,无系统性偏移 |

---

## 二、实施项明细(每项独立 commit)

### Tier 0:零/低风险纯通路(★ 逐位不变)

| # | 优化 | commit | 语义 | 缩放口径 before→after(update 152) |
|---|---|---|---|---|
| 0.1 | learner H2D:pinned+non_blocking+uint8 紧凑索引 | `53fa306` | ★ | collate_h2d 4.14→**0.98s**;update 157.5→153.4s;整轮 285.7→277.2s |
| 0.2 | 推理凑批「到齐即 flush」(quorum) | 见提交 | ★ | rollout 122.0→**90.1s**(−26%);queue_wait 325→5.0s;dispatch 由 100% 超时触发变为 quorum 7915/7998 |
| 0.3 | Rust release profile(thin LTO + 单 codegen unit + target-cpu=native) | `34fb828` | ★ | 整轮 244.5→241.9s(噪声内;Rust CPU 段本就仅占 rollout ~19%) |
| 0.4 | critic 装配布尔索引改算术 gather/scatter | `2632c13` | ★ | 整轮 241.9→241.3s(噪声内;主要价值是消除 nonzero GPU→CPU 同步点,为 compile 铺路)。真实 env 25 步新旧实现逐位一致(maxdiff=0) |
| 0.5 | minibatch 1536→2048 测量变体 | `11deec9` | ✗(未采纳) | 整轮 241.3→245.2s(负收益),显存峰值 +7GB → **不采纳**,配置留档 |

**Tier 0 小计**:285.7→241.3s(−15.5%)。

### Tier 1:中等改造(★ 逐位不变)

| # | 优化 | commit | 语义 | 缩放口径 before→after |
|---|---|---|---|---|
| 1.6 | SFT reference 预计算 collate 后台线程化 | `037a492` | ★ | reference_precompute 24.9→26.1s(噪声内,保留实现);逐位一致 |
| 1.7 | 推理 actor 事件循环解耦(run_in_executor) | `5aa049b` | ★ | rollout 90.8→87.3s(−3.9%) |
| 1.9 | Rust 批编码器释放 GIL + 停写全局 mjai_log | `368d0eb` `852b515` | ★ | 单独墙钟在噪声带内;结构性改进(GC/RSS/多 worker 并发铺路);per-player 事件流逐位不变 |
| 1.10 | worker→推理 RPC 载荷 uint8 紧凑化 | `e65a6d3` | ★ | rollout 95.4→90.3s;object store 流量缩 4 倍;载荷等价性验证逐位一致 |

**决策不做项**(按性价比,证据留档):
- **1.8 worker 双缓冲**:收益上限 ~10s/90s(3.5% 整体),但 2.14 重叠后 rollout 不在关键路径 → 不做;
- **1.11 driver 侧 42s gap**:实测未计时 gap 仅 ~17s/746s(≈2.3%,提示词的 42s 为高估:该口径把已重叠的 collate_soa_gather 45.4s 重复计入)→ 不做;
- **1.12 Transition 物化免拷贝**:收益是 worker RSS(29GB/worker)而非墙钟(transition_materialize 仅 1.2s/worker),为将来扩 env 铺路,当前不扩 → 不做。

**Tier 1 小计**:241.3→246.2s(±4s 机器噪声带;单看确定性收益项 0.1/0.2/1.7/1.10 合计约 −40s)。

### Tier 2:高收益(已获维护者 2026-09-01 批准,可关闭开关、默认开启)

| # | 优化 | commit | 语义 | 缩放口径 before→after |
|---|---|---|---|---|
| 2.13 | learner torch.compile(DDP 包装前编译原始模块;权重读写经 `_orig_mod` 保持 checkpoint 契约) | `f26bcdb` | △ | fwd 32.8→19.9s(−39%)、bwd 74.4→**29.0s**(−61%);update 154.2→111.3s;整轮 246.2→**203.7s** |
| 2.14 | rollout/update 流水线重叠(数据滞后一拍+逐轮冻结权重广播;首迭代退化为串行) | `40ed96b` | ✗ | 稳态整轮 203.7→**159.4s**;总墙钟 ≈ max(rollout, update)+收尾 |

**Tier 2 关/开消融**(缩放口径,update 152,其余优化全开):

| 配置 | algorithm_wall_s | 说明 |
|---|---|---|
| 交付态(全开) | **159.4s** | — |
| 关重叠(rollout_update_overlap: false) | 205.7s | 重叠单独收益 −46.3s(−22.5%) |
| 关编译(torch_compile: false,重叠仍开) | 203.8s | 编译使 update fwd+bwd 134s→48s;关掉后 update(202s)成为绝对瓶颈,总墙钟由 rollout 段决定 |

**重编译计数**:dynamo 对 `architecture.forward` 共触发 recompile_limit(8 次)上限,全部发生在 update 151(编译 warmup 期),触发原因是对 shared_capacity 的动态形状特化(`(shared_capacity²)%8 != 0` 的 SDPA guard);**update 152 稳态零重编译**。全量验证 iter151 warmup 904.3s(含编译),iter152 稳态 454.6s。

**2.15 env 一级公民**:按决策规则评估——重叠稳态下关键路径为 update 侧(451.3s > rollout 生成 ~180s),不在 rollout 侧 → **不实施,交付设计提案**(见第六节)。

---

## 三、5.2 全量正式验证(报告头条数字)

配置:`audit/reports/v18/scripts/v18_ppo_perf_fullscale_verify.yaml`(自包含,resume checkpoint_00150,iterations 152,2048×2,eval 关闭,输出重定向;其余键与 v18_ppo.yaml 逐项相同)。
运行:`logs/v18/perf_fullscale_verify_20260902.log`(2026-09-02 00:33–01:03,完整跑完 2 个 update)。

- **iteration=152(交付态稳态):algorithm_wall_s = 454.6s**(r5 稳态 746.0s,**−39.1%**);sps 3144(r5 1793,+75%);
- 流水线时间线:rollout(发出后与 update 并行)与 update(451.3s)重叠,总墙钟由 update 决定;
- 分项:fwd 99.3s + bwd 162.2s + reference_precompute 75.4s + collate_h2d 3.1s(编译态;r5 为 122.1 + 277.6 + 53.2 + 15.8);
- 两卡显存峰值:learner allocated 21.9GB / reserved 38.4GB(与 r5 同级,learner 与推理 actor 同卡共存无恶化);
- sanity:entropy 0.17416(r5 尾部 0.17454)/ approx_kl 1.15e-3(6.2e-4)/ value_loss 0.2707(0.2808)/ clipfrac 0.0115(0.0058)——同量级、无系统性偏移(允许轨迹漂移,不允许分布偏移);
- iteration=151 含编译 warmup(904.3s),符合「warmup 留在第 1 个 update」的预期。

---

## 四、Tier 2 语义影响与关闭方法

| 开关 | 默认 | 语义等级 | 影响 | 关闭方法 |
|---|---|---|---|---|
| `torch_compile` | true | △ 数学等价、浮点归约顺序改变 | 训练轨迹会漂移(与未编译不可逐位复现);模型输入语义逐位不变 | 配置置 `torch_compile: false`(须重启训练) |
| `rollout_update_overlap` | true | ✗ 训练语义变更(数据滞后一拍) | 采样权重比 update 起点旧一拍(逐轮冻结、陈旧度均匀);PPO ratio 逐行自洽(old_logprob 为采样时真实策略);`algorithm_wall_s ≠ rollout + update` | 配置置 `rollout_update_overlap: false` 回退严格串行 |

两者均为自包含配置键,已显式写入 `v18_ppo.yaml`(注明维护者批准日期 2026-09-01)。

---

## 五、maturin 重装记录(第六节坑规避)

凡改 Rust(Tier 0.3 / 1.9)均执行 `RUSTFLAGS="-C target-cpu=native" bash RiichiEnv/scripts/install_conda_extension.sh` 并以 `riichi.__file__` + site-packages 下两个 `.so` 的 mtime(20:06、21:47、22:11,均晚于源码 mtime)确认加载新编译产物。注:`riichi` 的 editable 安装指向不存在的 `file:///mnt/disk1/hubowen/zenith/RiichiEnv/riichi`,实际加载的是 site-packages 副本——install_conda_extension.sh 的 wheel 安装路径正确覆盖该副本。

---

## 六、未做项与后续建议

1. **Tier 2.15 env 一级公民(设计提案,按决策规则不实施)**:Rust `step_batch` 已支持部分动作 map 原地等待(`env.rs:116-144`),锁死点在 Python 编排:同步 for 循环、`batch_index = env_index*4 + seat_id` 槽位绑定(`bridge.py:36-37`)、collect 的齐步半庄停止条件。若未来 update 侧进一步压缩(如更激进的编译策略或显存允许的更大 chunk)使 rollout 生成段(~180s)成为关键路径,则按以下顺序实施:(a) worker 内存瘦身(1.12 的 flat+offset 变长布局,释放 ~29GB/worker);(b) env 状态机槽位解绑(决策粒度 env×seat 索引化);(c) collect 停止条件改为按 env 独立半庄计数。当前 451s update ≫ 180s rollout,无实施必要。
2. **recompile guard 特化**:动态 shared_capacity 触发 8 次 guard 特化(仅 warmup 期)。若未来延长训练,可在 collate 侧把 shared_capacity/critic_total_capacity pad 到 8 的倍数,消除该 guard 变体(收益:warmup −几分钟,稳态无变化)。
3. **worker 内存瘦身(1.12)**:扩 envs_per_worker 前的前置项,当前不做。
4. **minibatch 2048**:负收益(−3.9s)+7GB 显存,不采纳;消融配置留档 `v18_ppo_perf_scaled_mb2048.yaml`。

---

## 七、交付物清单

- 优化代码 + 测试:12 个独立 commit(`53fa306`…`40ed96b`),每个可独立回滚;
- 测试配置留档:`audit/reports/v18/scripts/v18_ppo_perf_scaled.yaml`(缩放口径)、`_mb2048.yaml`、`_ablation_no_overlap.yaml`、`_ablation_no_compile.yaml`、`v18_ppo_perf_fullscale_verify.yaml`(全量验证);
- 运行日志:`logs/v18/perf_scaled_*.log`、`logs/v18/perf_fullscale_verify_20260902.log`;
- 本报告 + `PROGRESS.md` 追加记录;
- 收尾 commit:`v18_ppo.yaml` 显式写入 Tier 2 开关键(注明批准日期)。

测试:`riichi_ppo_v1/tests` 224 passed;`RiichiEnv/tests` 190 passed + 2 skipped;`cargo test --workspace` 95 passed。测试产生的临时 checkpoint/日志已清理,正式 checkpoint 目录只读未动。

---

## 八、交付后修正与证据留存说明(2026-09-02,commit `788dd01`)

1. **2.14 首迭代数据簿记缺陷与修正**:初版重叠编排(`40ed96b`)在首个 update 串行
   收集本份 rollout 并消费后,又在 update 结束时收割**同一组 refs** 留给下一个
   update——同一份 rollout 被两个 update 各消费一次,超出获批的「滞后一拍」语义。
   证据:验证/缩放运行的相邻迭代 transitions 计数完全相同(全量验证 151/152 均为
   1429205;缩放 2.14 运行均为 765240)。
   修正(`788dd01`)改为规范形态:迭代 k 在 update 前发出「k+1 轮」rollout(与本轮
   update 并行,逐轮冻结权重),update k 消费上一轮收割的 rollout k;每份数据恰好
   消费一次,最后一个 update 不发起不会被消费的 rollout。
2. **对已报数字的影响**:计时口径有效——两次运行的 rollout 工作量相同(同
   games_per_update×epochs),墙钟对照不受影响;修正还移除了末轮「注定不消费的
   收割等待」(~3s),454.6s 为保守口径。**数值代表性受限**:5.2 验证与 2.14 缩放
   测试的 update 152 数值(entropy/approx_kl 等)采自重复消费的数据批(等效对同一
   批多跑一遍 epoch),量级 sanity 仍成立,但如需干净的「修正后」数值对照,应在
   `788dd01` 之后重跑(维护者已决定不再重跑)。
3. **证据留存**:原始成功验证(2026-09-02 00:33–01:03)的
   `perf_fullscale_verify_20260902.log` 与 `ppo_perf/fullscale_verify/performance.jsonl`
   在一次被取消的重复确认运行启动时被覆盖/清理;头条数字存档于 commit `11c6421`
   与 `d423e0d` 的提交信息及本报告第二节/第三节表格。缩放口径各阶段的
   `logs/v18/perf_scaled_*.log` 均完整留存。
4. **测试**:修正后 `riichi_ppo_v1/tests` 224 passed(含 CPU 上的 2.14 编排路径
   集成测试);`v18_ppo.yaml` 未再改动。
