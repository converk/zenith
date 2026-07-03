# Clean RL 风格 PPO 训练方法评估报告

**评估对象**: `test-model/` 目录
**对比参照**: CleanRL 官方 PPO 教程 (Basic + Advanced)
**评估日期**: 2026-07-02

---

## 总体评价：✅ **高度适合**，结构几乎等同于 CleanRL 风格

`test-model/ppo.py` 的代码结构与 CleanRL 的经典 PPO 实现**高度一致**——可以用"教科书级别"来形容。以下是逐项详细分析：

---

## 1. 采样阶段保存的数据（Rollout Buffer）

### CleanRL 标准做法
- 用一个固定大小 `[num_steps, num_envs, ...]` 的 Tensor 数组存储
- 每个 step 依次存入 `obs`, `actions`, `logprobs`, `rewards`, `dones`, `values`
- 采样结束后 flatten 成 `[batch_size, ...]` 送入 update

### test-model 实现

```python
# ppo.py collect_rollout(), lines 167-173
obs = torch.zeros((args.num_steps, num_actors, 34), device=device)
legal_masks = torch.zeros((args.num_steps, num_actors, 34), dtype=torch.bool, device=device)
actions = torch.zeros((args.num_steps, num_actors), dtype=torch.long, device=device)
logprobs = torch.zeros((args.num_steps, num_actors), device=device)
rewards = torch.zeros((args.num_steps, num_actors), device=device)
dones = torch.zeros((args.num_steps, num_actors), device=device)
values = torch.zeros((args.num_steps, num_actors), device=device)
```

**评价**: ✅ 完全符合。预分配固定大小的 rollout buffer，存储了所有必要字段。特别值得称赞的是额外存储了 `legal_masks`（action mask），这在实际博弈环境中至关重要——CleanRL 基础教程未涉及但 Advanced PPO 中有类似概念。

**注意**: `num_actors = num_envs * 4`（4 个玩家），这意味着每个玩家都被视为独立的 actor，采用**参数共享 + 独立采样**的方式，这是多智能体 PPO 的标准做法。

---

## 2. PPO 超参数（Hyperparameters）

### 对比表

| 参数 | CleanRL 典型值 | test-model 值 | 评价 |
|------|:---------:|:---------:|------|
| `learning_rate` | `2.5e-4` ~ `3e-4` | `3e-4` | ✅ |
| `gamma` | `0.99` | `0.99` | ✅ |
| `gae_lambda` | `0.95` | `0.95` | ✅ |
| `clip_coef` | `0.1` ~ `0.2` | `0.2` | ✅ |
| `ent_coef` | `0.01` ~ `0.1` | `0.01` | ✅ |
| `vf_coef` | `0.1` ~ `0.5` | `0.5` | ✅ |
| `max_grad_norm` | `0.5` | `0.5` | ✅ |
| `num_steps` | `128` | `128` | ✅ |
| `update_epochs` | `4` (Basic) / `3` (Advanced) | `4` | ✅ |
| `num_minibatches`| `4` | `4` | ✅ |
| `norm_adv` | `True` | `True` | ✅ |
| `clip_vloss` | `True` (Advanced) | `True` | ✅ |
| `anneal_lr` | `True` | `True` | ✅ |
| `target_kl` | `None` ~ `0.01` | `None` | ✅ |
| `total_timesteps` | `100_000` | `100_000` | ✅ |
| `batch_size` | 动态计算 | 动态计算 | ✅ |

**评价**: ✅ 参数几乎就是从 CleanRL 标准 PPO 移植过来的。`clip_vloss = True` 是 Advanced PPO 的特性，也正确实现了。`total_timesteps = 100_000`，以 `batch_size = 4 * 4 * 128 = 2048` 计算，约 48 个 iteration，对于 toy 环境足够，生产环境可以调大。

---

## 3. 并行环境（Parallel Environments）

### CleanRL 标准做法
- **Basic PPO**: 使用单个 PettingZoo ParallelEnv，单 episode 逐 step 采样
- **Advanced PPO**: 使用 `ss.pettingzoo_env_to_vec_env_v1` + `ss.concat_vec_envs_v1` 构建多环境向量化

### test-model 实现

```python
# 训练侧: RiichiVectorEnv (num_envs=4), 每步同时执行 4 个对局
# Rust 侧: VecEnv 包含 Vec<State>，每个 State 独立运行
# Python 侧: 不依赖 PettingZoo 的向量化工具，直接使用 Rust 批量接口
```

**评价**: ✅ **优于 CleanRL Basic，接近 Advanced 风格**。`RiichiVectorEnv` 是一个**原生批量环境**：

- 打破了 PettingZoo 的 `num_envs=1` 限制
- 底层 Rust `VecEnv` 持有 N 个独立的 `State`，在 Rust 层面并行迭代
- 没有使用 PettingZoo 的 `RiichiParallelEnv`（它只包装单个 game，保留给 API 兼容）
- 这是一个有意的架构选择：**训练用 `RiichiVectorEnv`（高性能）**，API 兼容用 `RiichiParallelEnv`（PettingZoo 标准）

**优势**:
- Rust 层面批量推理避免了 Python GIL 瓶颈
- 无需 `SubprocVecEnv` 的多进程开销（CleanRL Advanced 使用 `concat_vec_envs_v1` 配合 `num_cpus=0`）
- 玩家（4 个 agent）在环境中被批处理为 `[num_envs, 4, 34]` 的 Tensor

---

## 4. 训练循环结构（Collect → Compute → Update）

### CleanRL 标准伪代码

```
for iteration in range(num_iterations):
    # 1. Collect rollout (no_grad)
    obs, actions, logprobs, rewards, dones, values = rollout(env, agent)
    # 2. Compute GAE advantages + returns
    advantages, returns = compute_gae(...)
    # 3. Update policy (K epochs, minibatch SGD)
    for epoch in range(K):
        for minibatch in shuffled_batches:
            loss = pg_loss + vf_loss - entropy_bonus
            optimizer.step()
    # 4. Log metrics
```

### test-model 实现 (ppo.py `train()`, lines 334-414)

```python
for iteration in range(start_iteration, args.num_iterations + 1):
    if args.anneal_lr:                    # lr annealing
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        optimizer.param_groups[0]["lr"] = frac * args.learning_rate

    rollout, ... = collect_rollout(...)    # 1. 采样
    advantages, returns = compute_gae(...) # 2. GAE
    metrics = update_policy(...)           # 3. 更新
    log_metrics(...)                       # 4. 日志 (TensorBoard + WandB)
    eval_if_needed(...)                    # 5. 定期评估
    save_if_needed(...)                    # 6. 定期保存
```

**评价**: ✅ **完全一致**。CleanRL 标准的三阶段循环（Collect→Advantage→Update）被精确复制，并额外增加了定期评估和 checkpoint 保存。

---

## 5. GAE 计算

### test-model (`ppo.py` `compute_gae()`, lines 217-241)

```python
next_value = agent(next_obs, next_masks)["value"].reshape(1, -1)
advantages = torch.zeros_like(rollout["rewards"])
lastgaelam = 0

for t in reversed(range(args.num_steps)):
    if t == args.num_steps - 1:
        nextnonterminal = 1.0 - rollout["dones"][t]
        nextvalues = next_value.reshape(-1)
    else:
        nextnonterminal = 1.0 - rollout["dones"][t + 1]
        nextvalues = rollout["values"][t + 1]
    delta = rollout["rewards"][t] + args.gamma * nextvalues * nextnonterminal - rollout["values"][t]
    advantages[t] = lastgaelam = (
        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    )

returns = advantages + rollout["values"]
```

**评价**: ✅ 教科书级别的 GAE 实现。
- 正确处理了 bootstrap 边界：用 `next_value` 作为最后一个 step 的终态价值估计
- `nextnonterminal` 掩码正确处理了终止状态（done=1 时切断信用分配链）
- 与 CleanRL Advanced PPO 的 GAE 代码几乎逐行一致

---

## 6. 损失函数（Policy Loss + Value Loss + Entropy）

### Policy Loss (Clipped PPO) — `ppo.py` lines 287-293

```python
pg_loss1 = -mb_advantages * ratio
pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
pg_loss = torch.max(pg_loss1, pg_loss2).mean()
```

✅ 标准 PPO clipped objective。

### Value Loss (Clipped) — lines 295-303

```python
if args.clip_vloss:
    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
    v_clipped = b_values[mb_inds] + torch.clamp(
        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
    )
    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
```

✅ **这是 CleanRL Advanced PPO 的特性**，对 value 也做了 clip，防止 value function 更新过大。

### Total Loss — line 307

```python
loss = pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss
```

✅ 标准组合。

### Early Stopping — lines 313-314

```python
if args.target_kl is not None and approx_kl > args.target_kl:
    break
```

✅ KL-based early stopping（当前 `target_kl=None` 所以未启用，但机制已就位）。

### Advantage Normalization — lines 282-285

```python
if args.norm_adv:
    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
        mb_advantages.std() + 1e-8
    )
```

✅ 在 minibatch 级别进行 advantage normalization，与 CleanRL Advanced 一致。

---

## 7. 模型架构（Actor-Critic Network）

### 架构概览

`TileCountTransformerActorCritic` 使用：

1. **TileCountEmbedding**: `tile_embedding(34种牌)` + `count_embedding(0~4张)`，逐位置相加
2. **Decision Token**（类似 ViT/BERT 的 [CLS] token）：可学习的 `nn.Parameter` 拼接在序列末尾
3. **Transformer Encoder** (8 layers, 12 heads, d_model=384):
   - RMSNorm (Pre-LN)
   - FullSelfAttentionWithRoPE (RoPE 旋转位置编码)
   - SwiGLU FFN
4. **Dual Head Output**:
   - Policy Head: RMSNorm → Linear(1536) → SiLU → Dropout → Linear(34)
   - Value Head: RMSNorm → Linear(1536) → SiLU → Dropout → Linear(1)

### CleanRL 对比

| 维度 | CleanRL Basic/Advanced | test-model |
|------|:---:|:---:|
| 观测 | 84×84×4 图像 (CNN) | 34 维牌计数 |
| 网络 | CNN → FC | Transformer Encoder |
| Actor Head | FC(→n_actions) | MLP(→34) |
| Critic Head | FC(→1) | MLP(→1) |
| Action Mask | 未涉及 | ✅ `masked_fill(-inf)` |
| 架构复杂度 | 简单 | 较复杂（Transformer） |

CleanRL 教程使用简单的 CNN（用于图像观测）→ 全连接 → Actor/Critic 双头。test-model 的 Transformer 架构远更复杂，但**Actor-Critic 双头设计**的核心范式完全一致。

**关于模型规模的讨论**：对于 34 维牌计数输入，8 层 384 维的 Transformer（约 12M 参数）可能偏大。但这是一个合理的起点——可以先训练看效果，再根据实际需要缩小或调整。模型支持 `small`/`medium`/`large` 三档预设，实验对比很方便。

---

## 8. Action Mask（非法动作屏蔽）

**这是 test-model 相比 CleanRL 基础教程的一个关键增强**：

```python
# ppo_model.py lines 218-222
if legal_mask is not None:
    legal_mask = legal_mask.to(device=policy_logits.device, dtype=torch.bool)
    policy_logits = policy_logits.masked_fill(
        ~legal_mask, torch.finfo(policy_logits.dtype).min
    )
```

在麻将中，玩家只能打出手中持有的牌（count > 0）。通过将非法动作的 logit 设为 `-inf`（softmax 后为 0），保证了：

- 采样阶段不会选非法动作
- 训练阶段 gradient 不受非法动作影响
- rollout buffer 中保存 `legal_masks` 用于 update 时的 logits masking

✅ 这是处理离散动作空间约束的标准做法，在 CleanRL Advanced PPO 的环境设置中提到但不属于核心算法。**实现完全正确**——注意使用了 `torch.finfo(dtype).min` 而非硬编码的 `-1e9`，这是更健壮的做法。

---

## 9. 完整度检查

| 功能 | 状态 | 说明 |
|------|:---:|------|
| Rollout 采样 | ✅ | `collect_rollout()` — 完整的 buffer 存储 |
| GAE 计算 | ✅ | `compute_gae()` — 含 bootstrap + 终止态处理 |
| Policy Update | ✅ | `update_policy()` — clipped PPO + clipped value loss |
| LR Annealing | ✅ | 线性退火到 0 |
| Advantage Norm | ✅ | minibatch 级别归一化 |
| KL Early Stop | ✅ | 可选，机制已就位 |
| 模型保存 | ✅ | `save_checkpoint()` — 保存 model/optimizer/args/step |
| 模型加载 | ✅ | `load_checkpoint()` — 支持断点续训 (`--resume-from`) |
| 定期评估 | ✅ | `evaluate()` — 独立 eval env, 确定性策略 |
| TensorBoard | ✅ | `SummaryWriter` — 完整的 metrics 记录 |
| WandB | ✅ | 可选 `--track` |
| 随机种子 | ✅ | random/numpy/torch 三重种子 |
| Device 管理 | ✅ | cuda/cpu 自动检测 |
| 视频录制 | ❌ | CleanRL Advanced 有，本项缺失 |

**总结**：CleanRL 风格要求的核心功能**全部具备**。唯一缺失的是视频录制（`capture_video`），但这对麻将训练不是必需的。

---

## 10. 存在的不足和改进建议

### 10.1 ⚠️ 日志记录（Minor）

```python
# ppo.py line 197
episodic_returns.append(float(np.mean(episode_returns[env_index])))
```

在 4 个玩家上取 mean 可能会混淆个体表现——无法区分是某个玩家特别强还是所有人都在进步。建议分开记录各玩家的 reward，或者明确注释这是 per-agent 的平均。

### 10.2 ⚠️ SPS 语义（Minor）

```python
# ppo.py line 403
int(global_step / (time.time() - start_time))
```

`global_step` 每步增加 `num_envs * 4`（agent 数），所以 SPS 值实际上是 "agent-steps per second" 而非 "environment steps per second"。在多智能体设定下这是合理的选择，但建议加注释说明。

### 10.3 ⚠️ 环境奖励设计（Medium）

```rust
// game.rs lines 331-362
pub fn step(&mut self, discard: &[u8]) -> ([u8; 136], [f32; 4], bool) {
    for p in 0..4 { self.discard(discard[p]); }  // 4 players all discard
    for _p in 0..4 { self.draw(); }               // 4 players all draw
    reward[p] = enumerate_points(...);             // computed after everyone's move
}
```

这是一个**同步回合制**简化为**同时行动**的模式。当前环境将真实麻将的轮流弃牌（AEC 风格）改为所有玩家同时行动。对于 PPO 训练这可以工作，但与真实麻将不同：

- 没有轮次顺序（真实麻将按顺序摸牌、弃牌）
- 没有 response 阶段（碰/吃/杠/荣和）
- 奖励是 simultaneous 计算而非 sequential

**建议**: 如果后续需要更真实的麻将模拟，可能需要：
1. 将环境改为 AEC (Agent Environment Cycle) 格式
2. 引入 sequential decision making
3. 处理 response 动作空间（碰、吃、杠、荣、自摸）

### 10.4 ⚠️ 模型规模（配置级）

- 对于 34 维牌计数输入，`num_layers=8, d_model=384`（约 12M 参数）可能 overkill
- 建议先用 `model_size="small"`（d_model=128, layers=2, 约 1M 参数）验证训练能收敛，再逐步放大

### 10.5 ⚠️ 奖励稀疏（环境级）

```rust
// game.rs line 322
if (num_pairs == 1 && num_groups == 4) || (num_pairs == 7) {
    return 1f32;  // 和牌奖励 = 1
}
// ...
// game.rs line 340, 未和牌时:
reward[p] = -(self.cache.completion_distance(player_tiles, 1, 4) as f32);
```

奖励设计：
- 和牌时：+1（稀疏正奖励）
- 未和牌时：负的 completion distance（密集负奖励，引导接近和牌）
- 终局时：同样用 completion distance 作为负奖励

这个设计合理——completion distance 提供了足够的学习信号。但 `completion_distance` 的范围为 0-14（编辑距离），与 +1 的和牌奖励在量级上可能不平衡。建议监控 episode return 的数值范围。

---

## 11. 总结评分

| 维度 | 评分 | 说明 |
|------|:---:|------|
| 采样数据结构 | ⭐⭐⭐⭐⭐ | 预分配 buffer、存储全字段、含 action mask |
| PPO 参数设定 | ⭐⭐⭐⭐⭐ | 完全对标 CleanRL 标准 |
| GAE 计算 | ⭐⭐⭐⭐⭐ | 正确实现，含 bootstrap 边界处理 |
| 损失函数 | ⭐⭐⭐⭐⭐ | 含 clipped value loss (Advanced 特性) |
| 训练循环 | ⭐⭐⭐⭐⭐ | 标准三阶段，lr annealing，含 eval + checkpoint |
| 并行环境 | ⭐⭐⭐⭐ | Rust 批量环境优于 Python SubprocVecEnv |
| 模型架构 | ⭐⭐⭐⭐ | Transformer 可能 overkill 但设计规范，支持多档预设 |
| 日志/保存 | ⭐⭐⭐⭐⭐ | TensorBoard + WandB + 完整 checkpoint（save/load）|
| 代码整洁度 | ⭐⭐⭐⭐⭐ | 模块化清晰，类型标注完整 |

**总评**: 这份代码是一个**高质量的 CleanRL 风格 PPO 实现**，可以作为麻将 AI 训练的基础框架。核心的 PPO 算法实现没有任何原则性问题，结构完全遵循 CleanRL 的 single-file 哲学且几乎逐函数对应。主要的工作不在 PPO 算法本身，而在于：(1) 环境奖励设计是否合理，(2) 模型架构是否针对麻将问题做了合适的归纳偏置，(3) 是否需要在后续转向更真实的麻将环境（AEC 格式、response 动作空间）。

---

## 附录：文件职责一览

| 文件 | 职责 |
|------|------|
| `ppo.py` | PPO 主算法：采样、GAE、更新、训练循环、评估、checkpoint |
| `ppo_config.py` | 超参数配置 dataclass + CLI 解析 + 派生参数计算 |
| `ppo_model.py` | Transformer Actor-Critic 网络 + action mask |
| `riichi_parallel_env.py` | 环境包装器（VectorEnv for 训练 + ParallelEnv for 评估） |
| `riichi/src/game.rs` | Rust 麻将游戏逻辑（状态、和牌判断 via 形式枚举） |
| `riichi/src/lib.rs` | PyO3 绑定（VecEnv 批量接口，`[N, 4, 34]` → step → observation/reward/done） |
| `requirements.txt` | Python 依赖（torch, gymnasium, pettingzoo, tensorboard, wandb, maturin） |
