# V18 PPO 训练稳定性修改任务

你现在工作在 `converk/zenith` 仓库的 `V16` 分支；分支名虽为 V16，但当前实际模型/输入契约已经是 V18。V18 的 SFT 和 GRP 已训练完成，本任务只修改 PPO 训练阶段。不要修改 V18 token schema、动作空间、模型主体拓扑，不要重训 SFT/GRP，也不要新增会破坏现有 checkpoint 参数 key/shape 的模块。

先阅读完整 PPO 链路，至少包括：

当前 V18 PPO 实际使用的版本配置文件 *_ppo.yaml（找到 V18 对应配置，例如 v18_ppo.yaml / V18_ppo.yaml，以仓库实际命名为准）
PPO 配置加载和启动入口，确认实际运行时究竟加载哪个 YAML
training/learner.py
training/rollout_buffer.py
training/trajectory.py
training/learner_ddp.py
training/inference.py
model/architecture.py
model/dense_embedding.py
driver / worker / checkpoint / resume / TensorBoard 相关代码

training.yaml 已废弃，不再作为 PPO 正式训练配置使用。不要修改它，也不要根据其中的默认参数判断当前 PPO 行为。
不要只改配置，下面明确指出的实现问题也要修复，并补充相应测试。

## 1. 本次 PPO 的固定训练规模

本次训练与 V17 一样只做 **150 个 update**，每个 update rollout **2048 个半庄**，本次 V18 PPO 正式训练固定为 150 个 update × 每 update 2048 个半庄。
所有 LR、entropy、KL 等 schedule 都必须以 150 updates 为完整训练周期设计。

不要参考废弃的 training.yaml 中可能残留的训练规模或默认参数。

固定训练规模：

* `iterations: 150`
* `total_updates: 150`
* `games_per_update: 2048`
* `critic_bootstrap_updates: 2`
* `update_epochs: 4`
* `minibatch_size: 512`
* `gradient_accumulation_steps: 1`
* `gamma: 1.0`
* `gae_lambda: 0.95`
* `ppo_clip: 0.20`
* 双卡 DDP learner 保持
* GRP reward、合法动作 mask、SFT KL anchor 等核心定义保持不变

V17 PPO 暴露出的主要问题：

1. normalized entropy 在训练前半程基本不下降，甚至略有上升；中后期才开始明显下降，而且最后阶段下降偏快。希望 V18 前期保持较广策略探索，但从较早阶段开始平滑收缩，而不是“前面不动、后面突然坍缩”。其他实验中初期 entropy 大约可以达到后期的 5 倍，但不要把 5 倍作为硬约束。

2. actor grad norm 从前期到后期持续抬升，而 critic grad norm 从很高的位置持续下降。目前看不像 advantage 未归一化导致的梯度爆炸，更像 global grad clipping、critic target 高方差以及 actor/critic/shared 梯度耦合共同造成。

3. 我希望继续利用 **Adam 自身的一阶动量** 降低单个 minibatch 更新带来的随机噪声。这里不是要做真正的多 minibatch 梯度平均，不要通过 gradient accumulation、gradient queue 或自行实现 gradient EMA 来做平均。

4. V17 的 approx KL、clipfrac、ratio 偏移整体都比较小，说明实际策略步长偏保守。V18 可以适当提高有效 actor 更新强度，但必须同时加入合理的 KL guardrail。

5. 麻将属于高随机性、不完全信息、长 horizon 环境，V17 最终 value explained variance 大约 0.10~0.15 已经可以接受。不要为了追求更高 EV 过度强化 critic；重点是降低 critic 噪声以及 critic 对 actor/shared 的干扰。

---

## 2. Global gradient clipping 改成 branch-wise clipping

当前 learner 虽然统计 actor/critic/shared 三组梯度范数，但真正裁剪仍然使用全模型：

`clip_grad_norm_(self.model.parameters(), max_grad_norm)`

这意味着训练前期较大的 critic gradient 会决定全局缩放系数，把 actor 和 shared 的梯度一起缩小；随着 critic 后期逐渐稳定，actor 又突然获得更大的实际更新空间。

这会导致 actor 的 effective LR 在训练前后并不一致，也可能是 V17 entropy 前期长期不动、后期才快速下降的重要原因。

改成 **actor / shared / critic 三组独立 gradient clipping**，直接复用 optimizer 中的参数分组。

新增配置：

```yaml
actor_max_grad_norm: 0.5
shared_max_grad_norm: 0.5
critic_max_grad_norm: 1.0
```

旧的 `max_grad_norm` 可以保留为旧配置 fallback，但新的 V18 PPO 正式配置必须显式使用三组阈值。

要求：

* 只有真正准备执行 `optimizer.step()` 时才进行 clipping。
* 某一分支超过阈值不能缩放另外两个分支。
* 分别记录 actor/shared/critic 的 `pre_clip_norm`、`post_clip_norm`、`clip_scale`。
* 统计每个 update 内各分支发生 clipping 的 optimizer step 比例。
* DDP 下保证裁剪行为和 TensorBoard 指标语义正确。

---

## 3. 限制 privileged critic 对共享 token embedding 的梯度干扰

现有 `critic_public_grad_scale: 0.25` 已经控制 critic loss 从 public hidden 回流 public/shared backbone 的梯度强度。

但是 critic private tokens，例如：

* 对手真实手牌
* 未来牌山信息

仍然经过 actor 共用的 `token_embedding`。

因此 critic private loss 可以绕过 `critic_public_grad_scale`，以完整梯度修改 actor 同样依赖的共享 embedding，以及 embedding 内部共享 MLP。

这次 **不要新增独立 critic embedding 模块**，因为 V18 SFT/GRP checkpoint 已经训练完成，必须保持模型参数拓扑兼容。

采用最小改动，新增：

```yaml
critic_private_embedding_grad_scale: 0.25
```

对：

`critic_embeddings = token_embedding(...)`

产生的 **反向梯度** 做缩放，但 forward 数值保持完全相同。

可以使用与现有 public gradient scaling 相同的形式：

`detached + scale * (x - detached)`

注意：

* 只缩放 critic private embedding 回流共享 `token_embedding` 的梯度。
* `critic_backbone`、`value_head` 自身仍然使用完整 critic gradient。
* 不要直接把整个 critic loss ×0.25。
* bootstrap 阶段继续保持现有 actor/shared 冻结语义。

增加测试验证 `scale=0 / 0.25 / 1` 时 forward 输出完全相同，但共享 embedding 收到的梯度按比例变化。

---

## 4. Critic target 从纯 MC return 改为 GAE λ-return

当前 rollout 已经计算 raw GAE advantage，但 learner 又重新计算纯 Monte-Carlo reward-to-go 作为 critic target。

于是实际变成：

* Actor 使用 GAE(λ=0.95)
* Critic 拟合纯 MC return

在麻将这种高随机性环境中，MC return 会给 critic 引入额外 target variance。

修改成：

`lambda_return = rollout_old_value + raw_gae_advantage`

这里必须使用 **advantage normalization 之前的 raw GAE advantage**。

Actor 继续使用：

`normalized_advantage = (raw_advantage - mean) / (std + eps)`

Critic 则拟合：

`lambda_return = old_value + raw_advantage`

继续保留：

```yaml
value_coef: 0.5
value_loss: huber
value_target_normalization: batch_std
value_target_std_floor: 0.01
```

本次暂时不要额外叠加 value clipping，避免一次修改过多 critic 机制。

为了继续与 V17 对比，不要删除 MC return 计算，可以保留它作为 diagnostic，而不是训练 target。

TensorBoard 至少增加：

* `value_explained_variance_lambda`
* `value_explained_variance_mc`
* `lambda_return_mean/std`
* `mc_return_mean/std`

原来的 `value_explained_variance` 可以改成表示真正训练 target 的 λ-return EV，并在报告中明确说明它与 V17 的旧 EV 定义不同。

不要让 GAE 跨 kyoku。继续保持现有小局结束 `done=True` 的边界；GRP 已经承担跨小局的排名价值 shaping，本次不要扩大 GAE horizon。

---

## 5. Adam momentum：β1 提高到 0.95，但不要做真正梯度平均

我的目标不是实际把多个 minibatch gradient 求平均，而是利用 Adam 的一阶矩：

`m_t = beta1 * m_(t-1) + (1-beta1) * g_t`

让当前 minibatch 的更新方向参考更多之前 minibatch 的梯度方向，以降低单 minibatch 随机性。

正式配置：

```yaml
adam_beta1: 0.95
adam_beta2: 0.999
adam_epsilon: 1.0e-5
gradient_accumulation_steps: 1
```

不要：

* 自己实现 gradient EMA
* 保存历史 gradient queue
* 使用 gradient accumulation 模拟平均
* 在 epoch/update 边界 reset Adam moments

Adam moments 应在 minibatch、epoch、update 之间正常连续存在。

checkpoint/resume 必须精确恢复 optimizer state，包括 Adam 一阶矩和二阶矩。

虽然正式配置固定 `gradient_accumulation_steps=1`，但当前 accumulation 实现顺手修正确：

* 只有 `should_step=True` 时才能 gradient clip。
* 最后不足完整 accumulation group 时，要按照实际累积 minibatch 数正确缩放，而不能永远除完整 `accumulation_steps`。
* 增加对应单测。

但是 **V18 正式 PPO 不启用 accumulation**。

---

## 6. 重新设计 150-update 的 LR schedule

V17 后期 actual policy step 已经明显偏小。本次只有 150 updates，因此不要让 LR 在 u150 降到接近 0。

正式配置：

```yaml
actor_learning_rate: 4.0e-5
actor_learning_rate_min: 1.5e-5

shared_learning_rate: 5.0e-6
shared_learning_rate_min: 2.5e-6

critic_learning_rate: 4.0e-5
critic_learning_rate_min: 1.5e-5

critic_bootstrap_learning_rate: 2.0e-5
warmup_fraction: 0.02
```

前两个 update 仍是 critic bootstrap。

Actor/shared 的 LR schedule 必须按照 **真正的 policy update 数**计算：

* u1/u2：critic bootstrap
* u3：policy update 1

不要让 bootstrap 消耗掉 actor 的 warmup/decay 进度。

三分支保持独立 LR。

PPO 阶段：

```yaml
weight_decay: 0.0
```

原因是这是已经训练好的 SFT representation 上进行的 150-update PPO fine-tuning，没有必要继续使用统一 `0.01` weight decay 对 embedding、norm、backbone 等所有参数长期向 0 拉。

这个修改只针对 PPO，不要修改已有 SFT/GRP optimizer 设置。

---

## 7. Entropy 改成三点分段 schedule

V17 表明 entropy 对 coefficient 的响应明显非线性。简单的 start→end 线性退火容易形成：

前期 coefficient 虽下降，但 entropy 不动
→ 到达某个区间以后 entropy 才开始快速下降。

因此增加三点单调分段 schedule：

* `entropy_start`
* `entropy_middle`
* `entropy_end`
* `entropy_middle_fraction`

同时正式 V18 PPO 使用按合法动作数归一化后的 entropy：

`H_norm = H(policy) / log(max(num_legal_actions, 2))`

原因是不同麻将决策的合法动作数量差异很大。如果直接优化 raw entropy，同一个 entropy coefficient 对 2 个合法动作和 8~10 个合法动作产生的探索压力并不是同一尺度。

保留 raw entropy TensorBoard 指标。

正式配置：

```yaml
entropy_loss_mode: normalized

entropy_start: 0.020
entropy_middle: 0.012
entropy_end: 0.0045
entropy_middle_fraction: 0.33
```

注意：因为 loss 已经从 raw entropy 改成 normalized entropy，因此这些 coefficient **不能直接与 V17 的 raw entropy coefficient 数值比较**。

设计意图：

* 前约 1/3 policy updates，从较强探索压力下降到中等水平。
* 让 entropy 比 V17 更早开始持续下降。
* 后 2/3 再缓慢下降。
* 避免最后几十个 update entropy 下降速度突然明显加快。

为了兼容旧配置，可以支持：

```yaml
entropy_loss_mode: raw
entropy_loss_mode: normalized
```

但新的 V18 正式配置必须显式使用 `normalized`。

不要硬编码最终 entropy 必须等于初始值的 1/5；5 倍只是其他实验中的经验参考。真正关注的是曲线是否平滑，以及后期是否发生 entropy collapse。

---

## 8. 开启 PPO old-policy target KL guardrail

V17 actual approx KL 很低，因此过去 target KL 没有太大作用。

但是 V18 将同时：

* 修复 global clipping
* 提高 actor effective LR
* β1 从 0.9 提高到 0.95

因此需要一个安全阀。

正式配置：

```yaml
target_kl: 0.01
target_kl_check_interval: 8
```

当前实现只在完整 epoch 以后判断 KL，太迟。

修改成：

每 **8 个 optimizer step** 检查一次 sample-weighted approximate KL。

DDP 下对 KL sum/count 做全局聚合，保证所有 rank 同时触发。

如果 KL 超过 0.01：

* 停止当前 PPO update 剩余的 minibatch/epoch
* 正常完成 update 的日志汇总
* checkpoint/resume 状态保持正常
* 不要异常退出整个训练

`target_kl` 只作为 emergency early-stop guardrail，不要变成额外 KL loss。

现有 SFT reference KL U-shaped schedule 暂时保持：

```yaml
sft_kl_coef_start: 0.0025
sft_kl_coef_middle: 0.001
sft_kl_coef_end: 0.002
sft_kl_middle_fraction: 0.5
```

本任务不要顺便重新设计 SFT KL。

---

## 9. 改善 minibatch 随机性，但不要取消 length bucketing

当前 `bucketed_minibatches()` 基本是按照精确 sequence length 排序以后直接每 512 条切成 minibatch。

这样单个 minibatch 内的样本长度非常同质。

而麻将中的 sequence length 很可能和：

* 当前局面阶段
* 河长度
* 副露数量
* 合法动作数量
* 决策类型

存在相关性。

β1 提高以后，如果 minibatch 仍高度同质，momentum 可能连续记住某一类局面的梯度方向，而不只是过滤随机噪声。

但是不要完全取消 length bucketing，否则 V18 长序列 padding 成本可能明显上升。

改成 **coarse/windowed length bucketing**：

1. 按 sequence length 排序。
2. 每 `bucket_window_multiplier × minibatch_size` 条形成一个较大的窗口。
3. 窗口内部随机 shuffle。
4. 再切成 minibatch。
5. 最后再 shuffle minibatch 的执行顺序。

正式配置：

```yaml
bucket_window_multiplier: 8
```

这样仍然保留 padding 优势，但单个 minibatch 不再来自极窄的 exact-length 区间。

DDP 各 rank 不要使用完全相同的 minibatch RNG 序列。

从基础 `shuffle_seed` 派生 rank-specific seed，例如：

`base_seed + rank * 一个固定大质数`

同时保证 checkpoint/resume 后仍可复现。

TensorBoard 保留或补充：

* update padding fraction
* minibatch sequence length mean
* minibatch sequence length std

用于确认随机性改善以后 padding 没有明显恶化。

---

## 10. Rollout 与 update 的 dtype 配置保持一致

检查 `training/inference.py`。

rollout autocast 不应该只根据硬件支持直接写死 BF16，而应该服从：

```yaml
inference_dtype: bf16
```

如果配置 BF16，则 rollout 和 learner update 使用一致的 BF16 策略。

如果未来配置 FP32，也应该两边都切换，而不是：

rollout=BF16
update=FP32

否则即使参数还没有 optimizer step，old/new logprob 也可能产生额外数值差异。

PPO 的：

* logits
* logprob
* ratio
* loss
* KL

这些关键数值计算继续保持现有 FP32 路径。

增加一个小测试或 diagnostic：

参数完全没有更新时，同一 batch 再 forward 一次，old/new logprob 差异只能处于合理数值误差范围，ratio 应非常接近 1。

---

## 11. TensorBoard 监控增强

保留当前已有指标，并至少增加：

### Gradient

* actor/shared/critic pre-clip norm
* actor/shared/critic post-clip norm
* actor/shared/critic clip scale
* actor/shared/critic clip fraction

### Policy

* approx KL
* clipfrac
* ratio mean
* ratio p95
* raw entropy
* normalized entropy
* entropy coefficient
* SFT reference KL

### Value

* raw advantage mean/std
* normalized advantage mean/std
* λ-return mean/std
* MC return mean/std
* `value_explained_variance_lambda`
* `value_explained_variance_mc`
* value loss

### Optimizer

* actor LR
* shared LR
* critic LR

不要为了监控每个 minibatch 做额外 actor/critic 双 backward，也不要默认实现 shared gradient cosine similarity。这里优先保持训练效率。

另外在V18 PPO 的训练阶段和评测阶段补充一套统一的立直麻将业务监控指标，用于判断策略风格变化以及真实性能变化。

要求优先复用同一套指标统计逻辑，不要训练和评测分别实现两套不同口径。所有指标的 TensorBoard / 日志展示名称必须使用中文，代码内部字段名可以继续使用英文。

训练阶段

PPO rollout 为自对弈，因此不要统计一位率、Top2率、四位率、平均顺位等指标，因为全体玩家使用同一策略时这些指标没有实际区分度。

训练阶段重点监控：

和牌率
放铳率
流局率
荒牌流局率
流局听牌率
自摸率
立直率
立直机会接受率
副露率
立直后和牌率
立直后放铳率
平均和牌得点
平均放铳损失
半庄平均小局数
半庄平均决策数
平均每小局决策数
西入率

这些指标建议按 训练/小局、训练/打法、训练/半庄 分类展示，例如：

训练/小局/和牌率
训练/小局/放铳率
训练/打法/立直率
训练/半庄/平均小局数

评测阶段

评测存在明确候选模型，因此除上述小局和打法指标外，还必须增加最终强度指标：

一位率
二位率
三位率
四位率
Top2率
平均顺位
平均最终点数
平均分差
飞人率

建议分类为：

评测/顺位/一位率
评测/顺位/Top2率
评测/顺位/四位率
评测/顺位/平均顺位

同时继续记录候选模型的和牌率、放铳率、立直率、流局率等，用于解释“为什么模型变强或变弱”。

统计口径

统一明确 denominator：

和牌率、放铳率、立直率、副露率：以 player-kyoku 为分母
流局率：以小局数为分母
流局听牌率：以荒牌流局中的 player-kyoku 为分母
Top1/Top2/四位率：以候选模型参与的半庄数为分母
平均小局数：总小局数 / 半庄数

双响、三响必须正确统计，不要假设一个小局最多一个和牌者。

业务结果优先直接从环境/state-machine 的真实终局事件中读取，不要通过 action 或 reward 反推和牌、放铳、流局。

最终确保训练和评测共享统一的 business metrics accumulator，并把所有新增业务指标接入 TensorBoard、评测输出和最终汇总报告。
---

## 12. V18 PPO 最终正式配置

最终配置至少应等价于：(具体内容和格式参考V17的ppo训练配置文件的命名)

```yaml
iterations: 150
total_updates: 150
games_per_update: 2048

gamma: 1.0
gae_lambda: 0.95
ppo_clip: 0.20
update_epochs: 4
minibatch_size: 512
gradient_accumulation_steps: 1

critic_bootstrap_updates: 2
critic_bootstrap_learning_rate: 2.0e-5

actor_learning_rate: 4.0e-5
actor_learning_rate_min: 1.5e-5

shared_learning_rate: 5.0e-6
shared_learning_rate_min: 2.5e-6

critic_learning_rate: 4.0e-5
critic_learning_rate_min: 1.5e-5

warmup_fraction: 0.02

adam_beta1: 0.95
adam_beta2: 0.999
adam_epsilon: 1.0e-5
weight_decay: 0.0

actor_max_grad_norm: 0.5
shared_max_grad_norm: 0.5
critic_max_grad_norm: 1.0

critic_public_grad_scale: 0.25
critic_private_embedding_grad_scale: 0.25

value_coef: 0.5
value_loss: huber
value_target_normalization: batch_std
value_target_std_floor: 0.01

entropy_loss_mode: normalized
entropy_start: 0.020
entropy_middle: 0.012
entropy_end: 0.0045
entropy_middle_fraction: 0.33

target_kl: 0.01
target_kl_check_interval: 8

sft_kl_coef_start: 0.0025
sft_kl_coef_middle: 0.001
sft_kl_coef_end: 0.002
sft_kl_middle_fraction: 0.5

bucket_window_multiplier: 8

inference_dtype: bf16

checkpoint_interval_updates: 10
```

如果项目现有配置命名方式不同，可以按照现有工程风格修改字段名称，但语义必须保持一致。

---

## 13. 测试、兼容性和最终交付

必须保证：

1. 已训练的 V18 SFT/GRP checkpoint 可以直接用于新的 PPO。
2. 不改变模型参数拓扑。
3. 不修改 V18 token schema、动作空间或 critic private information 定义。
4. 不修改 GRP reward 定义。
5. 单卡和双卡 DDP 都可运行。
6. exact resume 能恢复 Adam moments、RNG、iteration 和 schedule 正确位置。
7. 不使用 gradient accumulation 替代 Adam momentum。
8. 不加入新的 Q-learning、Q-boosting 等算法。
9. 新配置尽量兼容旧配置 fallback。

至少补充测试覆盖：

* raw GAE + old value 正确生成 λ-return。
* advantage normalization 不污染 critic target。
* branch-wise clipping 中一个分支超阈值不会缩放其他分支。
* critic private embedding grad scale forward 不变、backward 按比例变化。
* entropy start/middle/end schedule 精确命中。
* normalized entropy 对不同 legal action count 计算正确。
* target KL 在单卡/DDP 下同步 early stop。
* accumulation>1 时只在 optimizer step 前 clip，并正确处理尾组。
* coarse/windowed bucketing 可复现。
* checkpoint/resume 后 Adam moments 与 LR/entropy schedule 连续。

完成后运行现有：

`riichi_ppo_v1/tests`

以及所有新增测试。

如果运行环境允许，再执行一个极短 PPO smoke test，确认：

* policy/value loss 无 NaN/Inf
* gradient 无 NaN/Inf
* entropy 正常
* KL 正常
* DDP 不死锁
* checkpoint 可以保存和恢复

最后在：

`audit/reports/v18/`

新增一份简洁的 PPO stability 修改报告，说明：

* 修改了哪些文件
* 每项修改解决的问题
* 最终 PPO 参数
* V17 与新 V18 PPO 的核心行为差异
* 测试结果
* 当前仍然存在的风险

请直接完成代码修改，不要只输出建议。

如果实现过程中发现本文描述与当前 V18 代码存在小差异，以现行代码为准做 **最小兼容修改**。

如果发现某项修改会破坏已有 V18 SFT/GRP checkpoint 拓扑、改变 reward 定义或明显改变 PPO 核心语义，不要自行扩大修改范围；保留兼容实现，并在最终报告中明确说明。
