# Goal：Step 3 — 200K Counterfactual Teacher Dataset + PPO-Attached Top3 Reranker + Paired Self-Play ROI Validation

## 一、最终目标

当前项目已经完成：

```text
Step 1
GRP V2 kyoku-ending long-term value validation
→ PASS / confidence-aware usable

Step 2
TopK paired counterfactual rollout
→ current kyoku end
→ GRP V2 Teacher
→ CONDITIONAL GO

Step 2.5
High-budget full-hanchan Monte Carlo audit
→ Teacher signal 与真实 full-hanchan utility 存在正相关
→ Teacher 可以作为有 uncertainty 的监督信号
→ CONDITIONAL GO
```

现在不再继续研究固定 Teacher threshold。

本阶段直接回答整个 Reranking 方向最重要的问题：

> **将 paired rollout + GRP V2 Teacher 的知识蒸馏到一个非常轻量的 PPO Reranker 挂件后，是否真的能够在完整半庄中优于原始 PPO？**

完整实验路线：

```text
Deployment PPO
     ↓
PPO Top3 candidates
     ↓
paired hidden-world rollout
     ↓
current kyoku end
     ↓
GRP V2
     ↓
200K Teacher-labelled decisions
     ↓
Train / Val / Test
     ↓
轻量 PPO-attached Top3 Reranker
     ↓
Offline Ranking Evaluation
     ↓
300 paired full-hanchan self-play
     ↓
PPO vs PPO + Reranker
```

本阶段最终判决权属于：

```text
完整半庄 paired self-play
```

而不是单纯的 validation loss 或 ranking accuracy。

---

## 二、本阶段核心研究问题

必须明确回答：

1. Counterfactual Teacher 的相对价值信息能否被一个小型 Reranker 学习？
2. Reranker 是否能够学习什么时候保留 PPO Top1、什么时候修正 PPO Top1？
3. Reranker 是否不会因为训练数据中的 hard cases 而产生过度 override？
4. Reranker 是否能在独立 Test Set 上降低相对 Teacher 的 expected regret？
5. 最关键：
   **PPO + Reranker 是否能在完整半庄中获得比原 PPO 更好的最终 ranking/reward？**
6. 如果完整半庄提升很小，则是否说明：
   **当前强 PPO 上真正可利用的 Top3 reranking headroom 本身已经很小？**

---

## 三、本阶段明确不做

不要：

* 重训 GRP；
* 重做 Step 1；
* 重做 Step 2；
* 重做 Step 2.5；
* 使用 full-hanchan MC 生成 200K Teacher labels；
* 人为确定固定 `|ΔGRP| >= x` online gate；
* 开发 MCTS；
* 扩展到 Top10；
* 第一版使用 Top4；
* 训练新的 PPO；
* 修改 PPO policy 参数；
* 使用 PPO critic/value head 训练 Reranker；
* 使用 critic private information；
* 使用对手真实隐藏手牌作为 Reranker 输入；
* 用 Expert Action 作为 Teacher ground truth；
* 因 offline 指标好看而跳过最终 self-play。

---

## 四、Deployment PPO

Candidate Policy、在线基线和 continuation policy 优先统一使用：

```text
checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt
```

如果当前仓库存在明确更新、且已经成为正式部署版本的 PPO checkpoint，可以检查实际项目配置后使用最新部署 checkpoint。

但必须在实验报告中明确记录最终 checkpoint。

同一 PPO 用于：

```text
1. Top3 candidate generation
2. policy logits / probabilities
3. public hidden feature extraction
4. Teacher rollout continuation
5. self-play baseline
6. self-play opponents
```

PPO 在 Reranker 训练过程中：

```text
完全冻结
```

不得被 Reranker loss 更新。

---

## 五、当前 PPO 网络结构：必须先确认后复用

当前 PPO v2 使用：

```text
isolated_action_query
```

已知核心数据流如下：

```text
token_factors
[B,T,10]
    ↓
token_embedding
[B,T,192]
    ↓
public_backbone
3-layer shared Transformer
    ↓
public_sequence
[B,T,192]
```

然后 Actor：

```text
public_sequence
    ↓
actor_backbone
1-layer actor-only Transformer
    ↓
各 action query hidden
    ↓
policy_head
    ↓
241 action logits
```

Critic 当前则是：

```text
public_sequence
    ↓
state_mask
    ↓
packed_public
[B,S,192]

+
critic private tokens
+
value_query
    ↓
critic_backbone
2 layers
    ↓
value_query hidden
    ↓
value_head
```

本阶段 Reranker **仿照 Critic 的 token injection / query readout 思路**，但：

```text
绝对不使用 critic private tokens
绝对不使用 value_query
绝对不使用 value_head
绝对不使用 critic_backbone
```

Reranker 是一条全新的独立 branch。

---

## 六、Reranker 的本质

Reranker 是：

```text
Frozen PPO
+
Trainable lightweight reranker branch
```

它是 PPO 的：

```text
外挂 / attached module
```

但参数独立保存。

推荐 checkpoint：

```text
ppo_checkpoint.pt
reranker_checkpoint.pt
```

运行时可以：

```text
reranker disabled
→ 原 PPO

reranker enabled
→ PPO + Reranker
```

以保证严格 A/B Test。

---

## 七、Candidate 固定 Top3

对于每个合法决策状态：

```text
A = PPO Top1
B = PPO Top2
C = PPO Top3
```

第一版 Reranker 不使用 Top4。

原因：

```text
Expert Recall@3 ≈ 98%
```

已经具有较高 candidate coverage。

Step 2.5 的 Rank4 override：

```text
n = 10
SUPPORTED = 0
REJECTED = 0
UNRESOLVED = 10
mean ΔFull ≈ 0.02
```

当前没有足够证据证明 Rank4 的额外 Recall 能稳定转化为长期收益。

第一版先验证 Top3 Reranking 本身是否具有 ROI。

---

## 八、数据集规模提高到 200K

最终必须得到：

```text
200,000 valid Teacher-labelled decisions
```

这里 valid 意味着：

* PPO Top3 合法；
* 状态能够正确恢复；
* Teacher rollout 正常结束；
* GRP 输出有限；
* 所需 policy / state / Teacher features 完整；
* 没有重复 decision；
* 没有损坏 shard；
* 能够进入训练数据。

失败数据不能计入 200K。

---

## 九、Train / Validation / Test

比例保持：

```text
80% / 10% / 10%
```

因此：

```text
Train = 160,000
Val   =  20,000
Test  =  20,000
```

---

## 十、禁止数据泄漏

绝对禁止直接：

```text
random split decisions
```

因为同一 hanchan 中相邻状态高度相关。

必须按照：

```text
source hanchan / game / trajectory
```

进行 group split。

即：

```text
Train source games
Val source games
Test source games
```

三者互不重叠。

必须验证：

```text
train_game_ids ∩ val_game_ids  = ∅
train_game_ids ∩ test_game_ids = ∅
val_game_ids   ∩ test_game_ids = ∅
```

最终报告必须输出 leakage audit。

---

## 十一、数据分布

Reranker 最终部署在 PPO 自己的决策分布上。

因此数据必须主要来自：

```text
PPO-representative state distribution
```

如果仓库中已有足够 PPO trajectory 数据，可以复用。

否则通过 PPO self-play 建立 state pool。

---

## 十二、Training Sampling Strategy

Teacher rollout 很昂贵，因此 Train Set 可以适度 oversample hard decisions。

建议：

```text
约 50% natural PPO distribution
约 50% hard/medium oversampling
```

hardness 可以在 Teacher 之前依据：

```text
Top1 probability
Top1-Top2 probability gap
policy entropy
```

决定。

不要根据最终 Teacher winner 决定是否采样。

避免 label-selection bias。

---

## 十三、Validation / Test Distribution

Val 与 Test 尽量保持：

```text
自然 PPO deployment distribution
```

尤其：

```text
20K Test
```

必须作为真实自然状态下的最终 offline benchmark。

不要人为大量 oversample hard states。

---

## 十四、Teacher Label Generation

对于：

```text
A = Top1
B = Top2
C = Top3
```

采样 hidden worlds：

```text
W1 ... WN
```

每个 Wi：

```text
Wi
├── force A
├── force B
└── force C
```

三个分支必须共享：

```text
同一个 sampled hidden world
```

然后全部使用：

```text
PPO greedy / argmax
```

继续到：

```text
当前 kyoku 结束
```

得到 kyoku-ending state。

使用：

```text
GRP V2
```

计算 expected final-hanchan utility。

得到：

```text
Q_A
Q_B
Q_C
```

以及：

```text
ΔBA = Q_B - Q_A
ΔCA = Q_C - Q_A
ΔCB = Q_C - Q_B
```

---

## 十五、Teacher World Budget

本阶段是大规模 supervised dataset generation。

不要使用 Step 2.5 的 full-hanchan 256-world MC。

建议首先 benchmark：

```text
16 paired worlds
```

如果 variance / stability 明显不足，可使用：

```text
32 paired worlds
```

允许在开始 200K 正式生成之前，用少量 decision 对：

```text
N=16
vs
N=32
```

做一次 throughput + label stability sanity check。

目的只是选训练数据成本，而不是重新验证 Teacher。

之后固定方案。

不要无限 adaptive sampling 直到每条数据显著。

Teacher uncertainty 应保留为训练信息。

---

## 十六、Hidden-World Sampler

优先复用 Step 2 已经审计过的 baseline sampler：

```text
根据当前玩家 public information
重建未知牌池
→ uniform random assignment
→ opponents hidden hands + wall
```

必须保持：

* 牌种守恒；
* 手牌数合法；
* dora slot 正确；
* MJAI event stream 一致；
* PPO observation consistency；
* candidate action legal。

继续明确记录 limitation：

```text
不包含基于弃牌 / 立直 / 副露 / 手切摸切的 posterior belief inference
```

但本 Goal 不开发新的 belief model。

---

## 十七、训练样本必须保存 PPO public token hidden

当前 PPO：

```text
public_sequence = public_backbone(...)
```

形状：

```text
[B,T,192]
```

Reranker 只使用：

```text
packed_public =
public_sequence[row, state_mask[row]]
```

即：

```text
[B,S,192]
```

其中：

```text
state_mask
```

负责去除每个 action 的：

```text
offense query
defense query
及其它 action query token
```

只保留真正描述游戏 public state 的 token hidden。

这些 hidden 是 PPO shared 3-layer Transformer 已经编码过的逐-token representation。

---

## 十八、不要压成单个 state vector

本阶段禁止把：

```text
packed_public [S,192]
```

简单：

* mean pooling；
* sum pooling；
* 只取一个 global token；
* flatten 到巨大 FC；
* 压成单个 PPO global state embedding；

然后再交给 Reranker。

原因：

Reranker 应保留 PPO shared backbone 已经学习到的：

```text
逐 token 局面结构
```

并让 Reranker 自己针对不同 candidate 决定关注哪些 token。

---

## 十九、Reranker State Projection

Reranker hidden dimension：

```text
128
```

因此对每个 public token 做共享投影：

```text
Linear(192 → 128)
```

得到：

```text
rerank_state_tokens
[B,S,128]
```

该 projection 属于 Reranker 参数：

```text
可训练
```

PPO public hidden：

```text
detach
```

PPO shared backbone：

```text
冻结
```

---

## 二十、Candidate Token

对每个 candidate action i：

```text
i ∈ {A,B,C}
```

建立：

```text
candidate_token_i
```

Candidate information 至少包括：

### Action identity

```text
action_id
→ Embedding(241, 32)
```

### Candidate-specific PPO features

```text
policy probability_i
log probability_i
raw policy logit_i
policy rank_i
probability gap to Top1
logit gap to Top1
```

### Global Top3 Policy features

```text
p1
p2
p3

logp1
logp2
logp3

p1-p2
p2-p3
p1-p3

logit1-logit2
logit2-logit3
logit1-logit3

legal-action entropy
Top3 cumulative probability
legal action count
```

---

## 二十一、Explicit Strategic Context

尽管 public hidden 已经包含大量状态信息，但为了帮助小型 Reranker 学习半庄战略，可以额外提供低成本显式 context：

```text
kyoku index
round / wind
honba
riichi sticks
self seat
dealer seat

four player scores
current self rank

score gap to rank1
score gap to rank2
score gap to rank3
score gap to rank4
```

根据实际 seat/rank 对无意义 gap 做一致编码。

所有连续值必须合理 normalization。

---

## 二十二、Candidate Feature Encoder

将：

```text
Action embedding
+
candidate PPO features
+
global PPO features
+
explicit strategic context
```

拼接后经过一个小型：

```text
Candidate Feature MLP
```

输出：

```text
candidate_token_i ∈ R^128
```

建议：

```text
input
→ Linear(...,128)
→ GELU
→ Linear(128,128)
```

不要做得过深。

---

## 二十三、Rerank Query

建立一个独立可训练：

```text
rerank_query ∈ R^128
```

作用类似 PPO Critic 中的：

```text
value_query
```

但语义不同。

`rerank_query` 的任务是：

> 通过 candidate-conditioned attention 汇聚整条 public state sequence，输出“当前 candidate 在这个 state 中应该获得怎样的 ranking correction”的 readout representation。

---

## 二十四、单 Candidate Reranker Sequence

对于一个 candidate i：

```text
rerank_sequence_i =
[
    state_token_1
    state_token_2
    ...
    state_token_S
    candidate_token_i
    rerank_query
]
```

形状：

```text
[S+2,128]
```

然后输入：

```text
reranker_backbone
```

---

## 二十五、Reranker Backbone

使用非常轻量的：

```text
1-layer Transformer
```

配置：

```text
d_model = 128
layers = 1
heads = 4
FFN hidden = 256
dropout = 0.1
```

优先复用当前 PPO decoder/block 实现风格，以降低工程风险。

如果使用 causal attention：

```text
rerank_query
```

必须位于最后，从而可以看到：

```text
全部 state tokens
+
candidate token
```

---

## 二十六、Candidate-Conditioned Attention

这个结构的核心能力是：

```text
(State, Action A)
(State, Action B)
(State, Action C)
```

可以对同一 public token sequence 形成不同 readout。

例如：

```text
打某张万子
```

可能更关注相关：

```text
手牌结构
万子河
对手副露
```

而另一个 candidate 可能关注不同 state token。

不要强制所有 candidate 共用一个固定 pooled state vector。

---

## 二十七、Top3 Candidate 必须共享 Reranker 参数

逻辑上分别计算：

```text
(State,A) → r_A
(State,B) → r_B
(State,C) → r_C
```

三者：

```text
使用完全相同的：
state projection
candidate encoder
reranker Transformer
reranker head
```

不得给 Top1 / Top2 / Top3 建独立网络。

Policy rank 作为 feature 输入。

---

## 二十八、不要把 A/B/C 一次塞进 causal sequence

第一版不要：

```text
state
A
B
C
query
```

因为 causal ordering 会造成 candidate order bias。

第一版使用标准共享 scorer 思路：

```text
(State,A)
(State,B)
(State,C)
```

分别评分。

训练时再将三个 score 放在一起计算 pairwise/listwise objective。

---

## 二十九、批量实现

虽然逻辑上三个 candidate 分别评分，但实现时必须批量。

例如：

```text
packed_state
[B,S,128]
```

expand 到：

```text
[B*3,S,128]
```

candidate token：

```text
[B*3,1,128]
```

query：

```text
[B*3,1,128]
```

合并：

```text
[B*3,S+2,128]
```

一次 batch 运行 reranker。

不要做三个 Python-level 独立 forward。

---

## 三十、Reranker Readout

Reranker Transformer 输出：

```text
reranker_hidden
```

只取：

```text
rerank_query position
```

得到：

```text
h_i ∈ R^128
```

然后共享：

```text
Reranker Head
128 → 64 → 1
```

输出：

```text
r_i
```

---

## 三十一、Residual Ranking

最终 score：

```text
S_i = log π_i + α * r_i
```

第一版可以：

```text
α = 1
```

也可以将 α 作为固定 scale 超参数通过 Validation 选择。

不要在 Test 上调 α。

Residual head 最后一层建议近零初始化，使训练开始时：

```text
r_i ≈ 0
```

因此：

```text
S_i ≈ log π_i
```

初始 Reranker 行为基本等价于 PPO。

这符合：

```text
PPO 是强 prior
Reranker 是低频 error corrector
```

的设计目标。

---

## 三十二、Reranker 参数必须与 PPO 解耦

训练时：

```text
public_sequence = PPO(...).detach()
```

或者在 feature extraction 阶段提前缓存。

不得通过 Reranker loss 更新：

```text
token_embedding
public_backbone
actor_backbone
policy_head
critic_backbone
value_head
```

只训练：

```text
state_projection
action_embedding
candidate_feature_mlp
rerank_query
reranker_backbone
reranker_head
```

---

## 三十三、不使用 PPO Value Branch

当前训练已经不再是 PPO。

因此 Reranker：

```text
不使用 value_head
不使用 critic_hidden
不使用 critic_private
不使用 critic_backbone
```

Teacher supervision 完全来自：

```text
paired rollout
→ kyoku end
→ GRP V2
```

这是一个：

```text
offline supervised Learning-to-Rank / Teacher Distillation
```

任务。

---

## 三十四、模型参数量

目标新增参数控制在：

```text
约 0.2M ~ 0.4M
```

具体根据：

```text
candidate feature input dimension
Transformer block 实现
```

计算真实参数。

必须在训练开始前输出：

```text
total PPO parameters
trainable Reranker parameters
frozen PPO parameters
```

确认：

```text
只有 Reranker 参数 requires_grad=True
```

---

## 三十五、Teacher Target

每个 decision 保留：

```text
Q_A
Q_B
Q_C

ΔBA
ΔCA
ΔCB

SE_BA
SE_CA
SE_CB

world count
```

不要只保留：

```text
Teacher Best
```

---

## 三十六、Pairwise Soft Target

对于 pair：

```text
i,j
```

定义 Teacher：

```text
ΔQ_ij = Q_i - Q_j
SE_ij
```

计算：

```text
z_ij = ΔQ_ij / (SE_ij + eps)
```

然后：

```text
p_teacher(i>j) = Φ(z_ij)
```

Student：

```text
p_student(i>j)
=
sigmoid(S_i - S_j)
```

使用 soft BCE：

```text
L_pairwise
```

Top3 一共：

```text
3 pairs
```

---

## 三十七、Near-Tie 不应成为硬错误

如果：

```text
ΔQ ≈ 0
SE 较大
```

则：

```text
p_teacher ≈ 0.5
```

Student 不应该被强迫学习一个任意严格排序。

这样 Teacher uncertainty 自然进入监督。

不要使用固定：

```text
|ΔQ| >= threshold
```

来筛掉所有弱样本。

---

## 三十八、Loss

第一版：

```text
L =
L_pairwise
+
λ_residual * mean(r_i²)
```

Residual regularization：

```text
λ_residual
```

可以从：

```text
1e-3 ~ 1e-2
```

范围选择。

目的：

```text
避免无意义大 residual
避免过度 override PPO
```

---

## 三十九、可选 Auxiliary Loss

如果实现非常简单，可加入：

```text
relative-value regression
```

例如：

```text
(S_i - S_j) ≈ scaled ΔQ_ij
```

但：

```text
Pairwise objective 是第一版主任务
```

不要因为复杂 auxiliary loss 阻塞 MVP。

---

## 四十、训练设置

数据：

```text
Train = 160K
Val   = 20K
Test  = 20K
```

建议：

```text
max_epochs = 10
minimum_epochs = 3
early_stopping_patience = 3
```

> 注：`max_epochs` 按用户要求由 20 调整为 10；不预先固定一定训练 3、5 或 10 epoch，由 Validation 自动决定 best checkpoint。

---

## 四十一、Optimizer

根据项目现有训练代码选择合理 optimizer。

建议默认：

```text
AdamW
```

初始学习率可以从：

```text
1e-3
3e-4
```

等合理范围小规模尝试。

由于模型非常小：

```text
不要为了搜索 learning rate 做大型 hyperparameter sweep
```

优先简单可靠。

---

## 四十二、Validation

每个 epoch 完成后必须在完整 20K Val 上评估。

至少输出：

```text
val pairwise soft loss

pairwise accuracy
(confident pairs subset)

Teacher best agreement

Policy baseline expected Teacher regret
Reranker expected Teacher regret

override rate

override precision vs Teacher

damage rate

policy preservation rate
```

---

## 四十三、Best Checkpoint Selection

不能简单：

```text
minimum val loss
```

作为唯一目标。

优先综合：

```text
Expected Teacher Regret ↓
Override Precision ↑
Damage Rate ↓
不过度 Override
```

选择 best checkpoint。

选择规则必须在 Test 前固定。

---

## 四十四、Test Set

所有：

```text
architecture
epoch
checkpoint
loss configuration
α
```

确定后：

```text
Test 20K
```

只做一次最终 offline test。

输出：

```text
PPO baseline expected Teacher regret
Reranker expected Teacher regret

PPO Teacher-best agreement
Reranker Teacher-best agreement

override rate
override precision
damage rate

Policy wrong → Reranker correct
Policy correct → Reranker wrong
```

并按：

```text
policy gap
PPO Top1 probability
kyoku stage
current rank
Teacher confidence
```

分桶。

---

## 四十五、Teacher Dataset Generation 的性能是一级任务

200K 数据的计算成本很高。

之前 Step 2.5 已明确观察到：

```text
GPU utilization 很低
CPU / environment simulation 是主要瓶颈
```

所以本 Goal 必须高度重视：

```text
并发 sampling 性能
```

---

## 四十六、CPU / Python Process 硬上限

允许最多使用：

```text
24 CPU cores
```

以及：

```text
最多 24 个 Python processes
```

任何情况下：

```text
CPU cores <= 24
Python worker processes <= 24
```

禁止超过。

---

## 四十七、避免线程 Oversubscription

每个 worker 原则上限制到：

```text
1 CPU thread
```

必须合理设置：

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

PyTorch 视实际实现设置：

```python
torch.set_num_threads(1)
```

避免：

```text
24 workers × 多个 BLAS/OpenMP threads
```

导致性能反而下降。

---

## 四十八、正式生成前必须 Benchmark Worker 数

不要默认：

```text
24 workers 就最快
```

先做固定小 workload：

```text
4 workers
8 workers
12 workers
16 workers
20 workers
24 workers
```

比较：

```text
valid Teacher decisions / second
world rollouts / second
environment steps / second
policy decisions / second

CPU utilization
GPU utilization
GPU batch size
queue waiting
RAM
```

然后选择实际吞吐最高且稳定的配置。

但永远不得超过 24 cores / 24 processes。

---

## 四十九、尽量 Batch GPU Inference

禁止：

```text
一个 env step
→ 一个 tiny GPU forward
```

尽量继续使用：

```text
multiple active trajectories
+
batched PPO inference
```

如果多个 Python worker 同时访问 GPU 导致 contention，则评估：

```text
CPU simulation workers
       ↓
shared inference queue
       ↓
centralized / limited GPU inference process
```

是否更快。

不要无依据重构。

必须通过 benchmark 数据决定。

---

## 五十、200K Dataset 必须支持 Resume

数据生成属于长时间任务。

必须：

```text
sharded
incremental writing
atomic completion markers
resume
dedup
failure retry
```

如果任务中断：

```text
继续已有结果
```

而不是从零开始。

---

## 五十一、建议 Shard

可以根据实际吞吐使用类似：

```text
1K ~ 5K valid decisions / shard
```

不要产生一个超大单文件。

每 shard 至少包含：

```text
sample count
source games
worker / seed
Teacher config
PPO checkpoint
GRP checkpoint
world budget
checksum / completion marker
```

---

## 五十二、PPO Public Hidden 的存储策略

需要 benchmark 两种方式：

### Option A

保存原始 state/input factors。

训练时每个 batch 重新通过 Frozen PPO public backbone。

### Option B

Dataset generation 时直接保存：

```text
packed_public [S,192]
```

或者可压缩表示。

训练时直接加载 frozen features。

由于 PPO 永远冻结：

```text
Option B 可能显著提升训练速度
```

但 200K × ~100 × 192 的 float32 数据可能占用很大磁盘。

因此必须估算：

```text
disk size
I/O throughput
training speed
```

可考虑：

```text
FP16 / BF16 feature storage
```

只要数值 sanity check 表明不会破坏训练。

通过 benchmark 决定，不要盲目保存巨大 features。

---

## 五十三、每 10 分钟报告进度

Agent 在所有长时间阶段必须每约：

```text
10 分钟
```

输出一次可见状态。

包括：

```text
Teacher dataset generation
Reranker training
Offline evaluation
300 paired self-play
```

---

## 五十四、进度格式

至少包含：

```text
[PROGRESS]

phase:
completed / total:
percent:
elapsed:
recent throughput:
moving-average throughput:
estimated remaining time:
active workers:
failed jobs:
retried jobs:
CPU utilization:
GPU utilization:
memory:
current output shard:
```

其中：

```text
estimated remaining time
```

必须根据近期 throughput 动态估计。

用户要求：

> **每 10 分钟说明一次当前进度和预计剩余时间。**

不得长时间静默执行。

如果某阶段少于 10 分钟，则阶段结束时报告即可。

---

## 五十五、训练进度

训练每 epoch 至少输出：

```text
epoch
train loss
val loss
val expected regret
val override rate
val override precision
val damage rate
learning rate
best epoch
early-stop counter
```

如果单 epoch 超过 10 分钟：

```text
epoch 中间仍需每约 10 分钟输出 ETA
```

---

## 五十六、最终 Self-Play ROI Experiment

完成 offline test 后执行：

```text
300 paired hanchans
```

具体：

```text
300 unique seeds
×
2 matched tables
=
600 total hanchans
```

---

## 五十七、每一个 Seed 同时建立两桌

对于 seed `s_k`：

## Table A：Baseline

目标座位：

```text
PPO
```

另外三家：

```text
PPO
PPO
PPO
```

## Table B：Treatment

相同目标座位：

```text
PPO + Reranker
```

另外三家仍然：

```text
PPO
PPO
PPO
```

---

## 五十八、两桌唯一实验变量

必须保证：

```text
same seed
same target seat
same opponent checkpoint
same PPO checkpoint
same environment config
same game rules
```

唯一变化：

```text
target seat 是否启用 Reranker
```

---

## 五十九、Seed Pairing

对于每一个 pair：

```text
Table A seed == Table B seed
```

两桌最好作为一个 paired job 同时推进和记录。

因为 Reranker 可能改变 action：

```text
后续游戏 trajectory 会自然分叉
```

因此不要声称：

```text
后续所有 wall / random events 都严格相同
```

但初始：

```text
environment seed
shuffle seed
seat
checkpoint
```

必须完全匹配。

---

## 六十、Seat Balance

300 pairs：

```text
75 pairs seat0
75 pairs seat1
75 pairs seat2
75 pairs seat3
```

必须严格平衡。

可直接：

```text
target_seat = pair_id % 4
```

seed schedule 必须提前生成并保存。

---

## 六十一、Self-Play 并发

Self-play 同样最多：

```text
24 CPU cores
24 Python processes
```

推荐：

```text
1 paired job
=
same seed Table A + Table B
```

由同一个 worker 管理两个 env 或作为明确关联 job。

先做小 benchmark 决定并发数。

---

## 六十二、Primary Self-Play Metric

每一个 seed pair：

```text
R_base,k
R_rerank,k
```

reward：

```text
1st = +10
2nd = +4
3rd = -4
4th = -10
```

paired difference：

```text
D_k = R_rerank,k - R_base,k
```

Primary：

```text
mean(D_k)
```

必须报告：

```text
95% paired bootstrap CI
```

或者合理 paired CI。

统计单位是：

```text
hanchan seed pair
```

而不是 action。

---

## 六十三、Secondary Self-Play Metrics

同时报告：

```text
average rank
average final score / pt

1st rate
2nd rate
3rd rate
4th rate

mean reward
```

比较：

```text
PPO
vs
PPO + Reranker
```

---

## 六十四、Reranker Online Behavior

必须记录：

```text
total target decisions

reranker evaluated decisions

override count
override rate

Top1 → Top2 count
Top1 → Top3 count

overrides per hanchan
```

以及按：

```text
policy gap
Top1 probability
kyoku
current rank
score situation
```

统计 override。

---

## 六十五、Residual 行为

每次 online decision 最好记录：

```text
logπ_A/logπ_B/logπ_C

r_A/r_B/r_C

final S_A/S_B/S_C

PPO margin
Reranker final margin
```

用于判断 Student 实际是：

```text
小修正
```

还是：

```text
大幅覆盖 PPO prior
```

---

## 六十六、Per-Hanchan Record

每一个 pair 保存：

```text
seed
target seat

base final score
reranker final score

base rank
reranker rank

base reward
reranker reward

paired reward difference

reranker decision count
override count
Top1→Top2
Top1→Top3
```

---

## 六十七、300 Pairs 固定，不得 Optional Stopping

本 Goal 的正式主实验就是：

```text
300 pairs
```

不能：

```text
结果显著就提前停止
结果不显著就擅自继续加样本
```

避免 optional stopping。

如果 CI 跨 0：

```text
诚实报告 INCONCLUSIVE
```

是否扩大到 1000+ pairs 由下一 Goal 决定。

---

## 六十八、最终必须回答的 Q1–Q10

## Q1

是否成功生成：

```text
200K valid decisions
```

并完成：

```text
160K / 20K / 20K
```

无 leakage split？

## Q2

Reranker 在 Test Set 上是否降低 Teacher expected regret？

## Q3

Reranker 是否提高 Teacher preference agreement？

## Q4

Reranker override rate 是多少？

是否保持低频 correction 特性？

## Q5

Override precision 是否足够高？

## Q6

Damage rate 是否足够低？

## Q7

300 paired self-play：

```text
mean ΔReward
```

是多少？

95% CI 是多少？

## Q8

Average rank 是否改善？

## Q9

1st / 4th rate 是否改善？

## Q10

整个 Reranking 方向是否真正具有足够大的 end-to-end ROI？

---

## 六十九、最终结论

必须给：

```text
GO
CONDITIONAL GO
NO-GO
```

## GO

如果：

```text
Offline:
Reranker 明显降低 Teacher regret
且没有明显 over-override

AND

Self-play:
PPO + Reranker 相对 PPO
表现出有意义且方向稳定的完整半庄收益提升
```

则：

```text
GO
```

可以继续：

* 更大数据；
* Top4 ablation；
* loss refinement；
* harder sampling；
* 更大 self-play；
* uncertainty-aware gate；
* architecture ablation。

## CONDITIONAL GO

如果：

```text
Offline 明显有效
Self-play 方向正
但 300 pairs CI 较宽
```

则：

```text
CONDITIONAL GO
```

下一阶段扩大 paired self-play 样本。

## NO-GO

如果：

```text
Student 无法学习 Teacher
```

或者：

```text
Student offline 学得很好
但完整 self-play 无收益甚至退化
```

则：

```text
NO-GO
```

重点考虑：

> 当前强 PPO 上 Top3 reranking headroom 是否本身就太小。

不要立即通过堆更大模型掩盖这个问题。

---

## 七十、最终输出文件

至少生成：

```text
sampling_benchmark.csv

teacher_dataset_manifest.json
teacher_dataset_shards/

train_game_ids.txt
val_game_ids.txt
test_game_ids.txt

train_ids.txt
val_ids.txt
test_ids.txt

reranker_config.json
training_history.csv
best_reranker.pt

offline_val_report.csv
offline_test_report.csv

selfplay_seed_schedule.csv
selfplay_pair_results.csv
selfplay_reranker_actions.csv
selfplay_summary.json

STEP3_REPORT.md
```

---

## 七十一、最终 STEP3_REPORT 开头必须给出

```text
Dataset
=======
Valid decisions:
Train:
Val:
Test:
Source leakage:

Sampling
========
Teacher worlds:
Workers:
CPU cores:
Teacher decisions/s:
Total rollout count:

Reranker
========
Architecture:
Trainable parameters:
Frozen PPO parameters:
Best epoch:

Offline Test
============
PPO expected Teacher regret:
Reranker expected Teacher regret:

Reranker override rate:
Override precision:
Damage rate:

Paired Self-Play
================
Pairs: 300

Base mean reward:
Reranker mean reward:
Mean paired ΔReward:
95% CI:

Base average rank:
Reranker average rank:
ΔRank:

Base 1st rate:
Reranker 1st rate:

Base 4th rate:
Reranker 4th rate:

Reranker Behavior
=================
Total evaluated decisions:
Overrides:
Override rate:
Overrides per hanchan:

Top1→Top2:
Top1→Top3:

Final
=====
GO / CONDITIONAL GO / NO-GO
```

---

## 七十二、最重要的执行原则

本 Goal 不追求：

```text
复杂 Reranker
最高 offline accuracy
最大 override rate
```

而追求：

```text
Frozen strong PPO
+
约 0.2M~0.4M lightweight reranking attachment
+
Teacher distillation
+
最终完整半庄 ROI
```

Reranker 必须被理解为：

> **基于 PPO 已经编码好的完整 public state token sequence，对 Top3 action 进行 candidate-conditioned correction 的小型外挂。**

而不是：

> 第二个 Policy。

最终实验真正要回答的是：

> **当前 PPO 已经很强的情况下，这种低频 Top3 correction 是否仍然能够带来足够大的完整半庄收益。**

如果答案是否定的，应接受：

> Reranking headroom 可能本身有限。

如果答案是肯定的，再继续投入后续优化。

---

# 附 A：本 Goal 模型与运行环境（2026-08-12）

## 模型路径

```text
Deployment PPO（候选生成 / 在线基线 / continuation / self-play 双方）：
  checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt

  - PPO v2，policy_head_type = isolated_action_query
  - layers=4（shared_layers=3，actor-only 1 层；critic_layers=2）
  - d_model=192，GQA 8q/2kv，head_dim=24，FFN=576，context=4096

GRP V2 Teacher（kyoku-ending expected final-rank utility）：
  checkpoints/train_riichi_ppo_goal_grp_v2/grp_rank_predictor.pt

  - 结构 20 → 256 → 128 → 4（rank probs）
  - 最终排名 utility：(10, 4, -4, -10)

相关但本阶段不使用：
  checkpoints/train_riichi_v13_sft/best_heuristic.pt
  （Step 2 的候选策略；Step 3 候选改由 Deployment PPO 自己生成）
```

## 显卡与 CUDA 映射

物理编号（nvidia-smi）：

```text
物理 GPU 0：NVIDIA L20 44GB
物理 GPU 1：NVIDIA L20 44GB（可能有其他用户任务，使用前先 nvidia-smi 确认）
物理 GPU 2：NVIDIA T400 4GB（不要用于训练/采样）
物理 GPU 3：NVIDIA L20 44GB
物理 GPU 4：NVIDIA L20 44GB（可能有其他用户任务，使用前先 nvidia-smi 确认）
```

CUDA 编号映射：

```text
CUDA_DEVICE=0 ↔ 物理 GPU 0
CUDA_DEVICE=1 ↔ 物理 GPU 1
CUDA_DEVICE=2 ↔ 物理 GPU 3
CUDA_DEVICE=3 ↔ 物理 GPU 4
```

默认性能/训练测试：

```text
CUDA_DEVICE=0,3 且 learner_gpus=2（对应物理 GPU 0 与 4）
```

如果物理 GPU 4 被占用（如 Step 2.5 实测），改用：

```text
CUDA_DEVICE=0,2（对应物理 GPU 0 与 3）
```

本 Goal 采样并发受 CPU 限制，GPU 利用率预计很低；正式运行前必须执行 §四十八 的 worker 数 benchmark。

## Conda 环境

```text
conda 环境：Mahjong-AI
游戏环境：RiichiEnv（4p-red-half）
所有 Python / 训练命令使用：
  conda run -n Mahjong-AI python ...
```

## 资源硬上限

```text
CPU cores ≤ 24
Python worker processes ≤ 24
每个 worker 限制 1 CPU thread：
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  torch.set_num_threads(1)
```
