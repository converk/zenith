# V19 信念注入残差化与 token 精简（零初始化 token_matrix + 8 token/家）实施方案

> 状态：已实施（2026-09-08；代码、测试与文档已按本文落地）。
> 背景：V19 重训前两项结构决策（用户确认）：
> 1. **token_matrix 零初始化（残差式 no-op 起步）**——补齐信念注入路径的残差语义；
> 2. **每玩家信念 token 10 → 8（总量 30 → 24）**——收缩注入接口容量。
> 梯度隔离方案不变：信念网络仅由四头监督标签更新，`summary.detach()` 保持，
> `belief_public_grad_scale=0` 保持（见 `V19_信念网络策略梯度隔离_实施方案.md`）。

## 1. 动机

### 1.1 token_matrix 零初始化（残差化）

残差旁路的完整语义 = ①加性旁路 ②主干不被旁路梯度污染 ③旁路有自己的监督锚
④**零初始化 no-op 起步**。V19 此前已有①②③：

- 30 token 是额外槽位（加性注入）、readout 是 `pair_hiddens +=`（加法）；
- `belief_public_grad_scale=0` 保证信念监督不回传公共权重；
- 四头标签监督锚定表征；
- readout 投影已零初始化（严格 no-op）。

缺④：`token_matrix` 用默认初始化，SFT 第一步起 30 个随机 token 即注入 actor
流，策略须先学会「忽略噪声」。零初始化后训练起点 24 个信念 token 全零
（近似 no-op：零向量经 RoPE 旋转仍为零、value 贡献为零；30/24 个槽位仍在
attention softmax 分母中，存在轻微稀释，实践无害），策略先学基础策略，
接口由策略/BC 梯度按需从零长出；四头表征学习与接口学习异步解耦、互不等待。

### 1.2 token 10/家 → 8/家

用户决策：收缩注入接口容量（30 → 24 token），减少 actor 序列中信念块占比。
协议布局不变式保持：信念块插在 SEP_ACTIONS 之后、第一对 Query 之前，最后一个
信念 token 距第一对 Query 恒距 1。

### 1.3 明确不做的（决策记录）

**策略损失不穿过 token_matrix 进入信念网络**（`summary.detach()` 保持）。
历史证据：2026-09-06 隔离前的穿透状态实测导致 wait 头校准漂移
（validation `belief_wait_loss` 6.86→7.59，train 同期降至 4.60）。detach 只挡
梯度不挡信息，策略已能经 24 token + 18 维读出获取全部信念信息；若策略需要
信念网络未提供的特征，正确修法是扩展标签/加头，而非让 PPO 梯度塑形四头。

## 2. 代码改动

| 文件 | 改动 |
| --- | --- |
| `model/belief_network.py` | `DEFAULT_TOKEN_COUNT` 10→8；`token_matrix` 权重与 bias `zeros_` 初始化（注释写明残差式 no-op 起步、梯度不受影响）；docstring 30→24、×10→×8 |
| `model/architecture.py` | 注入块 7 处硬编码 `30` 改为 `belief_token_total = BELIEF_PLAYERS * self.belief_network.token_count`（派生常量，不再硬编码）；隔离注释补零初始化说明、五头→四头 |
| `model/parameter_count.py` | 契约注释同步（摘要 121 维、8 token/家、实测 6,639,173） |

数据集/编码协议**不变**：信念 token 是模型内部产物（不进 Rust 编码器、不进
manifest/契约 hash），已编码数据集无需重跑 precompute。

## 3. 测试

- `tests/unit/test_v19_belief_network.py`：token 形状 (2,30,256)→(2,24,256)；
  `out_features == 8 * d_model`；**新增** `test_token_matrix_zero_init_noop_start`
  （初始 token 全零 + backward 后 `token_matrix.grad` 非零非 None）。
- `tests/unit/test_v19_architecture.py`：token 形状断言 30→24（含 `_tiny_config` 路径）。
- 隔离测试不改动即通过：零初始化下 `token_matrix.weight.grad is not None`
  仍成立（dL/dW = 上游梯度 ⊗ summary，summary 非零）。
- 实测：unit 套件 246 passed。

## 4. 文档同步

`docs/v19_input_protocol.md`（30→24、×10→×8、+30→+24、四头、零初始化、
121→2048 转换矩阵）、`docs/KyokuEventTupleProtocol.md`（30→24 ×2）、
`docs/v19_sft.md`（24 token、四头、`belief_public_grad_scale=0`）、
`AGENTS.md` 版本契约（24 token、8/家、token_matrix 零初始化）。

## 5. 兼容性

token 数变化改变 `token_matrix` 形状与 actor 增广序列布局：旧 V19 checkpoint
（30 token）与新模型 state-dict 不兼容，属预期（重训起点，无旧 checkpoint
迁移；历史 checkpoint 仅冷存储）。

## 6. P1 诊断与监控（同日追加，随本方案落地）

两个 P1 项，纯测量、不改变训练梯度：

### 6.1 信念接口使用度监控（P1-2）

- `belief_network.token_matrix` 权重/偏置的 L2 范数，随 update（PPO）/cadence（SFT）记录：
  - PPO：update 级 `belief/token_matrix_weight_norm`、`belief/token_matrix_bias_norm`
    （TB 白名单 `ppo/belief/...`）；
  - SFT：cadence 级 `train/belief_token_matrix_weight_norm`、`train/belief_token_matrix_bias_norm`。
- 零初始化起步，增长曲线直接回答「策略用了多少信念」，是判断 8 token/家容量是否够的第一手证据。

### 6.2 反事实「开闸②」梯度余弦（P1-1）

- **方法论要点**：完整梯度隔离下，策略/BC 损失与信念监督在信念私有参数上
  **没有任何共同图路径**——LIVE 图中互梯度恒为 None，任何「直接测余弦」都
  是空集。因此诊断用一次 **eager 反事实前向**（`belief_summary_detach=False`
  + `belief_readout_detach=False`，新增模型管路，默认 True 不改变任何训练行为）
  构造「若开闸②」的假想图，测策略梯度与信念监督梯度在信念私有参数
  （belief_query/backbone/四头，不含 token_matrix——它是策略梯度独占写者）
  上的余弦。
- 判读：持续负相关 → 开闸②必然互相干扰；接近正相关 → 开闸②「安全」
  （但不保证有益）。为零/None 时只记录范数。
- PPO：`_belief_gate_cosine_diagnostics`，复用逐损失项归因节奏
  （`grad_term_diagnostics_interval_updates`，首个 minibatch）；策略项为同一
  clipped surrogate 在反事实 logits 上的重现；信念监督项复用 LIVE 图张量
  （四头输出与 detach 开关无关，逐位一致）。
- SFT：`_belief_gate_cosine_scalars`，独立节奏 `grad_cosine_interval_steps`
  （默认 500，0 关闭；optimizer.step 后快照）；策略项为同一 BC CE。
- 实现纪律：unwrap 后 eager 前向（不进 compile 缓存、不触发 DDP 同步）、
  `torch.autograd.grad` 不触碰 `param.grad`、独立 try/except 一次性降级
  （`grad_cos/diagnostics_failed` / stderr 告警），绝不中断训练。
- 指标键：PPO `ppo/grad_cos/policy~belief/belief_private`（+两条范数、失败标记）；
  SFT `train/grad_cos_bc_belief_private`（+两条范数）。

### 6.3 测试

`tests/unit/test_v19_belief_diagnostics.py`：余弦助手（同向/反向/正交/None/零）、
私有参数判定、反事实覆写管路（False 时策略梯度可达四头与 belief_query，
True 时不可达——与隔离测试互为印证）。全套 282 passed。
