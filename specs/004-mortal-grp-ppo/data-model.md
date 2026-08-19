# Data Model: Mortal 式 GRP 与 PPO 训练数据

**版本**: v17 GRP 契约(信息编码协议保持 v16,token schema 13)
**日期**: 2026-08-19

## 1. GRP 特征(Single-Source Constants)

```python
GRP_SIZE = 7                      # 每 StartKyoku 一行
# 列语义(Mortal 一致):
# [0] grand_kyoku: E1=0..E4=3, S1=4..S4=7, W1=8..W4=11(4p 半庄只到 7)
# [1] honba
# [2] kyotaku(立直棒)
# [3..6] score[0..3] / 10000(实数,如 2.5)
```

- 不做 4 视角旋转:特征是全局绝对座位帧(与 Mortal 相同),不依赖 viewer。
- 每个半庄产生完整 prefix 序列 `feature[:i+1] for i in range(T)`,`T` =
  StartKyoku 数量。

## 2. 24 类排列标签

- `perms`:按 `itertools.permutations(range(4))` 顺序生成(24,4),作为
  `register_buffer` 固化。
- `perms_t`:转置 (4,24),用于 calc_matrix 的 mask。
- 标签:给定半庄最终 `rank_by_player`(= player → 0..3 最终顺位,按分数降序、
  同分按座位号稳定的 `rank_among` 排序),`label = perms 中与
  rank_by_player 全等的那一行的索引`。
- 每个 prefix 的标签都是该半庄最终排序(与 Mortal `get_label` 一致)。

## 3. 模型契约(model/grp.py)

```python
class GRPModel(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        self.rnn = nn.GRU(GRP_SIZE, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * num_layers, hidden_size * num_layers),
            nn.ReLU(),
            nn.Linear(hidden_size * num_layers, 24),
        )
        # register_buffer perms / perms_t
    def forward_packed(packed) -> logits (N, 24)
    def forward(inputs: list[Tensor]) -> logits(...)   # 兼容 list 输入
    def calc_matrix(logits) -> (N, 4, 4)                # [player][rank] 概率
    def get_label(rank_by_player) -> (N,) long
    def freeze() -> None                                # 全参数 requires_grad=False
```

## 4. Reward 契约(training/grp/reward.py)

- `RANK_UTILITY = (1.0, 1/3, -1/3, -1.0)`(rank 0..3)。
- `expected_rank_utility(matrix, player)` : `Σ_rank P(rank)·U(rank)`。
- 每小局边界 reward:
  - 非终局:`r = expected_utility(boundary_{k+1}, player) −
    expected_utility(boundary_k, player)`
  - 终局:`r = RANK_UTILITY[真实排名(player)] − expected_utility(最后一局开始,
    player)`
- 无 σ 归一化、无点差分量、无 clip(幅度天然 ≤ 2.0;如后续需要可加 ±2 clip,
  但默认关闭)。
- PPO 端:transition.reward 直接写这个小局 δ;GAE/return 用 discount γ=1 与
  现有 trajectory 机制(终局 done 重置)。

## 5. GRP 数据集格式(沿用 chunk npz)

`datasets/tenhou_grp_2024_2025_v17/{train,validation}/<split>-NNNNN.npz`:

```
offsets:        int64 (N+1,)                 # 样本边长累积偏移
features:       float32 (ΣT, 7)              # 7 维特征拼接
labels:         uint8  (N,)                  # 24 类排列标签
years:          int16  (N,)
game_ids:       str    (N,)
```

- N = 样本数(半庄数 × 4 视角?否——Mortal 无视角,每半庄 1 个样本序列,但
  样本以「半庄」为单位,每个半庄 1 行记录;训练时把半庄内 T 个 prefix 展开为
  T 个训练样本。为保持现有 chunk 迭代器形态,选择「每半庄 1 样本,序列内
  prefix 监督」或「每 prefix 1 样本」二选一:本项目沿用现有 `iter_grp_samples`
  的「样本 = 半庄,序列 = prefix」形态,输出 `(features 序列, label 序列)`。
  具体以 implementation 时与现有点通配为准。
- `dataset.json` 记录 40% 划分统计与 train/validation 半庄数。

## 6. PPO 训练配置数据流

- Rollout:worker 每个环境维护 1 条 7 维前缀序列(每 StartKyoku 追加一行);
  每小局边界用 GRU 末步输出算 expected utility;非终局记 δ,终局记真实排名 δ。
- Update:512 半庄 → transitions 集合(半庄内所有 current 座位的决策);
  2 GPU DDP 轮询分片 → 每 rank 1536 样本 minibatch(3072 global);
  `update_epochs=2`。
- Checkpoint:每 5 updates 保存 `checkpoint_00NNN.pt` 并 1v3 vs V16 SFT 4000
  半庄,汇总 JSON `audit/reports/v17/eval/vs_sft_uNNN.json`。