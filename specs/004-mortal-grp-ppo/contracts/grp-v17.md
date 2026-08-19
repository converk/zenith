# Contracts: GRP v17 协议

**协议版本**: grp-v17 | **关联**: 信息编码协议 v16(token schema 13)

## 1. 输入契约

- 每样本 = 一个半庄的 StartKyoku 前缀序列,每行 7 维 float32:
  `[grand_kyoku, honba, kyotaku, s0/1e4, s1/1e4, s2/1e4, s3/1e4]`
- `grand_kyoku`:E1=0..E4=3,S1=4..S4=7(4p 不支持 3 圈以上;超出 7 抛错)。
- 首局 `scores` 为 25000 → 2.5;honba/kyotaku 为整数。

## 2. 输出契约

- `logits`: `(N, 24)` 24 类排列 logits(排列按 `itertools.permutations(range(4))`
  顺序索引 0..23)。
- `calc_matrix(logits) -> (N, 4, 4)`:
  `matrix[i, player, rank] = Σ_{perm: perm[player]==rank} softmax(logits[i])[perm]`
  - 对每个 i、player,`Σ_rank matrix[i, player, rank] == 1`。
- `get_label(rank_by_player) -> (N,)`:`rank_by_player[player]` 为最终顺位
  (0..3),映射到排列索引。

## 3. 冻结契约

- `freeze()` 后所有参数 `requires_grad=False` 且 `eval()`;PPO 不更新 GRP。
- checkpoint 载荷:
  ```json
  {
    "model": <state_dict>,
    "config": <训练配置>,
    "model_config": {"grp_input_size": 7, "hidden": 64, "layers": 2, "num_classes": 24},
    "training_stage": "grp",
    "global_step": <steps>,
    "validation_loss": <best val loss>,
    "timestamp": <epoch seconds>
  }
  ```

## 4. Reward 契约(training 侧)

- utility `[1, 1/3, -1/3, -1]`;`expected = Σ P_rank · U_rank`。
- δ 规则与非终局/终局分支见 data-model.md §4。
- PPO 配置:`reward = δ`(无 σ、无点差);GRP 调用次数统计
  `grp_calls` 继续上报。

## 5. 版本化与存储

- 数据集:`datasets/tenhou_grp_2024_2025_v17/`
- GRP checkpoint:`checkpoints/train_riichi_v17/grp/best.pt`
- 与 V16 GRP 无兼容:旧 `GRP_CATEGORIES/GRP_NUMERIC_FEATURES/GRP_UTILITY`
  (24/8/-12/-24)被新契约取代;`model/grp.py` 全量重写。