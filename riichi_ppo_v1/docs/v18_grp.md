# V18 GRP 输入协议(Mortal 方案扩展)

GRP(全局排名预测)是 PPO 的唯一奖励信号来源:在每个小局边界对完整前缀序列做
一次 GRU 前向,输出 4! = 24 类最终排名全排列 logits,经 `calc_matrix` 聚合为
`[4,4]` 玩家排名概率后取期望 utility `[1, 1/3, -1/3, -1]`;小局边界奖励为
前后两次期望 utility 的差(纯 GRP delta)。V18 是其唯一活跃契约;V16/V17
GRP 数据集、checkpoint 与配置仅作冷存储。

## 1. 输入契约(21 维)

每个 StartKyoku 边界一行 `[21] float32`,字段顺序为单一来源
`riichi_ppo_v1/model/grp.py::GRP_INPUT_LAYOUT`:

| 索引 | 字段 | 语义 | 域 |
| ---: | --- | --- | --- |
| 0 | `grand_kyoku` | E1..E4=0..3、S/W1..4=4..7 | 0..7 |
| 1 | `honba` | 本场数 | ≥0 |
| 2 | `kyotaku` | 立直棒 | ≥0 |
| 3–6 | `s0..s3` | 各绝对座位分数 / 1e4 | ≈-6..6 |
| 7 | `game_type` | 整局风数-1:0=东风(4 局)、1=半庄(8 局)、2=西风(12 局) | 0/1/2 |
| 8 | `prev_result_type` | 上一小局结果:0=首局、1=荣和、2=自摸、3=流局、4=中止 | 0..4 |
| 9–12 | `wins0..3` | 各玩家截至本小局开始的累计和了次数 | ≥0 |
| 13–16 | `dealins0..3` | 各玩家累计放铳次数 | ≥0 |
| 17–20 | `tenpai0..3` | 各玩家累计听牌流局次数 | ≥0 |

约束:新增 14 维全部只来自公开小局结果与局风(边界状态),**不包含**手牌、牌河、
宝牌、巡目、未来信息等局内发展信息;首局行为全 0(计数与 `prev_result_type`)。

- `game_type` 离线由整局 `start_kyoku` 的 bakaze 集合推导(取最大风索引);
  在线由 `game_mode` 字符串后缀映射(`single`/`east`→0、`half`→1、`west`→2),
  未知模式抛 `ValueError`(fail-closed)。
- 计数推进规则单点定义于 `training/grp/prepare.py::result_increment`:荣和
  winner 和了+1、target 放铳+1;自摸仅 winner 和了+1;流局把 `tenpai_mask`
  中每位玩家听牌流局+1。

## 2. 模型结构

`input_size(21) → 2 层 GRU(hidden=96,batch_first) → 末层 hidden 拼接(192) →
Linear(192,192) → ReLU → Linear(192,24)`;约 13.2 万参数。

构造器 `GRPModel(input_size, hidden_size, num_layers, num_classes)` 带默认常量;
checkpoint 的 `model_config`(input_size/hidden/layers/feature_layout)为唯一
形状来源,PPO 加载时据此构造后 `load_state_dict(strict=True)` 并 `freeze()`。

## 3. 数据与训练

- 数据集:`datasets/tenhou_grp_2024_2025_v18`(**全量数据** `denominator=1`,
  约 36.0 万 train / 约 0.37 万 validation 半庄;与 SFT 60% 子集完全重叠——
  GRP 为冻结奖励模型,仅用于评分,无策略污染);tar shard 以
  `--workers`(默认 6)个进程并行
  解析,记录按 shard 顺序拼接,输出与串行处理逐位一致。shard 边界可能切断
  半庄(同 game_id 成员分居相邻 tar 头尾),prepare 先做只读 tar 头的预扫描,
  按 game_id 把相邻 shard 合并为分组后组内跨 tar 聚合,保证每场半庄完整。
- 标签:每半庄最终 `rank_by_player`(同分按座位号稳定)映射 24 类;全部 prefix
  独立监督同一标签。
- 训练:`configs/v18_grp.yaml`;epochs=30、batch=2048、AdamW lr=1e-5、
  每 200 步验证、validation-loss best 落盘 `checkpoints/train_riichi_v18/grp/best.pt`;
  训练结束冻结后重写 best.pt,权重 `requires_grad=False`。
- 一键流程:`audit/reports/v18/scripts/run_v18_grp_prepare_and_train.sh`
  (prepare → train,日志 `logs/v18/`;`--skip-prepare` 跳过准备)。

## 4. 运行时一致性

PPO 的 `GrpRollout` 每环境维护累计计数,按边界链以与离线相同的
`result_increment` 推进,再经共同的 `feature_row` 生成边界行;因此任意边界
序列在离线(`features_from_boundaries`)与在线(`start_match`/`boundary_reward`)
两条路径上的特征矩阵逐位一致(契约测试 `test_offline_online_feature_parity`)。
