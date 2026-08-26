# V18 GRP 输入扩展设计

日期:2026-08-26。范围:GRP(全局排名预测)的输入契约、模型结构、数据量与
训练产物,保持 Mortal 式纯 GRP reward 契约(24 类排列、calc_matrix、冻结、
只读)不变。相关 spec:`specs/009-v18-grp/`。

## 1. 动机与事实核查

- GRP 输入由 7 维 `[grand_kyoku, honba, kyotaku, s0..s3/1e4]` 扩展为 21 维;
  新增字段全部是"一局结束后此刻的状态"(边界状态),不含任何手牌、牌河、
  未来信息等局内发展信息(与用户需求一致)。
- 已核查原始数据(40 个 shard 抽样):庄位 `oya` 恒等于 `(kyoku-1) mod 4`,
  即庄位是 `grand_kyoku` 的纯推导,不提供新信息,故**不**把庄位加入特征。
- **缺口 1(局风)**:原始数据局类型分布为东风 88,957 / 半庄 74,011 / 西风
  872 局。东风与半庄在相同 (风,局) 边界处的 7 维特征完全相同,但前者本局后
  不再有小局(E4 即是终局),后者还有最多 4 小局——剩余局数不同直接影响
  最终排名可预测性。V17 模型只能学到一个混合分布;`game_type` 特征修复此缺口。
- **缺口 2(结果分布)**:各玩家截至当前边界的累计和了/放铳/听牌流局次数无法由
  分数推导(和了自摸与放铳的得点结构不同),是预测最终排名的直接信号。
- **运行时**:真实对局环境只有 `game_mode="4p-red-half"`(半庄),在线
  `game_type` 恒为 1;离线数据含东风/西风,模型按 `game_type` 条件化,
  训练数据因此比"只留半庄"多约 1.2 倍且无分布混合问题。

## 2. 输入布局(21 维)

| 索引 | 字段 | 说明 |
| ---: | --- | --- |
| 0–6 | `grand_kyoku, honba, kyotaku, s0..s3/1e4` | 与 V17 完全一致 |
| 7 | `game_type` | 0=东风、1=半庄、2=西风 |
| 8 | `prev_result_type` | 0=首局、1=荣和、2=自摸、3=流局、4=中止 |
| 9–12 | `wins0..3` | 各玩家累计和了次数 |
| 13–16 | `dealins0..3` | 各玩家累计放铳次数 |
| 17–20 | `tenpai0..3` | 各玩家累计听牌流局次数 |

- 计数在边界处为"截至本小局开始"的累计值,首局行全 0;推进规则
  (`result_increment`)单点定义,离线按边界链推进,在线逐边界
  `boundary.previous` 推进——两条路径共用 `feature_row` 纯函数,保证逐位一致
  (单测 `test_offline_online_feature_parity` 以 `np.array_equal` 验证)。
- `game_type` 离线由整局 `start_kyoku` 的 bakaze 集合取最大风索引;在线由
  `game_mode` 后缀映射,未知模式 fail-closed。

## 3. 模型结构

7→64×2 层 GRU(fc 128→128→24,58,584 参数)提升为 21→96×2 层 GRU
(fc 192→192→24),参数合计 `21×96×3 + 96×96×3×2 + 96×96×3 + 192×192 +
192×24 + 偏置` = **131,832(≈2.25×)**。GRU 中间层增大对 CPU 边界推理仍
微不足道(每次边界前向 <1ms 量级);`GRPModel` 构造器可配置,checkpoint
`model_config` 为形状唯一来源,PPO 加载不再依赖预设常量。

## 4. 数据量

- V17:40% 子集(denominator=5,remainders=0,1)× `max_shards=280` →
  43,407 train / 1,461 validation 半庄。
- V18:`datasets/tenhou_grp_2024_2025_v18`,**全量数据(denominator=1,
  remainders=0)× 全部 930 个 train shard**(移除 max_shards 截断,子集从
  40% → 66.7% → 100%)→ 约 359,5xx train / 约 3,653 validation 半庄
  (≈8.3× V17)。
- 重叠说明:全量数据与 SFT 60% 子集完全重叠,60/40 无重叠惯例放弃——GRP
  是冻结的奖励模型,只输出排名评分,不参与策略训练,数据复用对策略无污染。
  (66.7% 版已实证:best val loss 2.5191 显著低于 V17 的 2.6038。)
- 训练超参:epochs=30、**batch=2048**、lr=1e-5(batch 增大不线性放大 LR,
  沿用 v17 配方)、val 每 200 步;步数随数据量/batch 变化,约 5.6 万步量级。
- 数据准备以 `--workers`(默认 6)个进程并行解析 tar shard(spawn 上下文,
  记录按 shard 顺序拼接),输出与串行处理逐位一致(单测
  `test_prepare_grp_dataset_parallel_matches_serial` 以全部 npz 数组
  `np.array_equal` 验证)。
- **shard 边界会切断半庄**(实测:相邻 shard 边界普遍存在同 game_id 成员分居
  头尾的情况,如 `2024010212gm-00a9-0000-90717c18` 的 10 个小局分布在
  shard 1 尾与 shard 2 头)。并行实现先做一次**只读 tar 头的预扫描**,按
  game_id 归属用并查集把相邻 shard 合并为分组,worker 在组内跨 tar 聚合——
  每场半庄完整且只产出一条记录(单测
  `test_prepare_grp_dataset_merges_games_spanning_shards` 覆盖首轮线上事故:
  该事故正是 fail-closed 校验在真实数据上首先发现的此问题,随后修复)。

## 5. 验证

- 契约测试:21 维布局、局风映射(内容/模式)、`result_increment` 三类结果、
  计数跨边界推进、离线/在线逐位一致、参数预算 110K–150K、dataset.json
  `riichi-grp-v18` 格式、冻结/快照。
- 全链路冒烟:临时目录 prepare→train→best.pt→`model_config` 构造→冻结前向
  通过(已执行)。
- 训练产物落点:`checkpoints/train_riichi_v18/grp/`、`logs/v18/`、
  `audit/reports/v18/report/PROGRESS.md`;脚本
  `audit/reports/v18/scripts/run_v18_grp_prepare_and_train.sh`。
