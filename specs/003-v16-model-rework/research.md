# Research: V16 模型重构与训练

## 现状盘点(代码库事实)

- 模型:`model/architecture.py` 的 `ModelConfig` 现为 mid(d192/8Q/2KV/24hd/576FFN/
  3 shared+1 actor/2 critic)与 large(384/12/3/32/1152);策略头为
  `isolated_action_query`(offense/defense 相邻 query 对),`offense_fusion` 分支用
  zero-init 投影(第 296–303 行),`critic_head_type=action_value` 时 Q head 是
  241 维 `Linear(d_model, NUM_ACTIONS)`。
- Actor 输入(v13)= Rust 状态机 history/状态 token + 6 行状态摘要
  (hand/value/placement/threat×3,`actor_features.py`)+ 候选动作
  offense/defense query(`decision.py` 候选分析)+ segment=3 的四家牌河/副露公开
  汇总(`critic_features.encode_public_summary`)。Critic 另含三家对手闭手牌与
  后续 5 张牌山(`critic_features.py`),即特权信息已存在。
- 环境:`riichienv-state-machine`(Python 模块名 `riichi`,不依赖 `riichienv`)已有
  独立 shanten 实现与 `analysis.rs`(genbutsu/suji/wall/honor-visible/passed、
  ukeire、wait_count);`riichienv-core` 有 `shanten.rs`(向听/有效牌/best-ukeire)、
  `HandEvaluator`(PyO3 暴露 `get_waits`、`calc` 返回 `WinResult{is_win,han}`)、
  `yaku.rs`/`yaku_checker.rs`。等待/役/番数/振听当前在 Python
  `training/rewards/decision.py::_rule_state` 用 core `HandEvaluator` 计算。
- SFT:`prepare.py` 把 tenhou-to-mjai 年包转 tar shard(train/validation 1% 划分,
  `--archive-dir` 必填);`precompute.py` 用 `--subset-denominator 5
  --subset-remainders 0,1` 得到 40% 子集,输出 manifest(format
  `riichi-sft-encoded-v3`、token_schema_version 13、feature_schema_sha256、
  rust_analysis_version 4、decision_analysis_version 16)与约 93.9M/96 万
  train/validation 决策;`sft/train.py` 已统计 top1/top3 准确率(即 Recall@3)。
- PPO:v15 已有 Q-boost 机制(`qboost_lambda=0.95`、`critic_head_type=action_value`、
  `zero_q_head_on_sft_init`、40 updates critic bootstrap);`rewards/` 当前组合为
  效率奖励 + 终局小局分差(`terminal_kyoku_reward`,/1000 截断)+ 半庄排名奖励
  `[16,8,-8,-16]`;1v3 机制常量在 `evaluation/mechanism.py`
  (10/160/1600/30);SFT 节奏键在 `configs/sft.yaml`(3000 steps/96 hanchan)。

## 决策

| # | Decision | Rationale | Alternatives considered |
|---|----------|-----------|------------------------|
| R1 | 输入编码为单一新协议版本 **v16**,废弃 token schema/feature schema/rust
  analysis/decision analysis 多版本拆分与 `_v<A>_v<B>` 组合命名 | 用户澄清:协议
  就是"一个新的信息编码协议版本";单一编号避免宪法与数据集出现歧义 | v13→v14 顺延
  双编号(被用户否决);继续 v13_v16 组合命名(遗留方案,否决) |
| R2 | Actor 输入 = Objective Facts(保留现有历史事件)+ Compact Snapshot(九项基础
  场况 + 3 个相对分差 + 3×7 对手摘要)+ 每合法动作一对 10-slot Offense/Defense
  Query;删除全部 Derived Features 与 Snapshot 中三家牌河/副露重复表示 | 设计文档
  §3–§4 钦定;历史事件已含完整公共信息,重复表示只会放大输入 | 保留部分派生特征
  (违背"删除全部");保留牌河/副露汇总(与设计意图冲突) |
| R3 | 网络 v16 preset:d_model=256、Q=16、KV=4、head_dim=16、FFN=1088、
  shared=4、actor-only=1、critic=2;总参数 7.5–7.8M、Actor 推理约 5.3M | 设计文档
  §2;GQA+gated FFN 结构不变 | hidden 288/320(设计文档明确排除) |
| R4 | Query 每 slot 独立 categorical/bucket embedding,聚合为单 token;Offense/
  Defense 经对称融合(concat→Linear 512→256→SiLU→Policy MLP)输出 logit,删除
  zero-init projection | 设计文档 §5/§9;攻守对等、普通初始化、SFT 从头 | 保留
  offense 零初始化(设计明确禁止);每 slot 单独 token(增加序列长度,否决) |
| R5 | Critic 保持公共编码 + 三家对手手牌 + 后续 5 张牌山 + Value Query,不追加
  Action Query Token;其公共部分改用 V16 Snapshot | 设计文档 §10;特权信息仍只在
  Critic | 把对手摘要/分差重复给 Critic(公共部分已含,无需重复) |
| R6 | Top-3 Q-boosting:π 取 Top-3,scorer 输入 [z_critic; detach(h_a)]
  (512→256→SiLU→1);训练候选 = Top-3 ∪ 行为动作(≤4),boost = Top-3;移除 241 维
  action_value Q head | 设计文档 §11;利用 SFT Recall@3≈98% | 保留全 241 Q
  (无意义动作 Q 学习);不 detach(设计禁止 Q loss 直改 Actor) |
| R7 | GRP:Linear→64、2 层 GRU(64)、64→32、SiLU、32→4、Rank Softmax,约
  50–70K 参数,仅在小局边界执行一次 | 设计文档 §14;轻量且不在 rollout 热路径 |
  Transformer 型 GRP(参数超标);每动作执行(性能瓶颈,否决) |
| R8 | 奖励 R=0.7·clip(R_GRP/σ_GRP,±5)+0.3·clip(clip(Δscore/1000,±12)/σ_Score,±5);
  utility [12,4,-6,-10];σ 离线一次固化;终局用真实排名 utility | 设计文档
  §12/§13/§17;方差量纲不同必须归一化 | 原始数值直接 70/30(设计明确否定);
  训练期动态更新 σ(否定) |
| R9 | 分析函数归属:向听/有效牌/等待/有无役/基础番数/振听/可立直 → core(复用
  shanten/HandEvaluator/yaku);现物/筋/公开数/安全牌库存/门清/对手 7 项摘要 →
  state-machine(`riichi`);相对分差与 O9 宝牌赤牌聚合 → 模型输入转换侧 | 约束③的
  归属规则;core 已有手牌规则评价,state-machine 已有河牌掩码分析 | 全部放
  state-machine(反向依赖/重复实现手牌评价,否决) |
| R10 | SFT 重编码沿用 prepare→precompute 流水线;manifest 契约改为
  `format=riichi-sft-encoded-v16`(由单一常量派生)+ 单一
  `encoding_protocol_version=16` + 协议契约 sha256;数据集
  `datasets/tenhou_sft_2024_2025_encoded_40pct_v16`,GRP 数据集
  `datasets/tenhou_grp_2024_2025_v16`;40% 划分与 train/validation 比例沿用 | 约束
  ②与 R1;单一协议版本、单一来源 | 新写独立编码器绕过 precompute(重复建设);
  改变 40% 划分(破坏与历史对照) |
| R11 | 评测机制不动:1v3(10×160=1600,每 30 updates)与 SFT(每 3000 steps、最终
  96 hanchan)沿用单点常量;性能基线 target_kl=0.0、update_epochs=4、
  kyokus_per_worker=16、CUDA_DEVICE=0,1、learner_gpus=2、3 轮首轮预热 | 约束⑤与
  宪法原则 IV/V;机制改动必须走宪法修订 | 在 v16 实验配置里复制节奏键(禁止) |
| R12 | 删除初判清单(每项先 rg 零引用+测试通过,按主题分 commit):v13 语义契约
  `feature_schema.py`、v13 六行状态摘要与候选 query 编码、segment=3 牌河/副露
  汇总、`offense_fusion` zero-init、241 维 action_value Q head、效率/半庄排名
  奖励组合;checkpoint 与数据集一律不删 | 约束①;V16 输入与奖励已全部替换 | 保留
  双轨(宪法原则 II 禁止) |

## 未决项

无 NEEDS CLARIFICATION:版本编号经 `$speckit-clarify` 定为 v16;其余未写死细节已
在 spec.md Assumptions 拍板。以下仅为实现期需固定并登记契约常量的内部值:
协议契约 sha256、GRP 归一化统计量、GRP checkpoint 目录布局
(`checkpoints/train_riichi_v16/grp`)。

