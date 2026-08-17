# 目录职责清单

每个目录的职责用一句话说明,与仓库实际结构一致;新增代码前先核对本清单与
宪法原则 I。治理范围覆盖 riichi_ppo_v1、riichi_lab_bot、RiichiEnv 三组件。

## riichi_ppo_v1(训练框架)

| 目录 | 职责 |
| --- | --- |
| `configs/` | 打包默认配置与自包含版本实验配置(训练、监控、SFT、v14/v15) |
| `docs/` | 协议、动作空间与 v13 数据契约文档 |
| `evaluation/` | 跨阶段确定性评测:1v3 对抗、分片驱动与策略适配边界 |
| `model/` | 模型结构、领域常量单一来源、动作分组与模型/环境转换边界 |
| `sft/` | SFT 数据准备、预计算、训练、契约与 checkpoint 持久化 |
| `tools/` | 独立生产 CLI(事件统计、协议验证入口) |
| `training/` | PPO 训练循环、rollout、单/双卡 learner(DDP)、奖励与对手机制 |
| `tests/` | 单元/集成/协议测试,不含任何生产入口 |

## riichi_lab_bot(在线 bot)

| 目录 | 职责 |
| --- | --- |
| `src/riichi_lab_bot/` | RiichiLab 客户端:checkpoint 加载、单席 bridge、安全校验与 CLI |
| `tests/` | bridge 语义、checkpoint 加载、client 与 safety 测试 |

## RiichiEnv(环境库)

| 目录 | 职责 |
| --- | --- |
| `src/riichienv/` | Python 环境封装:常量、TID 转换、手牌/动作与对局接口 |
| `riichienv-core/` | Rust 核心:规则、计分、向听、观察编码与重放 |
| `riichienv-state-machine/` | MJAI 协议状态机与持久化,公开模块名 `riichi`,不依赖 `riichienv` |
| `riichienv-python/` | PyO3 绑定构建入口 |
| `tests/` | 环境行为、动作合法性、和了与协议测试 |
| `docs/` | 环境文档 |
| `scripts/` | 构建、安装与日志校验脚本 |
