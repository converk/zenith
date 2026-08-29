# Zenith 立直麻将训练系统

Zenith 的现行输入与模型契约是 V18。仓库由 `RiichiEnv/` 原生环境、
`riichi_ppo_v1/` SFT/PPO 框架和 `riichi_lab_bot/` 在线客户端组成。V16/V17 的
checkpoint、数据集、配置、日志和报告仅作冷存储；活跃运行路径不会加载或迁移旧契约。

## V18 快速入口

```bash
conda activate Mahjong-AI
bash RiichiEnv/riichienv-state-machine/scripts/install_conda_extension.sh
bash RiichiEnv/scripts/install_conda_extension.sh
python -m pip install -e riichi_ppo_v1 --no-deps --no-build-isolation
python -m pytest riichi_ppo_v1/tests RiichiEnv/tests riichi_lab_bot/tests
```

- 输入协议：[V18 输入协议](riichi_ppo_v1/docs/v18_input_protocol.md)
- 状态与环境边界：[Kyoku 状态协议](riichi_ppo_v1/docs/KyokuEventTupleProtocol.md)
- SFT 使用方式：[V18 SFT](riichi_ppo_v1/docs/v18_sft.md)
- 设计与验收：[V18 spec](specs/008-v18-input-architecture/spec.md)
- 可复现进度：[V18 PROGRESS](audit/reports/v18/report/PROGRESS.md)

V18 使用**决策时刻状态快照**（Shared 公共前缀 + 三家 Opponent Analysis + 每个合法动作
一对 Offense/Defense Query，全 token RoPE、公共双向 GQA、结构化 Actor mask）。模型为
`d_model=256`、16Q/4KV GQA、约 5.80M 参数的 Actor-Critic，Actor 与 Critic 信息边界
严格隔离；PPO/rollout 与 `riichi_lab_bot` 均已运行在 V18 当前局面输入上，
PPO 仍只采用宪法规定的固定 1v3 评测机制。
