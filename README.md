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

V18 使用 Objective Facts、固定 29 字段 Atomic Snapshot 和每个合法动作一对
Offense/Defense Query。模型为约 4.93M 参数的 GQA Actor-Critic，Actor 与 Critic
信息边界严格隔离。PPO 仍只采用宪法规定的固定 1v3 评测机制。
