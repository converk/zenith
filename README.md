# Zenith 立直麻将训练系统

Zenith 的现行输入与模型契约是 V19。仓库由 `RiichiEnv/` 原生环境、
`riichi_ppo_v1/` SFT/PPO 框架和 `riichi_lab_bot/` 在线客户端组成。历史代次的
checkpoint、数据集与产物只允许在磁盘上冷存储；仓库内不再保留旧代次的配置、文档
或脚本，活跃路径也不会加载或迁移旧契约。

## V19 快速入口

```bash
conda activate Mahjong-AI
bash RiichiEnv/riichienv-state-machine/scripts/install_conda_extension.sh
bash RiichiEnv/scripts/install_conda_extension.sh
python -m pip install -e riichi_ppo_v1 --no-deps --no-build-isolation
python -m pytest riichi_ppo_v1/tests RiichiEnv/tests riichi_lab_bot/tests
```

- 输入协议：[V19 输入协议](riichi_ppo_v1/docs/v19_input_protocol.md)
- 状态与环境边界：[Kyoku 状态协议](riichi_ppo_v1/docs/KyokuEventTupleProtocol.md)
- SFT 使用方式：[V19 SFT](riichi_ppo_v1/docs/v19_sft.md)
- 设计与验收：[V19 设计文档](audit/reports/v19/design/)
- 可复现进度：[V19 PROGRESS](audit/reports/v19/report/PROGRESS.md)

V19 使用**决策时刻状态快照 + 信念注入**（Shared 公共前缀 + 三家 Opponent Analysis +
三家 RIICHI_CARD + 模型内部 30 个信念 token + 每个合法动作一对 Offense/Defense
Query，全 token RoPE、公共双向 GQA、结构化 Actor mask）。模型为 `d_model=256`、
16Q/4KV GQA、约 7.11M 参数的 Actor-Critic，Actor 与 Critic 信息边界严格隔离；
PPO/rollout 与 `riichi_lab_bot` 均运行在 V19 当前局面输入上，PPO 只采用固定的
1v3 评测机制。
