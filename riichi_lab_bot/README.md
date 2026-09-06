# RiichiLab V19 独立对局客户端

客户端只接受 V19 `current_state_snapshot` checkpoint,并复用训练侧模型、
`riichi_ppo_v1.model.current_state.encode_batch` 编码路径与语义校验。旧代次
checkpoint 会被明确拒绝,不提供 fallback 或迁移。

## 数据流

每份线上 Observation 经 base64 反序列化后,由 `ThreatSnapshotTracker` 重建线上
payload 缺失的公开字段(摸切、手切、立直、振听等),再经 `ObservationView.
native_observation` 物化交给 Rust 编码器。输入装配直接走 V19 当前局面快照路径:
Shared 公共前缀(含三家 `RIICHI_CARD`) + 三家 Opponent Analysis + 每个合法动作
一对 Offense/Defense Query;模型 forward 只消费 `actor_factors`/`actor_numeric`/
`actor_lengths`/`query_action_ids`/`query_pair_counts`/`legal_mask`。四席共享模型
权重、各自维护 MJAI 状态机。模型动作必须同时属于环境合法动作与服务端
`possible_actions`,无法证明合法时不会以 fallback 替换。

## 完整 MJAI 事件日志

从 `start_game` 起把收到的全部 MJAI 事件原文逐行写入专用 JSONL:
`logs/v19/bot_mjai/<session>-<game_id>-<yyyymmdd_hhmmss>.jsonl`
(每行 `log_no`/`game_no`/`seat`/`timestamp`/`event`)。参数:
`--mjai-log-dir`(缺省 `logs/v19/bot_mjai`)与 `--session`(缺省 `bot`)。
写失败只记录错误并继续,不中断对局;可用
`riichi_lab_bot.telemetry.replay_mjai_log` 回放重建终局分数/顺位/和牌数。

## 安装与本地验证

```bash
conda activate Mahjong-AI
bash RiichiEnv/riichienv-state-machine/scripts/install_conda_extension.sh
bash RiichiEnv/scripts/install_conda_extension.sh
python -m pip install -e ./riichi_lab_bot

CUDA_DEVICE=0,1 riichi-lab-bot local --games 3 --seed 20260825 \
  --device cuda:0 --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v19/sft/best.pt
```

第 1 场作为 warm-up,后两场单独统计。

## 在线 validation

```bash
read -rsp "RiichiLab bot token: " RIICHI_BOT_TOKEN
export RIICHI_BOT_TOKEN
CUDA_DEVICE=0,1 riichi-lab-bot validate --device cuda:0 --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v19/sft/best.pt \
  --jsonl-log logs/v19/bot-validate.jsonl
```

token 不得写入仓库或命令参数。客户端只响应 `request_action` 并回传原
`request_id`。每次决策调用两次 `assert_actor_input_semantics`(桥接装配后与
forward 前,按 checkpoint 的 `context_tokens` 复核):严格校验固定 Snapshot
schema、规范 token 顺序、supplier 相对座次、action-ID 集合与 legal mask;V19
rollout 约定下 `query_rows` 传 `None`,跳过逐 query 行一致性校验。隐藏牌、未来
牌山和 Critic 张量不会进入 Actor。

常用参数:`--checkpoint`/`RIICHI_CHECKPOINT`、`--device`、`--dtype`、
`--jsonl-log`、`--mjai-log-dir`、`--session`、`--url`。checkpoint 必须包含精确
V19 `model_config` 与权重;运行中不会热更新。开发验证:

```bash
conda run -n Mahjong-AI python -m pytest riichi_lab_bot/tests
```
