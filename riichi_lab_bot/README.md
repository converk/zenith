# RiichiLab V18 独立对局客户端

客户端只接受 V18 `current_state_snapshot` checkpoint,并复用训练侧模型、
`BatchedStateBridge.prepare()`、V18 当前局面编码路径与语义校验。V16/V17
checkpoint 会被明确拒绝,不提供 fallback 或迁移。

## 数据流

每份线上 Observation 经 base64 反序列化后,由 `ThreatSnapshotTracker` 重建线上
payload 缺失的公开字段(摸切、手切、立直、振听等),再经 `ObservationView.
native_observation` 物化交给 Rust 编码器。输入装配完全走 V18 当前局面快照路径:
Shared 公共前缀 + 三家 Opponent Analysis + 每个合法动作一对 Offense/Defense
Query;模型 forward 只消费 `actor_factors`/`actor_numeric`/`actor_lengths`/
`query_action_ids`/`query_pair_counts`/`legal_mask`。四席共享模型权重、各自维护
MJAI 状态机。模型动作必须同时属于环境合法动作与服务端 `possible_actions`,无法
证明合法时不会以 fallback 替换。

## 安装与本地验证

```bash
conda activate Mahjong-AI
bash RiichiEnv/riichienv-state-machine/scripts/install_conda_extension.sh
bash RiichiEnv/scripts/install_conda_extension.sh
python -m pip install -e ./riichi_lab_bot

CUDA_DEVICE=0,1 riichi-lab-bot local --games 3 --seed 20260825 \
  --device cuda:0 --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v18/sft/best.pt
```

第 1 场作为 warm-up,后两场单独统计。

## 在线 validation

```bash
read -rsp "RiichiLab bot token: " RIICHI_BOT_TOKEN
export RIICHI_BOT_TOKEN
CUDA_DEVICE=0,1 riichi-lab-bot validate --device cuda:0 --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v18/sft/best.pt \
  --jsonl-log logs/v18/bot-validate.jsonl
```

token 不得写入仓库或命令参数。客户端只响应 `request_action` 并回传原
`request_id`。每次决策调用两次 `assert_actor_input_semantics`(桥接装配后与
forward 前,按 checkpoint 的 `context_tokens` 复核):严格校验固定 Snapshot
schema、规范 token 顺序、supplier 相对座次、action-ID 集合与 legal mask;V18
rollout 约定下 `query_rows` 传 `None`,跳过逐 query 行一致性校验。隐藏牌、未来
牌山和 Critic 张量不会进入 Actor。

常用参数:`--checkpoint`/`RIICHI_CHECKPOINT`、`--device`、`--dtype`、
`--jsonl-log`、`--url`。checkpoint 必须包含精确 V18 `model_config` 与权重;运行中
不会热更新。开发验证:

```bash
conda run -n Mahjong-AI python -m pytest riichi_lab_bot/tests
```
