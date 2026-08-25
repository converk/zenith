# RiichiLab V18 独立对局客户端

客户端只接受 V18 `isolated_action_query` checkpoint，并复用训练侧模型、
`BatchedStateBridge.prepare()`、Atomic Snapshot 与语义校验。V16/V17 checkpoint
会被明确拒绝，不提供 fallback 或迁移。

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

第 1 场作为 warm-up，后两场单独统计。每份 Observation 会经过 base64 往返；四席
共享模型权重、各自维护 MJAI 状态机。模型动作必须同时属于环境合法动作与服务端
`possible_actions`，无法证明合法时不会以 fallback 替换。

## 在线 validation

```bash
read -rsp "RiichiLab bot token: " RIICHI_BOT_TOKEN
export RIICHI_BOT_TOKEN
CUDA_DEVICE=0,1 riichi-lab-bot validate --device cuda:0 --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v18/sft/best.pt \
  --jsonl-log logs/v18/bot-validate.jsonl
```

token 不得写入仓库或命令参数。客户端只响应 `request_action` 并回传原
`request_id`；事件流会重建线上 Observation 缺失的摸切、手切、立直与振听等公开
字段。每次决策调用 `assert_actor_input_semantics`，严格校验固定 Snapshot schema、
Query 成对 metadata、supplier 相对座次、action-ID 集合与 legal mask。隐藏牌、未来
牌山和 Critic 张量不会进入 Actor。

常用参数：`--checkpoint`/`RIICHI_CHECKPOINT`、`--device`、`--dtype`、
`--jsonl-log`、`--url`。checkpoint 必须包含精确 V18 `model_config` 与权重；运行中
不会热更新。开发验证：

```bash
conda run -n Mahjong-AI python -m pytest riichi_lab_bot/tests
```
