# RiichiLab 独立对局客户端

这个目录包含一个可独立运行的四人麻将客户端，用 `--checkpoint` 指定的 V16/V17
actor 模型(例如
`checkpoints/train_riichi_v17/archive_20260819_V1run1/ppo/latest.pt`)连接
[RiichiLab](https://riichi.dev/)。

它直接复用 `riichi_ppo_v1` 的 `KyokuTransformerActorCritic`、训练侧
`BatchedStateBridge.prepare_v16()`、Compact Snapshot 与 Offense/Defense
Query 编码；本目录只保留单席在线状态 bridge、WebSocket 客户端和安全
校验。V16 SFT 与 V17 PPO 的 actor 输入和拓扑一致,共用同一套 bot 逻辑。
运行时需要本仓库提供的 `riichi`、`riichienv` 原生扩展以及
`riichi_ppo_v1` 源码。

## 安装

要求 Python 3.12，并使用项目现有的 `Mahjong-AI` Conda 环境：

```bash
conda activate Mahjong-AI
bash RiichiEnv/riichienv-state-machine/scripts/install_conda_extension.sh
bash RiichiEnv/scripts/install_conda_extension.sh
python -m pip install -e ./riichi_lab_bot
```

安装会使用 `torch==2.7.1` 和 `websockets==16.1.1`。可以用下面的命令
检查入口与原生扩展：

```bash
python -c "import riichi, riichienv, riichi_lab_bot; print('runtime ok')"
riichi-lab-bot --help
```

## 本地测试

默认运行三场固定种子半庄。第 1 场作为 warm-up，第 2、3 场分别输出
推理统计，最终汇总只计算后两场：

```bash
CUDA_DEVICE=0,1 riichi-lab-bot local \
  --games 3 \
  --seed 20260730 \
  --device cuda:0 \
  --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v17/archive_20260819_V1run1/ppo/latest.pt
```

本地测试会对每份 Observation 先执行 base64 序列化和反序列化，以模拟
RiichiLab 的 `request_action.observation`。四席共享一份模型权重，但各自
持有独立的 MJAI 状态机。每个输出动作都必须同时通过：

1. 当前 `Observation.legal_actions()` / `select_action_from_mjai()`；
2. 当前请求的 `possible_actions`。

CPU 正确性测试也可直接运行：

```bash
riichi-lab-bot local --games 1 --device cpu --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v16/sft/best.pt
```

## 创建并验证机器人

1. 登录 RiichiLab，在 Bots 页面创建一个四人机器人并复制只显示一次的
   token。
2. 不要把 token 写入仓库或命令参数。将其放入环境变量：

```bash
read -rsp "RiichiLab bot token: " RIICHI_BOT_TOKEN
echo
export RIICHI_BOT_TOKEN
```

3. Pending 状态的机器人连接 validation：

```bash
CUDA_DEVICE=0,1 riichi-lab-bot validate \
  --device cuda:0 \
  --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v17/archive_20260819_V1run1/ppo/latest.pt \
  --jsonl-log logs/v17/bot-validate.jsonl
```

默认连接 `wss://game.riichi.dev/ws/validate`。客户端会等待
`validation_result`；通过时退出码为 0，失败或未收到通过结果时退出码为
2。validation 只运行一次，不会自动重试。

## 排位说明

`ranked` 子命令与 `RANKED_URL` 按需求保留，供后续线上排位使用。本次清理
任务的验收不执行 ranked、不传 `--forever`、也不连接真实排位端点；
`validate` 路径与 ranked 路径相互独立。

## 配置

所有子命令都支持：

| 参数或环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `--checkpoint` | 必填(无内置默认) | checkpoint 文件,不锁定历史版本 |
| `RIICHI_CHECKPOINT` | 未设置 | `--checkpoint` 未显式给出时提供模型路径 |
| `--device` | `auto` | `auto`、`cpu`、`cuda` 或 `cuda:index` |
| `--dtype` | `auto` | `auto`、`fp32` 或 `bf16` |
| `CUDA_DEVICE` | 未设置 | 在导入 PyTorch 前映射为 `CUDA_VISIBLE_DEVICES` |
| `--jsonl-log` | 未设置 | 追加写入结构化运行日志,按规范写 `logs/<版本号>/`(如 `logs/v17/bot-validate.jsonl`) |
| `--log-level` | `INFO` | `DEBUG`、`INFO`、`WARNING` 或 `ERROR` |
| `--url` | RiichiLab 官方 endpoint | 连接本地/自托管服务器时覆盖 |

配置优先级为 `--checkpoint`、`RIICHI_CHECKPOINT`;两者都缺时程序报错退出。
模型在进程启动时加载一次;运行期间 checkpoint 文件发生变化不会热更新
当前进程。

`auto` 在支持 BF16 的 CUDA 上选择 BF16；本任务的验证与本地对局统一显式
使用 `--dtype fp32`。bot 只支持 V16/V17 actor 结构,checkpoint 必须提供
`model_config` 和 `model`,且 `policy_head_type` 必须是
`symmetric_action_query`;权重按 `model_config` strict load。旧 token schema、
feature hash、PPO v2 与旧 SFT contract 不再作为 bot 运行时闸门。

## 协议与安全行为

- 只在 `request_action` 上响应，并始终原样回传当前 `request_id`。
- 其他 MJAI 事件只作为信息处理；未知事件、未知字段和二进制帧会被忽略。
- 线上 Observation 缺少 `riichi_accepted`、`riichi_declaration_indices`、
  `missed_agari_*`、`tiles_left`、`tsumogiri_flags`、`last_tedashis`、
  `riichi_sutehais` 或 `drawn_tile` 等字段时,由客户端事件流重建后再编码
  V16 输入。
- 模型动作通过 RiichiEnv 回转后，再与服务器 `possible_actions` 比较
  `type`、`pai`、`consumed` 和服务端提供时的 `tsumogiri`。
- 每个决策前调用 `assert_v16_actor_input_semantics`,校验 history actor
  可见性、snapshot kind/宽度/有限值、query offense/defense 成对、
  `query_action_ids` 与 `legal_mask` 一一对应,且总长度不超过 context。
- 模型动作无法证明合法时不会用 fallback 顶替；任何 fallback/withheld 都
  是任务失败信号，必须修复根因。
- 如果本地处理时间已进入 `deadline_ms` 前 250 ms 的安全区，也不会发送
  可能变成迟到响应的动作。
- 日志记录 request id、动作类型、推理耗时和 `action_ack`，但不会记录
  token、Authorization header 或 base64 Observation。

相关服务端约定：

- [MJAI Protocol](https://riichi.dev/docs/protocol)
- [Bot Validation](https://riichi.dev/docs/validation)
- [Local Testing](https://riichi.dev/docs/local-testing)
## 开发测试

```bash
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest \
  -q riichi_ppo_v1/tests/unit/test_semantic_validation.py \
  riichi_ppo_v1/tests/integration/test_v16_encoding_bridge.py \
  riichi_ppo_v1/tests/integration/test_v16_query_semantics.py \
  riichi_lab_bot/tests
```

测试包括 V16/V17 checkpoint strict load、V16 输入语义校验、红五/摸切、
见逃与立直振听重建、mock WebSocket、validation 流程、deadline 不发送、
完整本地半庄,以及单席 online bridge 与训练侧 `prepare_v16()` 的逐段等价
检查。
