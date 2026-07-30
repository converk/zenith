# RiichiLab 独立对局客户端

这个目录包含一个可独立运行的四人麻将客户端，用
`checkpoints/train_riichi_v10_sft` 中的语义 token 模型连接
[RiichiLab](https://riichi.dev/)。

它不会导入 `riichi_ppo_v1`。checkpoint 兼容的模型结构、单席状态
bridge、公开牌河/副露特征、241 维动作空间、WebSocket 客户端和安全校验
都包含在本目录中。运行时只需要本仓库提供的 `riichi`、`riichienv`
原生扩展。

## 安装

要求 Python 3.12，并使用项目现有的 `Mahjong-AI` Conda 环境：

```bash
conda activate Mahjong-AI
bash riichi/scripts/install_conda_extension.sh
bash RiichiEnv/scripts/install_conda_extension.sh
python -m pip install -e ./riichi_lab_bot
```

安装会使用 `torch==2.7.1` 和 `websockets==16.1.1`。可以用下面的命令
检查入口、原生扩展和默认 checkpoint：

```bash
python -c "import riichi, riichienv, riichi_lab_bot; print('runtime ok')"
test -f checkpoints/train_riichi_v10_sft/best_heuristic.pt
riichi-lab-bot --help
```

## 本地测试

默认运行三场固定种子半庄。第 1 场作为 warm-up，第 2、3 场分别输出
推理统计，最终汇总只计算后两场：

```bash
CUDA_DEVICE=0,3 riichi-lab-bot local \
  --games 3 \
  --seed 20260730 \
  --device cuda:0 \
  --dtype bf16
```

本地测试会对每份 Observation 先执行 base64 序列化和反序列化，以模拟
RiichiLab 的 `request_action.observation`。四席共享一份模型权重，但各自
持有独立的 MJAI 状态机。每个输出动作都必须同时通过：

1. 当前 `Observation.legal_actions()` / `select_action_from_mjai()`；
2. 当前请求的 `possible_actions`。

CPU 正确性测试也可直接运行：

```bash
riichi-lab-bot local --games 1 --device cpu --dtype fp32
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
CUDA_DEVICE=0,3 riichi-lab-bot validate \
  --device cuda:0 \
  --dtype bf16
```

默认连接 `wss://game.riichi.dev/ws/validate`。客户端会等待
`validation_result`；通过时退出码为 0，失败或未收到通过结果时退出码为
2。validation 只运行一次，不会自动重试。

## Ranked 对局

机器人通过 validation 变成 active 后，可加入一场 ranked 半庄：

```bash
CUDA_DEVICE=0,3 riichi-lab-bot ranked \
  --games 1 \
  --device cuda:0 \
  --dtype bf16
```

连续排队：

```bash
CUDA_DEVICE=0,3 riichi-lab-bot ranked \
  --forever \
  --device cuda:0 \
  --dtype bf16
```

单场结束后客户端先断开再重新排队。异常断线使用 5、10、20、40、80、
120 秒上限的指数退避，避免同一 bot 的旧对局仍存在时触发并发连接保护。
按 `Ctrl-C` 可安全停止。

## 配置

所有子命令都支持：

| 参数或环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `--checkpoint` | `checkpoints/train_riichi_v10_sft/best_heuristic.pt` | checkpoint 文件 |
| `RIICHI_CHECKPOINT` | 未设置 | `--checkpoint` 未指定时覆盖默认模型 |
| `--device` | `auto` | `auto`、`cpu`、`cuda` 或 `cuda:index` |
| `--dtype` | `auto` | `auto`、`fp32` 或 `bf16` |
| `CUDA_DEVICE` | 未设置 | 在导入 PyTorch 前映射为 `CUDA_VISIBLE_DEVICES` |
| `--jsonl-log` | 未设置 | 追加写入结构化运行日志 |
| `--log-level` | `INFO` | `DEBUG`、`INFO`、`WARNING` 或 `ERROR` |
| `--url` | RiichiLab 官方 endpoint | 连接本地/自托管服务器时覆盖 |

配置优先级为 `--checkpoint`、`RIICHI_CHECKPOINT`、项目默认路径。模型在
进程启动时加载一次；运行期间 checkpoint 文件发生变化不会热更新当前
进程。

`auto` 在支持 BF16 的 CUDA 上选择 BF16，否则使用 FP32。该模型只支持
四人 Observation 和 schema 11；三人 Observation、错误 schema、缺失
`model_config` 或无法严格加载的权重都会在连接前报错。

## 协议与安全行为

- 只在 `request_action` 上响应，并始终原样回传当前 `request_id`。
- 其他 MJAI 事件只作为信息处理；未知事件、未知字段和二进制帧会被忽略。
- 模型动作通过 RiichiEnv 回转后，再与服务器 `possible_actions` 比较
  `type`、`pai`、`consumed` 和服务端提供时的 `tsumogiri`。
- 模型动作无法证明合法时，依次尝试合法的和牌、`none`、摸切和其他动作。
- 如果仍找不到双重合法的响应，客户端不会发送消息，由服务器执行无
  chombo 的默认动作。
- 如果本地处理时间已进入 `deadline_ms` 前 250 ms 的安全区，也不会发送
  可能变成迟到响应的动作。
- 日志记录 request id、动作类型、推理耗时和 `action_ack`，但不会记录
  token、Authorization header 或 base64 Observation。

相关服务端约定：

- [MJAI Protocol](https://riichi.dev/docs/protocol)
- [Bot Validation](https://riichi.dev/docs/validation)
- [Local Testing](https://riichi.dev/docs/local-testing)
- [Ranked Matches](https://riichi.dev/docs/ranked)
- [Matchmaking](https://riichi.dev/docs/matchmaking)

## 开发测试

```bash
conda run --no-capture-output -n Mahjong-AI \
  pytest -q riichi_lab_bot/tests
```

测试包括 checkpoint 严格加载、动作签名、红五/摸切、mock WebSocket、
validation/ranked 流程、deadline 不发送、完整本地半庄，以及独立单席
bridge 与训练 bridge 的可选逐项等价检查。

2026-07-30 的 GPU 验收结果（`CUDA_DEVICE=0,3`，BF16）：

| 轮次 | 用途 | 决策数 | 总耗时 | decisions/s | 推理 p50 / p95 / max |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | warm-up | 773 | 5.35 s | 144.57 | 4.02 / 8.52 / 259.13 ms |
| 2 | measured | 844 | 5.31 s | 158.85 | 3.42 / 7.47 / 9.82 ms |
| 3 | measured | 858 | 6.08 s | 141.21 | 3.71 / 8.35 / 16.71 ms |

第 2、3 轮合计 1702 个决策、11.39 秒、149.44 decisions/s；三轮均为
0 fallback、0 withheld、0 非法动作。
