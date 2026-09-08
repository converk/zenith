# akagi_bridge — Akagi v3 云端推理适配服务

把 Zenith 的 V18 checkpoint 变成 **Akagi v3** 可以直接调用的远程推理服务。

Akagi 的 `bot.api`(设置里的「云端推理」)在每次决策前 POST 一条**从
`start_game` 起的完整 mjai 事件流**(对手手牌隐藏为 `"?"`),本服务把它重建成
RiichiEnv 的当前局面 `Observation`,复用 `riichi_lab_bot` 的
`OnlineStateBridge` + `PolicyEngine` 得到 V18 动作,再解码回 mjai JSON 返回。

> 只在**能跑模型的那台机器**上启动(通常是训练/推理服务器);Akagi 本体
> (抓包 + HUD)始终跑在游戏所在的本机。

## 数据流

```
Akagi(本机)  --POST /v3/react {player_id, events:[mjai...]}-->  akagi_bridge(服务器)
                                                                    │
                          RiichiEnv.apply_event × N  →  Observation   │
                          OnlineStateBridge.prepare → V18 Actor 输入  │
                          PolicyEngine.infer        → action_id       │
                          bridge.decode             → mjai 动作 JSON  │
                                                                    ▼
Akagi(本机)  <-- {reaction, candidates, model} ---------------------┘
```

## 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v3/react` | 决策。请求 `{"model"?: str, "player_id": 0-3, "events": [mjai…]}` |
| `GET` | `/v3/models` | `{"models":[{"id","game","desc"}]}` |
| `GET` | `/v3/key` | 密钥状态(`plan`/`expires_at`/`topk`…) |
| `GET` | `/healthz` | 无鉴权存活探针 |

响应示例:

```json
{
  "reaction": {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": false},
  "candidates": [{"action": "dahai:5m", "prob": 0.71}],
  "model": "zenith-v18"
}
```

**失败即让 Akagi 兜底**:任何非 2xx 响应都会让 Akagi 立刻改用它内嵌的本地
模型,不会让对局卡住。因此本服务在无法给出合法动作时显式返回 4xx/5xx
(例如事件流末尾不在决策窗口、事件被 RiichiEnv 拒绝、语义校验失败),
而不是猜一个动作。

## 安装

```bash
conda activate Mahjong-AI
# 原生扩展(riichienv / riichi)按 RiichiEnv 的脚本安装:
bash RiichiEnv/riichienv-state-machine/scripts/install_conda_extension.sh
bash RiichiEnv/scripts/install_conda_extension.sh
# 两个 Python 包一起装(akagi_bridge 运行时依赖 riichi_lab_bot)
python -m pip install -e ./riichi_lab_bot -e ./akagi_bridge
```

## 启动

```bash
export AKAGI_API_KEY='<你自己设的任意字符串>'
akagi-bridge \
  --checkpoint checkpoints/train_riichi_v18/ppo/best.pt \
  --device cuda:0 --dtype fp32 \
  --host 0.0.0.0 --port 8090 \
  --api-key "$AKAGI_API_KEY" \
  --topk 5 \
  --jsonl-log logs/v18/akagi-bridge.jsonl
```

常用参数:`--host`(默认 `127.0.0.1`,要让本机 Akagi 连就得用 `0.0.0.0` 或
内网地址)、`--port`、`--api-key`(缺省读 `$AKAGI_API_KEY`,为空则不校验)、
`--topk`(0 关闭候选,HUD 只显示推荐动作)、`--rule`
(`tenhou`/`mjsoul`,决定从 mjai 重建 Observation 时用的规则集)、
`--model-id`/`--model-desc`(HUD 上显示的模型名)。

## Akagi 侧配置

Akagi 的 `configs/config.toml`:

```toml
[general]
developer_mode = true          # 解锁 UI 里的推理服务器地址编辑器

[bot]
active_4p = "akagi-native"     # 内置 bot 负责走云端 API 与兜底
active_3p = "akagi-native3p"

[bot.api]
enabled = true
base_url = "http://<服务器地址>:8090"
key = "<与 --api-key 相同的字符串>"
model_4p = "zenith-v18"
react_timeout_ms = 3000        # 500–10000,按你的网络与模型延迟调
```

`is_active()` 要求 `enabled` + `base_url` + `key` 三者齐全;`key` 只是 Bearer
令牌,自建服务用它做鉴权,填任意非空字符串即可。

## 自检

```bash
curl -s http://127.0.0.1:8090/healthz
curl -s -H "Authorization: Bearer $AKAGI_API_KEY" http://127.0.0.1:8090/v3/models

curl -s -X POST http://127.0.0.1:8090/v3/react \
  -H "Authorization: Bearer $AKAGI_API_KEY" -H 'Content-Type: application/json' \
  -d '{"player_id":0,"events":[
        {"type":"start_game","names":["a","b","c","d"]},
        {"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyoutaku":0,"oya":0,
         "scores":[25000,25000,25000,25000],"dora_marker":"1p",
         "tehais":[["1m","1m","1m","2m","3m","4m","5m","6m","7m","8m","9m","9m","9m"],
                   ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
                   ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
                   ["?","?","?","?","?","?","?","?","?","?","?","?","?"]]},
        {"type":"tsumo","actor":0,"pai":"5m"}]}'
```

## 已知边界

- **仅四麻**。V18 是 4 席模型;三麻对局会被拒(HTTP 422),Akagi 自动退回内置 bot。
- **立直是两步**。模型选 `reach` 时返回的是 `{"type":"reach","actor":N}`(不含切牌),
  Akagi 会把这条事件追加到流里再问一次切牌;本服务按同一逻辑重建 Observation。
- **延迟预算**。每次请求要重放整局 mjai(单局几百条)+ 一次 forward。
  Akagi 侧 `react_timeout_ms` 默认 3s、上限 10s;超过 `SLOW_DECISION_MS`
  会打 WARNING 日志便于定位。
- **无状态**。每个请求独立重建状态,不依赖上一请求;因此重启服务不会影响进行中的对局。

## 测试

```bash
conda run -n Mahjong-AI python -m pytest akagi_bridge/tests
```

`tests/test_app.py` 是纯 HTTP 契约测试(不需要原生扩展/权重);
`tests/test_engine.py` 需要 `riichienv`/`riichi`/torch,缺失时自动跳过。
