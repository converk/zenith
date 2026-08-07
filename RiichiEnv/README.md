# RiichiEnv

Zenith 使用的 Riichi mahjong 训练环境。核心逻辑位于 Rust crate
`riichienv-core`，Python 绑定位于 `riichienv-python`，另有独立的 MJAI
协议状态机子包 `riichi`（Python 模块名仍为 `riichi`）。

## 目录结构

```text
RiichiEnv/
  riichienv-core/     # 规则、手牌、得分、环境状态机
  riichienv-python/   # PyO3 Python 绑定
  riichi/             # MJAI 协议状态转换与保存（独立于 riichienv）
  src/riichienv/      # Python 公共 API
  tests/              # 环境与协议测试
  scripts/            # 安装与验证脚本
```

本项目不包含 ML/UI/WASM/visualizer/agents 等上游组件；训练与 bot 逻辑位于
仓库顶层 `riichi_ppo_v1/` 与 `riichi_lab_bot/`。

## 安装

```bash
conda activate Mahjong-AI
bash scripts/install_conda_extension.sh
bash riichi/scripts/install_conda_extension.sh
```

## 使用

```python
from riichienv import RiichiEnv

env = RiichiEnv(game_mode="4p-red-half")
obs = env.reset()
```

`riichienv.RiichiEnv`、`BatchedRiichiEnv`、`MjaiReplay`、`HandEvaluator`、
`calculate_shanten` 等公共 API 均保持不变。

## 测试

```bash
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest tests -q
```
