# Development Guide

本指南只覆盖 Zenith 保留的 Rust/Python 核心开发流程；上游
ML/UI/WASM/visualizer/agents 组件已从本仓库移除。

## 前置依赖

- Rust（cargo/rustc）
- Python 3.10+
- `maturin`

## Workspace 结构

```text
riichienv-core/   # 纯 Rust：规则、手牌、得分、环境状态机
riichienv-python/ # PyO3 绑定，依赖 riichienv-core 的 python feature
riichienv-state-machine/ # 独立 Rust/Python 子包：MJAI 协议状态转换与保存
src/riichienv/    # Python 公共 API
tests/            # pytest 测试
scripts/          # 安装与日志验证脚本
```

`riichienv-state-machine` 与 `riichienv` 通过独立类型/序列化边界交互，不互相依赖。

## Rust 开发

```bash
cargo check -p riichienv-core
cargo test -p riichienv-core
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

## Python 扩展

```bash
python -m maturin develop --release
```

或使用仓库脚本：

```bash
bash scripts/install_conda_extension.sh
bash riichienv-state-machine/scripts/install_conda_extension.sh
```

## Python 测试

```bash
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest tests -q
```
