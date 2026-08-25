#!/usr/bin/env bash
# Build the local RiichiEnv wheel for the active Conda interpreter without
# invoking pip dependency resolution (which is unnecessary for this workspace
# and can fail behind an offline/enterprise index).
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "error: activate the target Conda environment first (for example: conda activate Mahjong-AI)" >&2
    exit 2
fi

python_bin="${CONDA_PREFIX}/bin/python"
if [[ ! -x "${python_bin}" ]]; then
    echo "error: no Python executable at ${python_bin}" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYO3_PYTHON="${python_bin}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

cd "${repo_root}"
# 主扩展 riichienv(PyO3 绑定)构建安装;状态机作为其依赖被静态链接。
"${python_bin}" -m maturin build --release
wheel=(target/wheels/riichienv-*.whl)
"${python_bin}" -m pip install --no-deps --force-reinstall "${wheel[0]}"
# riichi 状态机模块必须与主扩展同源安装:Python 侧会独立 import riichi,
# 若只装 riichienv,状态机逻辑会停留在旧编译产物(两者必须同一次构建)。
"${python_bin}" -m maturin build -m riichienv-state-machine/Cargo.toml --release
state_wheel=(target/wheels/riichienv_state_machine-*.whl)
"${python_bin}" -m pip install --no-deps --force-reinstall "${state_wheel[0]}"
"${python_bin}" -c 'from riichienv import BatchedRiichiEnv; import riichi; print("RiichiEnv extension installed:", BatchedRiichiEnv, "| state machine:", riichi.MjaiKyokuStateMachineManager)'
