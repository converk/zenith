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
"${python_bin}" -m maturin build --release
wheel=(target/wheels/riichienv-*.whl)
"${python_bin}" -m pip install --no-deps --force-reinstall "${wheel[0]}"
"${python_bin}" -c 'from riichienv import BatchedRiichiEnv; print("RiichiEnv batch extension installed:", BatchedRiichiEnv)'
