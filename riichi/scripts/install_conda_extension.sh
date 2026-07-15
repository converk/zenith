#!/usr/bin/env bash
# Install the local PyO3 extension into the *currently activated* Conda env.
#
# `maturin develop` otherwise follows whichever Python happens to be first on
# PATH.  Pinning PYO3_PYTHON avoids accidentally building for the system
# interpreter and then failing to import from Mahjong-AI.
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

"${python_bin}" -m maturin develop --release --manifest-path "${repo_root}/Cargo.toml"
"${python_bin}" -c 'import riichi; from riichi import MjaiKyokuStateMachineManager; MjaiKyokuStateMachineManager(1); print("riichi extension installed:", riichi.__file__)'
