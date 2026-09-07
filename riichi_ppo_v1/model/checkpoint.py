"""Checkpoint 键规范工具:剥离 torch.compile 包装保存的 ``_orig_mod.`` 前缀。

torch.compile 包装后的模块 ``state_dict()`` 键带 ``_orig_mod.`` 前缀;
SFT 保存端(sft/checkpoint.py)已修复为保存解包后的键,本工具供加载端
兼容既有带前缀工件(如 V19 SFT fuzzy 运行的 best.pt)。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_COMPILE_PREFIX = "_orig_mod."


def strip_compile_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """返回剥离 ``_orig_mod.`` 前缀后的键值;无前缀时按原键浅拷贝返回。"""
    if not any(key.startswith(_COMPILE_PREFIX) for key in state_dict):
        return dict(state_dict)
    return {
        (key[len(_COMPILE_PREFIX):] if key.startswith(_COMPILE_PREFIX) else key): value
        for key, value in state_dict.items()
    }
