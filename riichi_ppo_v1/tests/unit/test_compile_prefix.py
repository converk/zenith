"""torch.compile 键前缀剥离工具测试。"""

from __future__ import annotations

import torch
from torch import nn

from riichi_ppo_v1.model.checkpoint import strip_compile_prefix


def test_clean_state_dict_passthrough() -> None:
    """无前缀的 state_dict 原键浅拷贝返回(幂等)。"""
    state = {"a.weight": torch.zeros(1), "b": torch.ones(2)}
    stripped = strip_compile_prefix(state)
    assert set(stripped) == {"a.weight", "b"}
    assert stripped["a.weight"] is state["a.weight"]


def test_prefixed_keys_stripped() -> None:
    """全部键带 _orig_mod. 前缀时逐键剥离,张量为同一对象。"""
    state = {
        "_orig_mod.value_query": torch.zeros(2, 4),
        "_orig_mod.token_embedding.segment.weight": torch.ones(3),
    }
    stripped = strip_compile_prefix(state)
    assert set(stripped) == {"value_query", "token_embedding.segment.weight"}
    assert stripped["value_query"] is state["_orig_mod.value_query"]


def test_mixed_keys_keep_unprefixed() -> None:
    """混合键时无前缀键保持原样,不重复剥离。"""
    state = {"_orig_mod.a": torch.zeros(1), "b": torch.ones(1)}
    stripped = strip_compile_prefix(state)
    assert set(stripped) == {"a", "b"}


def test_compiled_module_keys_match_eager_after_strip() -> None:
    """compile 包装(backend=eager,免编译)的 state_dict 剥离后与裸模块键集一致。"""
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    compiled = torch.compile(model, backend="eager")
    raw_keys = set(model.state_dict())
    compiled_keys = set(compiled.state_dict())
    assert compiled_keys != raw_keys  # 前提:包装确实引入前缀差异
    stripped = strip_compile_prefix(compiled.state_dict())
    assert set(stripped) == raw_keys
