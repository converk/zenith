from __future__ import annotations

import torch
import pytest

from riichi_ppo_v1.training.inference import inference_autocast_config


def test_inference_dtype_fp32_disables_autocast() -> None:
    dtype, enabled = inference_autocast_config("fp32", device_type="cuda")
    assert dtype is torch.float32
    assert enabled is False


def test_inference_dtype_bf16_uses_hardware_gate() -> None:
    dtype, enabled = inference_autocast_config("bf16", device_type="cuda")
    assert dtype is torch.bfloat16
    assert enabled is bool(torch.cuda.is_bf16_supported())


def test_inference_dtype_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="inference_dtype"):
        inference_autocast_config("fp16", device_type="cuda")
