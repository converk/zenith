from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

import riichi_lab_bot.client as client
from riichi_lab_bot import cli
from riichi_ppo_v1.sft.train import load_config as load_sft_config
from riichi_ppo_v1.training.train import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAMES = ("training.yaml", "monitoring.yaml", "sft.yaml")


def test_remaining_tools_are_importable_and_covered() -> None:
    tools = sorted((ROOT / "tools").glob("*.py"))
    assert tools
    for tool in tools:
        if tool.name == "__init__.py":
            continue
        module = importlib.import_module(f"riichi_ppo_v1.tools.{tool.stem}")
        assert hasattr(module, "canonical_step_events") or hasattr(module, "main")


def test_remaining_configs_are_loadable() -> None:
    training = load_config()
    assert training["policy_head_type"] == "isolated_action_query"
    assert training["checkpoint_dir"] == "checkpoints/train_riichi_ppo"
    sft = load_sft_config(ROOT / "configs" / "sft.yaml")
    assert sft["policy_head_type"] == "isolated_action_query"
    assert sft["checkpoint_dir"] == "checkpoints/train_riichi_v13_sft"
    for name in CONFIG_NAMES:
        assert (ROOT / "configs" / name).exists()


def test_riichi_lab_bot_has_no_ranked_execution_path() -> None:
    combined = inspect.getsource(client) + inspect.getsource(cli)
    assert "wss://game.riichi.dev/ws/ranked" not in combined
    assert "RANKED_URL" not in combined
    assert "run_ranked" not in combined
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["ranked"])
