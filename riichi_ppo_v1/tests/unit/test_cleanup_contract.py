from __future__ import annotations

import importlib
import inspect
from pathlib import Path

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


def test_riichi_lab_bot_retains_ranked_but_validate_path_is_independent() -> None:
    assert client.RANKED_URL == "wss://game.riichi.dev/ws/ranked"
    assert hasattr(client, "run_ranked")
    parser = cli.build_parser()
    ranked = parser.parse_args(["ranked"])
    assert ranked.command == "ranked"
    validate = parser.parse_args(["validate"])
    assert validate.command == "validate"
    assert validate.url == client.VALIDATION_URL
    combined = inspect.getsource(client) + inspect.getsource(cli)
    assert "def run_ranked" in combined
    assert "def play_connection" in combined
