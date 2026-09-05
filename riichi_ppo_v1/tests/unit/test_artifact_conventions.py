"""产物存储与评测机制固化的契约测试。"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
PPO_DIR = ROOT / "riichi_ppo_v1"
CONFIG_DIR = PPO_DIR / "configs"


def _git_check_ignored(path: str) -> bool:
    """返回给定仓库相对路径是否被 git 忽略。"""
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", path],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        content = yaml.safe_load(file)
    if not isinstance(content, dict):
        raise AssertionError(f"{path} 必须为映射配置")
    return content


def test_ray_logging_redirects_to_stderr(monkeypatch) -> None:
    """训练入口把 Ray/子进程日志收敛到 stderr,由脚本重定向到 `logs/<版本号>/`。"""
    from riichi_ppo_v1.training.train import configure_ray_stderr_logging

    monkeypatch.delenv("RAY_LOG_TO_STDERR", raising=False)
    configure_ray_stderr_logging()
    assert os.environ["RAY_LOG_TO_STDERR"] == "1"


def test_gitignore_allows_current_audit_type_dirs() -> None:
    """audit:design/report/scripts 入库,eval 与版本目录根散落文件忽略。"""
    allowed = (
        "audit/reports/v16/design/x.md",
        "audit/reports/v16/report/x.md",
        "audit/reports/v16/scripts/x.py",
        "audit/reports/v17/design/x.md",
        "audit/reports/v17/report/x.md",
        "audit/reports/v17/scripts/x.py",
    )
    ignored = (
        "audit/reports/v16/eval/x.json",
        "audit/reports/v17/eval/x.json",
        "audit/reports/v16/x.txt",
    )
    for path in allowed:
        assert not _git_check_ignored(path), f"{path} 应被 git 跟踪"
    for path in ignored:
        assert _git_check_ignored(path), f"{path} 应保持忽略"


def test_active_audit_version_dirs_have_fixed_types() -> None:
    """活跃 `audit/reports/<版本号>/` 只允许 design/report/eval/scripts 四类子目录。"""
    reports = ROOT / "audit" / "reports"
    for version in ("v16", "v17"):
        version_dir = reports / version
        assert version_dir.is_dir(), f"{version_dir} 不存在"
        entries = {path.name for path in version_dir.iterdir() if path.is_dir() and path.name != "__pycache__"}
        assert entries == {"design", "report", "eval", "scripts"}, (
            f"{version_dir} 应只含四个固定类型子目录,实际: {sorted(entries)}"
        )


def test_packaged_configs_are_current_and_neutral() -> None:
    from riichi_ppo_v1.sft.train import load_config as load_sft_config
    from riichi_ppo_v1.training.train import load_config

    training = load_config()
    assert training["policy_head_type"] == "current_state_snapshot"
    assert training["model_size"] == "v19"
    assert training["checkpoint_dir"] == "checkpoints/train_riichi_v19/ppo"
    sft = load_sft_config(CONFIG_DIR / "sft.yaml")
    assert sft["policy_head_type"] == "current_state_snapshot"
    assert sft["model_size"] == "v19"
    assert sft["context_tokens"] == 320
    assert sft["checkpoint_dir"] == "checkpoints/train_riichi_v19/sft"
    assert sft["belief_sft_coef"] == 1.0
    assert sft["belief_wait_danger_weight"] == 0.05


def test_current_datasets_present() -> None:
    datasets = ROOT / "datasets"
    assert (datasets / "tenhou_sft_2024_2025").is_dir()
    assert (datasets / "tenhou_sft_2024_2025_encoded_60pct_v16").is_dir()
    assert (datasets / "tenhou_grp_2024_2025_v17").is_dir()


def test_1v3_mechanism_constants() -> None:
    """1v3 机制常量(10/600/6000/5)在 mechanism.py 单一来源。"""
    from riichi_ppo_v1.evaluation import mechanism

    assert mechanism.REQUIRED_1V3_PROCESSES == 10
    assert mechanism.DEFAULT_1V3_HANCHANS_PER_PROCESS == 600
    assert mechanism.TOTAL_1V3_HANCHANS == 6000
    assert mechanism.DEFAULT_1V3_INTERVAL_UPDATES == 5


def test_head_to_head_cli_defaults_align_with_mechanism() -> None:
    """独立 1v3 CLI 默认值必须与固定机制一致(6000/600/seed 0)。"""
    from riichi_ppo_v1.evaluation.head_to_head_1v3 import _parser

    args = _parser().parse_args(
        ["--model-a", "a", "--model-b", "b", "--output", "out"]
    )
    assert args.hanchans == 6000
    assert args.parallel_hanchans == 600
    assert args.seed_base == 0


def test_progress_md_path_lives_in_report() -> None:
    """PROGRESS.md 落在 `audit/reports/<版本号>/report`,输出目录缺省时跳过。"""
    from riichi_ppo_v1.evaluation.mechanism import progress_md_path
    from riichi_ppo_v1.training.train import _progress_md_path

    assert progress_md_path("audit/reports/v17/eval") == Path(
        "audit/reports/v17/report/PROGRESS.md"
    )
    assert _progress_md_path({"eval1v3_output_dir": "audit/reports/v17/eval"}) == Path(
        "audit/reports/v17/report/PROGRESS.md"
    )
    assert _progress_md_path({}) is None


def test_sft_cadence_single_point() -> None:
    """SFT 节奏只在契约常量定义,默认/实验配置不得复制节奏键。"""
    from riichi_ppo_v1.sft.contract import (
        SFT_CADENCE_STEPS,
        SFT_FINAL_EVAL_HANCHAN_COUNT,
    )

    assert SFT_CADENCE_STEPS == 3000
    assert SFT_FINAL_EVAL_HANCHAN_COUNT == 96
    cadence_keys = {
        "validation_interval_steps",
        "checkpoint_interval_steps",
    }
    for name in ("sft.yaml",):
        config = _read_yaml(CONFIG_DIR / name)
        duplicated = cadence_keys & set(config)
        assert not duplicated, f"{name} 复制了节奏键: {sorted(duplicated)}"


def test_no_historical_locks_in_active_sources() -> None:
    """活跃源码/配置/文档不得再引用旧实验路径或旧 SFT API。"""
    needles = "|".join(
        (
            "riichi-sft-" + "audit",
            "validate_" + "v13_manifest",
            "validate_" + "v15_reused_manifest",
            "load_" + "v13_weights_only",
            "V13" + "PolicyAdapter",
            "PPO" + "PolicyAdapter",
            "train_riichi_" + "ppo_v14",
            "train_riichi_" + "v13_sft",
            "audit/reports/v14_ppo_20260812",
            "audit/reports/v15_ppo_20260814",
            "v13_sft_20260802",
            "v15_ppo_20260814",
        )
    )
    result = subprocess.run(
        [
            "rg", "-n", "--hidden", "-g", "!*.pyc",
            "-g", "!test_artifact_conventions.py",
            "-g", "!audit/reports/v16/report/PROGRESS.md",
            needles,
            "riichi_ppo_v1/configs",
            "riichi_ppo_v1/README.md",
            "riichi_ppo_v1/docs",
            "riichi_lab_bot/README.md",
            "AGENTS.md",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "", f"仍有历史版本锁:\n{result.stdout}"
