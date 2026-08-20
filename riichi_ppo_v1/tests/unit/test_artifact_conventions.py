"""产物存储与评测机制固化的契约测试。

覆盖宪法 III/IV/VI 的布局与机制约定:日志、audit、checkpoint、数据集、1v3 机制
与 SFT 节奏。路径与键位约定见 `specs/002-artifact-storage-eval/contracts/`。
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
PPO_DIR = ROOT / "riichi_ppo_v1"
CONFIG_DIR = PPO_DIR / "configs"


def _git_check_ignored(path: str) -> bool:
    """返回给定仓库相对路径是否被 git 忽略。"""
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", path],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _read_yaml(path: Path) -> dict:
    """读取 YAML 配置;非映射内容按契约测试需求视为错误。"""
    with path.open(encoding="utf-8") as file:
        content = yaml.safe_load(file)
    if not isinstance(content, dict):
        raise AssertionError(f"{path} 必须为映射配置")
    return content


def test_sft_audit_report_default_is_none() -> None:
    """`sft/audit.py` 的 `--report` 默认不落盘,不得写 `logs/` 根目录。"""
    from riichi_ppo_v1.sft.audit import build_parser

    args = build_parser().parse_args([])
    assert args.report is None


def test_ray_logging_redirects_to_stderr(monkeypatch) -> None:
    """训练入口把 Ray/子进程日志收敛到 stderr,由脚本重定向到 `logs/<版本号>/`。"""
    from riichi_ppo_v1.training.train import configure_ray_stderr_logging

    monkeypatch.delenv("RAY_LOG_TO_STDERR", raising=False)
    configure_ray_stderr_logging()
    assert os.environ["RAY_LOG_TO_STDERR"] == "1"


def test_gitignore_allows_audit_type_dirs() -> None:
    """audit 方案 A:design/report/scripts 入库,eval 与版本目录根散落文件忽略。"""
    allowed = (
        "audit/reports/v15/design/x.md",
        "audit/reports/v15/report/x.md",
        "audit/reports/v15/scripts/x.py",
    )
    ignored = (
        "audit/reports/v15/eval/x.json",
        "audit/reports/v15/x.txt",
    )
    for path in allowed:
        assert not _git_check_ignored(path), f"{path} 应被 git 跟踪"
    for path in ignored:
        assert _git_check_ignored(path), f"{path} 应保持忽略"


def test_audit_version_dirs_have_fixed_types() -> None:
    """`audit/reports/<版本号>/` 只允许 design/report/eval/scripts 四类子目录。"""
    reports = ROOT / "audit" / "reports"
    for version in ("v13", "v14", "v15", "v16"):
        version_dir = reports / version
        assert version_dir.is_dir(), f"{version_dir} 不存在"
        entries = {path.name for path in version_dir.iterdir()}
        assert entries == {"design", "report", "eval", "scripts"}, (
            f"{version_dir} 应只含四个固定类型子目录,实际: {sorted(entries)}"
        )


def test_checkpoint_layout_uses_train_riichi_versions() -> None:
    """checkpoint 顶层必须为 `train_riichi_<版本号>` 规范布局。"""
    checkpoints = ROOT / "checkpoints"
    names = {path.name for path in checkpoints.iterdir() if path.is_dir()}
    assert names == {
        "train_riichi_v13",
        "train_riichi_v14",
        "train_riichi_v15",
        "train_riichi_v16",
        "train_riichi_v17",
    }


def test_sft_checkpoint_default_is_neutral() -> None:
    """SFT 代码默认 checkpoint 目录必须中性,不得锁定历史版本。"""
    from riichi_ppo_v1.sft.train import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["checkpoint_dir"] == "checkpoints/train_riichi_current"


def test_no_legacy_checkpoint_paths_in_tracked_sources() -> None:
    """三组件受版本控制源码不得引用旧 checkpoint 路径(拼接避免自引用)。"""
    legacy_v14 = "train_riichi_" + "ppo_v14"
    legacy_v13 = "train_riichi_" + "v13_sft"
    result = subprocess.run(
        [
            "rg", "-n", "--hidden",
            "-g", "!*.pyc", "-g", "!*.pt", "-g", "!*.jsonl", "-g", "!*.log",
            "-g", "!test_artifact_conventions.py",
            f"{legacy_v14}|{legacy_v13}",
            "riichi_ppo_v1", "riichi_lab_bot", "RiichiEnv",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "", f"仍有旧 checkpoint 路径引用:\n{result.stdout}"


def test_prepare_archive_dir_is_required() -> None:
    """`prepare.py --archive-dir` 必填,默认值不得重建已废弃目录。"""
    from riichi_ppo_v1.sft.prepare import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "datasets/example"])
    args = parser.parse_args(
        ["--output", "datasets/example", "--archive-dir", "datasets/raw"]
    )
    assert args.archive_dir == Path("datasets/raw")


def test_current_datasets_present_and_obsolete_absent() -> None:
    """现行数据集保留,两个废弃数据集不存在。"""
    datasets = ROOT / "datasets"
    assert (datasets / "tenhou_sft_2024_2025").is_dir()
    assert (datasets / "tenhou_sft_2024_2025_encoded_40pct_v13_v16").is_dir()
    assert not (datasets / "tenhou-to-mjai").exists()
    assert not (
        datasets / "tenhou_sft_2024_2025_encoded_remaining_80pct_v11"
    ).exists()


def test_1v3_mechanism_constants() -> None:
    """1v3 机制常量(10/400/4000/5)在 mechanism.py 单一来源。"""
    from riichi_ppo_v1.evaluation import mechanism

    assert mechanism.REQUIRED_1V3_PROCESSES == 10
    assert mechanism.DEFAULT_1V3_HANCHANS_PER_PROCESS == 400
    assert mechanism.TOTAL_1V3_HANCHANS == 4000
    assert mechanism.DEFAULT_1V3_INTERVAL_UPDATES == 5


def test_head_to_head_cli_defaults_align_with_mechanism() -> None:
    """独立 1v3 CLI 默认值必须与固定机制一致(4000/400/seed 0)。"""
    from riichi_ppo_v1.evaluation.head_to_head_1v3 import _parser

    args = _parser().parse_args(
        ["--model-a", "a", "--model-b", "b", "--output", "out"]
    )
    assert args.hanchans == 4000
    assert args.parallel_hanchans == 400
    assert args.seed_base == 0


def test_progress_md_path_lives_in_report() -> None:
    """PROGRESS.md 落在 `audit/reports/<版本号>/report`,输出目录缺省时跳过。"""
    from riichi_ppo_v1.evaluation.mechanism import progress_md_path
    from riichi_ppo_v1.training.train import _progress_md_path

    assert progress_md_path("audit/reports/v15/eval") == Path(
        "audit/reports/v15/report/PROGRESS.md"
    )
    assert _progress_md_path({"eval1v3_output_dir": "audit/reports/v15/eval"}) == Path(
        "audit/reports/v15/report/PROGRESS.md"
    )
    assert _progress_md_path({}) is None


def test_eval_output_dirs_match_version_convention() -> None:
    """版本配置的 `eval1v3_output_dir` 必须为 `audit/reports/<版本号>/eval`。"""
    for name in ("v14_ppo.yaml", "v14_ppo_resume.yaml", "v15_ppo.yaml"):
        config = _read_yaml(CONFIG_DIR / name)
        assert re.fullmatch(
            r"audit/reports/v[0-9]+/eval", config["eval1v3_output_dir"]
        ), config["eval1v3_output_dir"]


def test_sft_cadence_single_point() -> None:
    """SFT 节奏只在 sft.yaml(及契约常量)定义,实验配置零复制。"""
    from riichi_ppo_v1.sft.contract import (
        SFT_CADENCE_STEPS,
        SFT_FINAL_EVAL_HANCHAN_COUNT,
    )

    assert SFT_CADENCE_STEPS == 3000
    assert SFT_FINAL_EVAL_HANCHAN_COUNT == 96
    sft = _read_yaml(CONFIG_DIR / "sft.yaml")
    assert sft["validation_interval_steps"] == SFT_CADENCE_STEPS
    assert sft["checkpoint_interval_steps"] == SFT_CADENCE_STEPS
    assert sft["heuristic_evaluation_interval_steps"] == SFT_CADENCE_STEPS
    assert (
        sft["heuristic_evaluation_hanchan_count"]
        == SFT_FINAL_EVAL_HANCHAN_COUNT
    )
    assert (
        sft["heuristic_evaluation_final_hanchan_count"]
        == SFT_FINAL_EVAL_HANCHAN_COUNT
    )
    cadence_keys = {
        "validation_interval_steps",
        "checkpoint_interval_steps",
        "heuristic_evaluation_interval_steps",
        "heuristic_evaluation_hanchan_count",
        "heuristic_evaluation_final_hanchan_count",
        "heuristic_evaluation_enabled",
    }
    for name in ("v15_sft_offense_warmup.yaml", "v15_sft_actor_finetune.yaml"):
        config = _read_yaml(CONFIG_DIR / name)
        duplicated = cadence_keys & set(config)
        assert not duplicated, f"{name} 复制了节奏键: {sorted(duplicated)}"


def test_no_historical_locks_in_configs_and_docs() -> None:
    """配置与文档不得锁定历史版本/日期目录/废弃数据集(拼接避免自引用)。"""
    needles = "|".join(
        (
            "train_riichi_" + "ppo_v14",
            "train_riichi_" + "v13_sft",
            "80pct_v11",
            "tenhou-to-mjai",
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
            needles,
            "riichi_ppo_v1/configs",
            "riichi_ppo_v1/README.md",
            "riichi_ppo_v1/docs",
            "riichi_lab_bot/README.md",
            "AGENTS.md",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "", f"仍有历史版本锁:\n{result.stdout}"
