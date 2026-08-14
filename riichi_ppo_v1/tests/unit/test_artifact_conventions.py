"""产物存储与评测机制固化的契约测试。

覆盖宪法 III/IV/VI 的布局与机制约定:日志、audit、checkpoint、数据集、1v3 机制
与 SFT 节奏。路径与键位约定见 `specs/002-artifact-storage-eval/contracts/`。
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


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
