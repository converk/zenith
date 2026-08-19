"""固定 1v3 评测机制常量的单一来源。

这些是机制常量:修改必须走宪法修订(宪法原则 IV),禁止在实验配置或 CLI 默认值
中悄悄改变。可变项(对手模型、种子基数、设备、输出目录)一律由版本配置/CLI
提供,不在本模块提供任何锁定历史版本的默认值。

2026-08-19 宪法修订(1.5.0):PPO 1v3 机制由 1600 semi-hanchan / 每 30 updates
调整为 4000 hanchan / 每 5 updates(进程数保持 10,单进程 400)。
"""

from __future__ import annotations

from pathlib import Path


REQUIRED_1V3_PROCESSES = 10
DEFAULT_1V3_HANCHANS_PER_PROCESS = 400
TOTAL_1V3_HANCHANS = REQUIRED_1V3_PROCESSES * DEFAULT_1V3_HANCHANS_PER_PROCESS
DEFAULT_1V3_INTERVAL_UPDATES = 5


def progress_md_path(output_dir: str | Path) -> Path:
    """推导 1v3 进度报告路径:`audit/reports/<版本号>/report/PROGRESS.md`。"""
    return Path(output_dir).parent / "report" / "PROGRESS.md"
