"""固定 1v3 评测机制常量的单一来源。

这些是机制常量:修改必须走宪法修订(宪法原则 IV),禁止在实验配置或 CLI 默认值
中悄悄改变。可变项(对手模型、种子基数、设备、输出目录)一律由版本配置/CLI
提供,不在本模块提供任何锁定历史版本的默认值。

2026-08-19 宪法修订(1.6.0):PPO 1v3 机制调整为 6000 hanchan / 每 5 updates
(12 进程 × 500,双卡各 6 进程)。
2026-08-20 宪法修订(1.7.0):PPO 1v3 机制调整为 4000 hanchan / 每 5 updates
(10 进程 × 400,双卡各 5 进程);分片内部并行度(每批半庄数)由版本配置提供,
不属于机制常量。
2026-08-29 宪法修订(1.9.0):PPO 1v3 机制调整为 6000 hanchan / 每 5 updates
(10 进程 × 600,双卡各 5 进程);分片内部并行度(每批半庄数)与种子基由版本
配置提供,不属于机制常量。
"""

from __future__ import annotations

from pathlib import Path


REQUIRED_1V3_PROCESSES = 10
DEFAULT_1V3_HANCHANS_PER_PROCESS = 600
TOTAL_1V3_HANCHANS = REQUIRED_1V3_PROCESSES * DEFAULT_1V3_HANCHANS_PER_PROCESS
DEFAULT_1V3_INTERVAL_UPDATES = 5


def progress_md_path(output_dir: str | Path) -> Path:
    """推导 1v3 进度报告路径:`audit/reports/<版本号>/report/PROGRESS.md`。"""
    return Path(output_dir).parent / "report" / "PROGRESS.md"
