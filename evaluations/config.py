"""评测默认配置文件。

功能：
    集中配置定时评测、Elo 排名和 checkpoint 目录。平时如果只是想换
    需要评测的训练目录、评测局数、评测间隔、显卡编号等，优先改这里。
    命令行参数仍然可以临时覆盖这些默认值。

使用方法：
    直接编辑本文件，然后运行：
    python -m evaluations.scheduled_checkpoint_match
"""

from __future__ import annotations

from pathlib import Path


# 只关注这几个训练 run 目录；Elo 排名和定时评测都会过滤到这些目录内。
RUN_DIRS = (
    Path("checkpoints/ddp_mid_gae095_env64__1__1783427843"),
    Path("checkpoints/delta_done_mid_stable__1__1783263253"),
)

# 定时评测输出目录和状态文件。
MATCH_LOGS_DIR = Path("evaluations/match_logs")
SCHEDULED_STATE_PATH = MATCH_LOGS_DIR / "scheduled_state.json"

# 每隔多少 global step 触发一次定时评测。
STEP_INTERVAL = 6_000_000

RANDOM_SEED = 1

# 从当前 Elo 排行榜中随机抽取若干 checkpoint 对局。
# 抽取场数随候选模型数量增长，最少 2 场、最多 6 场。
RANKING_RANDOM_MIN_MATCHES = 2
RANKING_RANDOM_MAX_MATCHES = 6

# Elo 参数。
INITIAL_ELO = 50.0
K_FACTOR = 64.0
K_BASE_SCORE = 50_000
K_SCALE_CAP = 3.0

# 定时任务参数。
POLL_SECONDS = 30

# 对局评测参数。
NUM_GAMES = 75_000
NUM_ENVS = 256
CUDA_DEVICE = "1"
PROGRESS_INTERVAL = 5_000
