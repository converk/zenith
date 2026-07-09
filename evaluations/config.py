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

RANDOM_SEED = 1

# 每一轮处理完所有 runs 后的休息时间，避免常驻任务空转。
LOOP_SLEEP_SECONDS = 3

# batch 评测次数。每个小 match 都会重新抽取 4 个模型，每个小 match 只打 NUM_GAMES 局。
# 常规 batch 每轮都会跑；new batch 只在发现 run 下出现更新的 checkpoint 时跑。
REGULAR_SELF_MATCHES = 20
REGULAR_RANKING_MATCHES = 40
NEW_SELF_MATCHES = 10
NEW_RANKING_MATCHES = 20

# Elo 参数。
INITIAL_ELO = 50.0
K_FACTOR = 32.0
K_BASE_SCORE = 100
K_SCALE_CAP = 2.0

# 对局评测参数。这里的 NUM_GAMES 是每个小 match 的局数，不是整个 batch 的总局数。
NUM_GAMES = 10
NUM_ENVS = 256
CUDA_DEVICE = "1"

# 手动运行 evaluations.checkpoint_match 时，每多少局打印一次进度。
# 自动 batch 评测内部会关闭单场进度打印。
PROGRESS_INTERVAL = 5_000
