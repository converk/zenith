"""训练日志定时评测脚本。

功能：
    命令行入口。常驻扫描配置中的 checkpoint run 目录，读取 step 最大
    的 checkpoint 文件。当某个训练 run 相比上次评测又前进指定 global step 间隔时，
    自动安排 interval、top6_random、ranking_random 等四模型对局。结果写到
    evaluations/match_logs/，并刷新 Elo 排名文件。默认参数集中写在
    evaluations/config.py，命令行参数可临时覆盖。

使用方法：
    先按需要编辑 evaluations/config.py，然后运行：
    python -m evaluations.scheduled_checkpoint_match \\
      --poll-seconds 30 \\
      --num-games 80000 \\
      --num-envs 256 \\
      --cuda-device 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from evaluations import config as eval_config
from evaluations.core.elo import DEFAULT_RUN_DIRS, parse_run_dirs, refresh_elo_ranking
from evaluations.core.scheduler import (
    config_from_args,
    load_state,
    maybe_schedule_matches,
    save_state,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=[str(path) for path in DEFAULT_RUN_DIRS],
        help="only schedule matches for these checkpoint run directories",
    )
    parser.add_argument("--step-interval", type=int, default=eval_config.STEP_INTERVAL)
    parser.add_argument("--random-seed", type=int, default=eval_config.RANDOM_SEED)
    parser.add_argument("--initial-elo", type=float, default=eval_config.INITIAL_ELO)
    parser.add_argument("--k-factor", type=float, default=eval_config.K_FACTOR)
    parser.add_argument("--k-base-score", type=int, default=eval_config.K_BASE_SCORE)
    parser.add_argument("--k-scale-cap", type=float, default=eval_config.K_SCALE_CAP)
    parser.add_argument("--poll-seconds", type=int, default=eval_config.POLL_SECONDS)
    parser.add_argument("--num-games", type=int, default=eval_config.NUM_GAMES)
    parser.add_argument("--num-envs", type=int, default=eval_config.NUM_ENVS)
    parser.add_argument("--cuda-device", default=eval_config.CUDA_DEVICE)
    parser.add_argument("--progress-interval", type=int, default=eval_config.PROGRESS_INTERVAL)
    parser.add_argument(
        "--ranking-random-min-matches",
        type=int,
        default=eval_config.RANKING_RANDOM_MIN_MATCHES,
    )
    parser.add_argument(
        "--ranking-random-max-matches",
        type=int,
        default=eval_config.RANKING_RANDOM_MAX_MATCHES,
    )
    parser.add_argument("--output-dir", default=str(eval_config.MATCH_LOGS_DIR))
    parser.add_argument("--state-path", default=str(eval_config.SCHEDULED_STATE_PATH))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    config = config_from_args(args)
    state_path = Path(args.state_path)
    state = load_state(state_path)
    refresh_elo_ranking(
        logs_dir=Path(args.output_dir),
        run_dirs=parse_run_dirs(args.run_dirs),
        initial_elo=args.initial_elo,
        k_factor=args.k_factor,
        k_base_score=args.k_base_score,
        k_scale_cap=args.k_scale_cap,
    )

    while True:
        changed = maybe_schedule_matches(config, state)
        if changed:
            save_state(state_path, state)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
