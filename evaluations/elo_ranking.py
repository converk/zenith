"""根据 match 日志计算 checkpoint Elo 排名。

功能：
    命令行入口。扫描 evaluations/match_logs/ 下的新 batch 对局日志，
    读取每一行 MATCH_RESULT，把每个 checkpoint 当成独立选手，刷新 Elo
    排名，并输出 JSON/CSV 文件。
    默认日志目录、run 目录和 Elo 参数来自 evaluations/config.py。

CSV 字段：
    rank: 当前 Elo 排名，1 表示最高。
    elo: 当前 checkpoint 的 Elo 分数。
    matches: 该 checkpoint 参与过的 match 日志数量。
    score_for: 该 checkpoint 所属模型累计胡牌得分。
    score_against: 对手累计胡牌得分。
    score_rate: score_for / (score_for + score_against)，累计得分率。
    run_name: checkpoint 所属训练 run 目录名。
    step: checkpoint 文件名里的 global step。
    checkpoint: checkpoint 文件路径。

使用方法：
    python -m evaluations.elo_ranking
    python -m evaluations.elo_ranking --k-factor 64 --initial-elo 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluations import config as eval_config
from evaluations.core.elo import parse_run_dirs, print_ranking, refresh_elo_ranking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=str(eval_config.MATCH_LOGS_DIR))
    parser.add_argument("--run-dirs", nargs="+", default=None)
    parser.add_argument("--initial-elo", type=float, default=eval_config.INITIAL_ELO)
    parser.add_argument("--k-factor", type=float, default=eval_config.K_FACTOR)
    parser.add_argument("--k-base-score", type=int, default=eval_config.K_BASE_SCORE)
    parser.add_argument("--k-scale-cap", type=float, default=eval_config.K_SCALE_CAP)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    records = refresh_elo_ranking(
        logs_dir=Path(args.logs_dir),
        run_dirs=parse_run_dirs(args.run_dirs),
        initial_elo=args.initial_elo,
        k_factor=args.k_factor,
        k_base_score=args.k_base_score,
        k_scale_cap=args.k_scale_cap,
    )
    print_ranking(records, args.limit)


if __name__ == "__main__":
    main()
