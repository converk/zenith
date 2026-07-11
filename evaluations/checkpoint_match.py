"""checkpoint 模型对局评测脚本。

功能：
    命令行入口。支持两种模式：
    1. 传 --checkpoint-a/--checkpoint-b 时，两个模型各控制两个座位。
    2. 传 --checkpoints 四个路径时，四个模型各控制一个座位。
    胡牌玩家所属模型加 1 分，同一巡多人胡牌会分别加分，流局只计入 draws。

使用方法：
    python -m evaluations.checkpoint_match \\
      --checkpoint-a checkpoints/run_a/checkpoint_x.pt \\
      --checkpoint-b checkpoints/run_b/checkpoint_y.pt
"""

from __future__ import annotations

import argparse

from evaluations import config as eval_config
from evaluations.core.match import NUM_PLAYERS, parse_seats, run_checkpoint_match, run_multi_checkpoint_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-a",
        default="checkpoints/delta_done_mid_stable__1__1783263253/checkpoint_840697344.pt",
    )
    parser.add_argument(
        "--checkpoint-b",
        default="checkpoints/ddp_mid_gae095_env64__1__1783427843/checkpoint_862124032.pt",
    )
    parser.add_argument(
        "--checkpoints",
        nargs=4,
        default=None,
        help="four checkpoint paths; each one controls one seat",
    )
    parser.add_argument("--num-games", type=int, default=eval_config.NUM_GAMES)
    parser.add_argument("--num-envs", type=int, default=eval_config.NUM_ENVS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--seats-a",
        type=lambda value: parse_seats(value, NUM_PLAYERS),
        default=(0, 2),
        help="two seats for model A",
    )
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-interval", type=int, default=eval_config.PROGRESS_INTERVAL)
    args = parser.parse_args()

    if args.checkpoints is not None:
        run_multi_checkpoint_match(
            checkpoints=args.checkpoints,
            num_games=args.num_games,
            num_envs=args.num_envs,
            seed=args.seed,
            use_cuda=args.cuda,
            progress_interval=args.progress_interval,
        )
    else:
        run_checkpoint_match(
            checkpoint_a=args.checkpoint_a,
            checkpoint_b=args.checkpoint_b,
            num_games=args.num_games,
            num_envs=args.num_envs,
            seed=args.seed,
            seats_a=args.seats_a,
            use_cuda=args.cuda,
            progress_interval=args.progress_interval,
        )


if __name__ == "__main__":
    main()
