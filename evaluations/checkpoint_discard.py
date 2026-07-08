"""单个 checkpoint 切牌检查脚本。

功能：
    加载一个 PPO checkpoint，输入一副 14 张手牌，查看模型会切哪张牌，
    并打印概率最高的若干候选切牌。也可以随机生成手牌做批量抽查。

使用方法：
    python -m evaluations.checkpoint_discard \\
      --checkpoint checkpoints/run/checkpoint_x.pt \\
      --hand '1m2m3m4m5m6m7m8m9m1s2s3s7s8s'
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import torch

from ppo.model import TileCountTransformerActorCritic, make_toy_ppo_config


TILE_NAMES = [
    *(f"{rank}m" for rank in range(1, 10)),
    *(f"{rank}p" for rank in range(1, 10)),
    *(f"{rank}s" for rank in range(1, 10)),
    "E",
    "S",
    "W",
    "N",
    "White",
    "Green",
    "Red",
]


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = list(run_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint_*.pt files found in {run_dir}")

    def checkpoint_step(path: Path) -> int:
        match = re.search(r"checkpoint_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    return max(checkpoints, key=checkpoint_step)


def random_hand_counts(rng: random.Random) -> list[int]:
    wall = [tile for tile in range(27,34) for _copy in range(4)]
    hand = rng.sample(wall, 14)
    counts = [0] * 34
    for tile in hand:
        counts[tile] += 1
    return counts


def format_hand(counts: list[int]) -> str:
    tiles: list[str] = []
    for tile, count in enumerate(counts):
        tiles.extend([TILE_NAMES[tile]] * count)
    return " ".join(tiles)


def parse_hand(text: str) -> list[int]:
    counts = [0] * 34
    compact = text.replace(",", " ").replace("/", " ")
    tokens = compact.split()
    if len(tokens) == 1:
        tokens = re.findall(r"\d[mps]|E|S|W|N|White|Green|Red", tokens[0])
    if len(tokens) != 14:
        raise ValueError(f"hand must contain 14 tiles, got {len(tokens)}: {tokens}")
    name_to_index = {name: index for index, name in enumerate(TILE_NAMES)}
    for token in tokens:
        if token not in name_to_index:
            raise ValueError(f"unknown tile {token!r}")
        tile = name_to_index[token]
        counts[tile] += 1
        if counts[tile] > 4:
            raise ValueError(f"too many copies of {token!r}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="checkpoints/ppo_config__1__1783094996",
        help="checkpoint run directory",
    )
    parser.add_argument(
        "--checkpoint",
        # lamda=95
        #default="checkpoints/delta_done_mid_stable__1__1783263253/checkpoint_840697344.pt",
        # lamda=98
        default="checkpoints/ddp_mid_gae095_env64__1__1783427843/checkpoint_862124032.pt",
        # lamda=98
        # default="checkpoints/lamda_98__1__1783269696/checkpoint_706252288.pt",
        help="checkpoint path; uses --run-dir latest when omitted or empty",
    )
    parser.add_argument("--num-hands", type=int, default=1, help="number of random hands to test")
    parser.add_argument(
        "--hand",
        default=None,
        help="manual 14-tile hand, e.g. '1m2m3m1p2p3p1s2s3s7m8m9m7s8s'",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(Path(args.run_dir))
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # torch.load uses pickle internally; only load checkpoints you trust.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_size = checkpoint.get("args", {}).get("model_size", "medium")
    agent = TileCountTransformerActorCritic(make_toy_ppo_config(model_size)).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    rng = random.Random(args.seed)
    print(f"checkpoint: {checkpoint_path}")
    print(f"model_size: {model_size}")
    print("tile mapping: 0-8=1m-9m, 9-17=1p-9p, 18-26=1s-9s, 27-33=E/S/W/N/White/Green/Red")

    hands = [parse_hand(args.hand)] if args.hand else [
        random_hand_counts(rng) for _ in range(args.num_hands)
    ]
    for hand_index, counts in enumerate(hands, start=1):
        obs = torch.tensor([counts], dtype=torch.float32, device=device)
        legal_mask = torch.tensor([[count > 0 for count in counts]], dtype=torch.bool, device=device)

        with torch.no_grad():
            outputs = agent(obs, legal_mask)
            logits = outputs["policy_logits"][0]
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.argmax(logits).item())
            top_values, top_indices = torch.topk(probs, k=min(5, int(legal_mask.sum().item())))

        print()
        print(f"hand {hand_index}: {format_hand(counts)}")
        print(f"discard: index={action} tile={TILE_NAMES[action]} prob={float(probs[action]):.4f}")
        print("top candidates:")
        for probability, tile in zip(top_values.tolist(), top_indices.tolist()):
            print(f"  index={tile:2d} tile={TILE_NAMES[tile]:>5s} prob={probability:.4f}")


if __name__ == "__main__":
    main()
