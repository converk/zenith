"""对局评测核心逻辑。

功能：
    提供 checkpoint 加载、动作选择、并行环境对局统计等可复用函数。
    命令行脚本 evaluations.checkpoint_match 只负责解析参数，实际对局逻辑
    都放在这里，方便定时评测和其他脚本复用。

使用方法：
    from evaluations.core.match import run_checkpoint_match
    result = run_checkpoint_match("a.pt", "b.pt", num_games=200000)
    result = run_multi_checkpoint_match(["a.pt", "b.pt", "c.pt", "d.pt"], num_games=200000)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ppo.model import TileCountTransformerActorCritic, make_toy_ppo_config
from riichi_parallel_env import AGENTS


@dataclass(frozen=True)
class MatchResult:
    games: int
    model_a_score: int
    model_b_score: int
    draws: int
    seat_scores: tuple[int, ...]
    checkpoint_scores: tuple[int, ...] = ()

    @property
    def winner_margin(self) -> int:
        if self.checkpoint_scores:
            ordered = sorted(self.checkpoint_scores, reverse=True)
            return ordered[0] - ordered[1] if len(ordered) > 1 else 0
        return abs(self.model_a_score - self.model_b_score)

    @property
    def winner(self) -> str:
        if self.checkpoint_scores:
            best_score = max(self.checkpoint_scores)
            winners = [
                seat for seat, score in enumerate(self.checkpoint_scores) if score == best_score
            ]
            if len(winners) == 1:
                return f"seat {winners[0]}"
            return "tie"
        if self.model_a_score > self.model_b_score:
            return "model A"
        if self.model_b_score > self.model_a_score:
            return "model B"
        return "tie"


def load_agent(checkpoint_path: Path, device: torch.device) -> TileCountTransformerActorCritic:
    # torch.load uses pickle internally; only load checkpoints you trust.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_size = checkpoint.get("args", {}).get("model_size", "medium")
    agent = TileCountTransformerActorCritic(make_toy_ppo_config(model_size)).to(device)
    model_state_dict = checkpoint["model_state_dict"]
    if next(iter(model_state_dict), "").startswith("module."):
        model_state_dict = {
            key.removeprefix("module."): value for key, value in model_state_dict.items()
        }
    agent.load_state_dict(model_state_dict)
    agent.eval()
    return agent


def make_action_masks(observations: np.ndarray) -> np.ndarray:
    return observations > 0


@torch.no_grad()
def choose_actions(
    observations: np.ndarray,
    action_masks: np.ndarray,
    agents_by_seat: tuple[TileCountTransformerActorCritic, ...],
    device: torch.device,
) -> np.ndarray:
    num_envs = observations.shape[0]
    actions = np.zeros((num_envs, len(AGENTS)), dtype=np.uint8)

    for seat, agent in enumerate(agents_by_seat):
        obs = torch.as_tensor(
            observations[:, seat, :].reshape((-1, 34)),
            dtype=torch.float32,
            device=device,
        )
        masks = torch.as_tensor(
            action_masks[:, seat, :].reshape((-1, 34)),
            dtype=torch.bool,
            device=device,
        )
        logits = agent(obs, masks)["policy_logits"]
        actions[:, seat] = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.uint8)

    return actions


@torch.no_grad()
def choose_two_model_actions(
    observations: np.ndarray,
    action_masks: np.ndarray,
    agent_a: TileCountTransformerActorCritic,
    agent_b: TileCountTransformerActorCritic,
    seats_a: tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    seats_b = tuple(seat for seat in range(len(AGENTS)) if seat not in seats_a)
    agents_by_seat: list[TileCountTransformerActorCritic] = [agent_b for _seat in AGENTS]
    for seat in seats_a:
        agents_by_seat[seat] = agent_a
    for seat in seats_b:
        agents_by_seat[seat] = agent_b
    return choose_actions(observations, action_masks, tuple(agents_by_seat), device)


def parse_seats(text: str | tuple[int, int]) -> tuple[int, int]:
    if isinstance(text, tuple):
        seats = text
    else:
        seats = tuple(int(part) for part in text.split(","))
    if (
        len(seats) != 2
        or len(set(seats)) != 2
        or any(seat < 0 or seat >= len(AGENTS) for seat in seats)
    ):
        raise ValueError("--seats-a must contain two distinct seat indexes from 0 to 3")
    return seats


def run_checkpoint_match(
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    num_games: int,
    num_envs: int = 512,
    seed: int = 1,
    seats_a: tuple[int, int] = (0, 2),
    use_cuda: bool = True,
    progress_interval: int = 5_000,
    log: Callable[[str], None] = print,
) -> MatchResult:
    if num_games <= 0:
        raise ValueError("num_games must be greater than 0")
    if num_envs <= 0:
        raise ValueError("num_envs must be greater than 0")

    checkpoint_a = Path(checkpoint_a)
    checkpoint_b = Path(checkpoint_b)
    seats_a = parse_seats(seats_a)
    seats_b = tuple(seat for seat in range(len(AGENTS)) if seat not in seats_a)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    agent_a = load_agent(checkpoint_a, device)
    agent_b = load_agent(checkpoint_b, device)

    import riichi

    raw_env = riichi.VecEnv(num_envs, seed)
    observations = np.asarray(raw_env.reset(), dtype=np.uint8).reshape(num_envs, len(AGENTS), 34)
    action_masks = make_action_masks(observations)

    games = 0
    draws = 0
    model_a_score = 0
    model_b_score = 0
    seat_scores = np.zeros(len(AGENTS), dtype=np.int64)

    log(f"model A: {checkpoint_a}")
    log(f"model B: {checkpoint_b}")
    log(f"model A seats: {seats_a}; model B seats: {seats_b}")
    log(f"running {num_games} completed games with {num_envs} parallel envs on {device}")

    while games < num_games:
        actions = choose_two_model_actions(
            observations,
            action_masks,
            agent_a,
            agent_b,
            seats_a,
            device,
        )
        raw_obs, _raw_rewards, raw_dones, raw_winners = raw_env.step_with_winners(actions)
        observations = np.asarray(raw_obs, dtype=np.uint8).reshape(num_envs, len(AGENTS), 34)
        action_masks = make_action_masks(observations)
        dones = np.asarray(raw_dones, dtype=np.bool_).reshape(num_envs)
        winners = np.asarray(raw_winners, dtype=np.bool_).reshape(num_envs, len(AGENTS))

        done_indexes = np.nonzero(dones)[0]
        if done_indexes.size == 0:
            continue

        remaining = num_games - games
        if done_indexes.size > remaining:
            done_indexes = done_indexes[:remaining]
        completed_winners = winners[done_indexes]
        seat_scores += completed_winners.sum(axis=0, dtype=np.int64)
        model_a_score += int(completed_winners[:, seats_a].sum())
        model_b_score += int(completed_winners[:, seats_b].sum())
        draws += int(np.count_nonzero(completed_winners.sum(axis=1) == 0))
        games += int(done_indexes.size)

        if progress_interval > 0 and games % progress_interval < done_indexes.size:
            log(
                f"games={games}/{num_games} "
                f"A={model_a_score} B={model_b_score} draws={draws}"
            )

    result = MatchResult(
        games=games,
        model_a_score=model_a_score,
        model_b_score=model_b_score,
        draws=draws,
        seat_scores=tuple(int(value) for value in seat_scores.tolist()),
        checkpoint_scores=(model_a_score, model_b_score),
    )
    log("")
    log("final result")
    log(f"games: {result.games}")
    log(f"model A score: {result.model_a_score}")
    log(f"model B score: {result.model_b_score}")
    log(f"winner margin: {result.winner_margin}")
    log(f"draws: {result.draws}")
    log(f"seat scores: {list(result.seat_scores)}")
    log(f"winner: {result.winner}")
    return result


def run_multi_checkpoint_match(
    checkpoints: list[str | Path] | tuple[str | Path, ...],
    num_games: int,
    num_envs: int = 512,
    seed: int = 1,
    use_cuda: bool = True,
    progress_interval: int = 5_000,
    log: Callable[[str], None] = print,
) -> MatchResult:
    if len(checkpoints) != len(AGENTS):
        raise ValueError(f"expected {len(AGENTS)} checkpoints, got {len(checkpoints)}")
    if num_games <= 0:
        raise ValueError("num_games must be greater than 0")
    if num_envs <= 0:
        raise ValueError("num_envs must be greater than 0")

    checkpoint_paths = tuple(Path(checkpoint) for checkpoint in checkpoints)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    agents_by_seat = tuple(load_agent(checkpoint, device) for checkpoint in checkpoint_paths)

    import riichi

    raw_env = riichi.VecEnv(num_envs, seed)
    observations = np.asarray(raw_env.reset(), dtype=np.uint8).reshape(num_envs, len(AGENTS), 34)
    action_masks = make_action_masks(observations)

    games = 0
    draws = 0
    seat_scores = np.zeros(len(AGENTS), dtype=np.int64)

    for seat, checkpoint in enumerate(checkpoint_paths):
        log(f"seat {seat} checkpoint: {checkpoint}")
    log(f"running {num_games} completed games with {num_envs} parallel envs on {device}")

    while games < num_games:
        actions = choose_actions(observations, action_masks, agents_by_seat, device)
        raw_obs, _raw_rewards, raw_dones, raw_winners = raw_env.step_with_winners(actions)
        observations = np.asarray(raw_obs, dtype=np.uint8).reshape(num_envs, len(AGENTS), 34)
        action_masks = make_action_masks(observations)
        dones = np.asarray(raw_dones, dtype=np.bool_).reshape(num_envs)
        winners = np.asarray(raw_winners, dtype=np.bool_).reshape(num_envs, len(AGENTS))

        done_indexes = np.nonzero(dones)[0]
        if done_indexes.size == 0:
            continue

        remaining = num_games - games
        if done_indexes.size > remaining:
            done_indexes = done_indexes[:remaining]
        completed_winners = winners[done_indexes]
        seat_scores += completed_winners.sum(axis=0, dtype=np.int64)
        draws += int(np.count_nonzero(completed_winners.sum(axis=1) == 0))
        games += int(done_indexes.size)

        if progress_interval > 0 and games % progress_interval < done_indexes.size:
            scores_text = " ".join(
                f"seat{seat}={int(score)}" for seat, score in enumerate(seat_scores.tolist())
            )
            log(f"games={games}/{num_games} {scores_text} draws={draws}")

    checkpoint_scores = tuple(int(value) for value in seat_scores.tolist())
    result = MatchResult(
        games=games,
        model_a_score=checkpoint_scores[0],
        model_b_score=checkpoint_scores[1],
        draws=draws,
        seat_scores=checkpoint_scores,
        checkpoint_scores=checkpoint_scores,
    )
    log("")
    log("final result")
    log(f"games: {result.games}")
    for seat, score in enumerate(result.seat_scores):
        log(f"seat {seat} score: {score}")
    log(f"winner margin: {result.winner_margin}")
    log(f"draws: {result.draws}")
    log(f"seat scores: {list(result.seat_scores)}")
    log(f"winner: {result.winner}")
    return result
