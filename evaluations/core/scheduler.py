"""定时 checkpoint 对局调度核心逻辑。

功能：
    从配置的 checkpoint run 目录中发现最新 checkpoint，按固定 step 间隔安排 interval、
    top6_random、ranking_random 等四模型对局，并在每次对局后刷新 Elo 排名。这里不解析
    命令行，方便后续复用或单元测试。

使用方法：
    from evaluations.core.scheduler import SchedulerConfig, maybe_schedule_matches
    changed = maybe_schedule_matches(config, state)
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from evaluations import config
from evaluations.core.elo import (
    DEFAULT_RUN_DIRS,
    checkpoint_step,
    parse_run_dirs,
    ranked_checkpoints,
    refresh_elo_ranking,
)


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    step: int


@dataclass(frozen=True)
class MatchTarget:
    kind: str
    opponents: tuple[CheckpointInfo, CheckpointInfo, CheckpointInfo]


@dataclass
class RunState:
    last_evaluated_step: int
    completed_matches: list[str]


@dataclass(frozen=True)
class SchedulerConfig:
    run_dirs: tuple[Path, ...] = DEFAULT_RUN_DIRS
    step_interval: int = config.STEP_INTERVAL
    random_seed: int = config.RANDOM_SEED
    initial_elo: float = config.INITIAL_ELO
    k_factor: float = config.K_FACTOR
    k_base_score: int = config.K_BASE_SCORE
    k_scale_cap: float = config.K_SCALE_CAP
    num_games: int = config.NUM_GAMES
    num_envs: int = config.NUM_ENVS
    cuda_device: str | None = config.CUDA_DEVICE
    progress_interval: int = config.PROGRESS_INTERVAL
    ranking_random_min_matches: int = config.RANKING_RANDOM_MIN_MATCHES
    ranking_random_max_matches: int = config.RANKING_RANDOM_MAX_MATCHES
    output_dir: Path = config.MATCH_LOGS_DIR


def list_checkpoints(run_dir: Path) -> list[CheckpointInfo]:
    checkpoints = [
        CheckpointInfo(path=path, step=checkpoint_step(path))
        for path in run_dir.glob("checkpoint_*.pt")
    ]
    return sorted(
        [checkpoint for checkpoint in checkpoints if checkpoint.step >= 0],
        key=lambda checkpoint: checkpoint.step,
    )


def latest_checkpoint_in_run_dir(run_dir: Path) -> CheckpointInfo | None:
    checkpoints = list_checkpoints(run_dir)
    if not checkpoints:
        return None
    return checkpoints[-1]


def recent_checkpoints_before(
    run_dir: Path,
    latest: CheckpointInfo,
    count: int,
) -> list[CheckpointInfo]:
    candidates = [
        checkpoint for checkpoint in list_checkpoints(run_dir) if checkpoint.step < latest.step
    ]
    return candidates[-count:]


def ranking_random_match_count(
    candidate_count: int,
    min_matches: int,
    max_matches: int,
) -> int:
    if candidate_count <= 0:
        return 0
    if candidate_count < min_matches:
        return candidate_count
    return min(max_matches, max(min_matches, math.ceil(math.sqrt(candidate_count))))


def sample_ranking_checkpoints(
    ranked_paths: list[Path],
    count: int,
    excluded: set[Path],
    seed: int,
) -> list[CheckpointInfo]:
    candidates = [path for path in ranked_paths if path not in excluded]
    if not candidates:
        return []
    rng = random.Random(seed)
    sample_size = min(count, len(candidates))
    return [
        CheckpointInfo(path=path, step=checkpoint_step(path))
        for path in rng.sample(candidates, sample_size)
    ]


def sample_checkpoint_infos(
    checkpoints: list[CheckpointInfo],
    count: int,
    excluded: set[Path],
    seed: int,
) -> tuple[CheckpointInfo, ...]:
    candidates = [checkpoint for checkpoint in checkpoints if checkpoint.path not in excluded]
    if len(candidates) < count:
        return ()
    rng = random.Random(seed)
    return tuple(rng.sample(candidates, count))


def match_key(kind: str, newer: CheckpointInfo, opponents: tuple[CheckpointInfo, ...]) -> str:
    opponent_text = "|".join(str(opponent.path) for opponent in opponents)
    return f"{kind}|{newer.path}|{opponent_text}"


def load_state(state_path: Path) -> dict[str, RunState]:
    if not state_path.exists():
        return {}
    raw = json.loads(state_path.read_text())
    return {
        run_name: RunState(
            last_evaluated_step=int(values["last_evaluated_step"]),
            completed_matches=list(values.get("completed_matches", [])),
        )
        for run_name, values in raw.items()
    }


def save_state(state_path: Path, state: dict[str, RunState]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {run_name: asdict(values) for run_name, values in sorted(state.items())}
    state_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")


def run_match(
    newer: CheckpointInfo,
    opponents: tuple[CheckpointInfo, CheckpointInfo, CheckpointInfo],
    match_kind: str,
    output_dir: Path,
    num_games: int,
    num_envs: int,
    cuda_device: str | None,
    progress_interval: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = newer.path.parent.name
    opponent_text = "_".join(str(opponent.step) for opponent in opponents)
    output_path = output_dir / (
        f"{run_name}__{match_kind}__new_{newer.step}__opps_{opponent_text}__{timestamp}.log"
    )
    command = [
        sys.executable,
        "-m",
        "evaluations.checkpoint_match",
        "--checkpoints",
        str(newer.path),
        *(str(opponent.path) for opponent in opponents),
        "--num-games",
        str(num_games),
        "--num-envs",
        str(num_envs),
        "--progress-interval",
        str(progress_interval),
    ]
    if cuda_device is not None:
        command = ["env", f"CUDA_VISIBLE_DEVICES={cuda_device}", *command]

    with output_path.open("w") as output:
        output.write(f"match type: {match_kind}\n")
        output.write(f"new checkpoint: {newer.path}\n")
        for index, opponent in enumerate(opponents, start=1):
            output.write(f"opponent {index} checkpoint: {opponent.path}\n")
        output.write(f"command: {' '.join(command)}\n\n")
        output.flush()
        subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)

    return output_path


def build_match_targets(
    latest: CheckpointInfo,
    run_dirs: tuple[Path, ...],
    output_dir: Path,
    random_seed: int,
    initial_elo: float,
    k_factor: float,
    k_base_score: int,
    k_scale_cap: float,
    ranking_random_min_matches: int,
    ranking_random_max_matches: int,
) -> list[MatchTarget]:
    targets: list[MatchTarget] = []
    used_target_keys: set[str] = set()

    def add_target(kind: str, opponents: tuple[CheckpointInfo, ...]) -> None:
        if len(opponents) != 3:
            return
        if len({opponent.path for opponent in opponents}) != 3:
            return
        key = match_key(kind, latest, opponents)
        if key in used_target_keys:
            return
        used_target_keys.add(key)
        targets.append(MatchTarget(kind=kind, opponents=opponents))

    recent = recent_checkpoints_before(latest.path.parent, latest, count=15)
    add_target(
        "interval",
        sample_checkpoint_infos(
            checkpoints=recent,
            count=3,
            excluded={latest.path},
            seed=random_seed + latest.step + 11,
        ),
    )

    ranking = ranked_checkpoints(
        logs_dir=output_dir,
        run_dirs=run_dirs,
        initial_elo=initial_elo,
        k_factor=k_factor,
        k_base_score=k_base_score,
        k_scale_cap=k_scale_cap,
        exclude={latest.path},
    )
    ranked_paths = [Path(record.checkpoint) for record in ranking]
    add_target(
        "top6_random",
        tuple(sample_ranking_checkpoints(
            ranked_paths=ranked_paths[:6],
            count=3,
            excluded={latest.path},
            seed=random_seed + latest.step + 17,
        )),
    )

    ranking_candidates = [
        CheckpointInfo(path=path, step=checkpoint_step(path))
        for path in ranked_paths
        if path != latest.path
    ]
    ranking_random_count = ranking_random_match_count(
        candidate_count=len(ranking_candidates),
        min_matches=ranking_random_min_matches,
        max_matches=ranking_random_max_matches,
    )
    for index in range(1, ranking_random_count + 1):
        add_target(
            f"ranking_random_{index}",
            sample_checkpoint_infos(
                checkpoints=ranking_candidates,
                count=3,
                excluded={latest.path},
                seed=random_seed + latest.step + 31 + index,
            ),
        )

    return targets

def maybe_schedule_matches(config: SchedulerConfig, state: dict[str, RunState]) -> bool:
    changed = False
    for run_dir in config.run_dirs:
        latest = latest_checkpoint_in_run_dir(run_dir)
        if latest is None:
            print(f"[skip] no checkpoint_*.pt found in {run_dir}", flush=True)
            continue

        run_name = latest.path.parent.name
        run_state = state.get(run_name)
        if run_state is None:
            state[run_name] = RunState(last_evaluated_step=latest.step, completed_matches=[])
            print(
                f"[init] {run_name}: latest_step={latest.step}; "
                "future matches start from this step",
                flush=True,
            )
            changed = True
            continue

        if latest.step - run_state.last_evaluated_step < config.step_interval:
            continue

        targets = build_match_targets(
            latest=latest,
            run_dirs=config.run_dirs,
            output_dir=config.output_dir,
            random_seed=config.random_seed,
            initial_elo=config.initial_elo,
            k_factor=config.k_factor,
            k_base_score=config.k_base_score,
            k_scale_cap=config.k_scale_cap,
            ranking_random_min_matches=config.ranking_random_min_matches,
            ranking_random_max_matches=config.ranking_random_max_matches,
        )
        if not targets:
            print(
                f"[wait] {run_name}: no match targets found for step {latest.step}",
                flush=True,
            )
            continue

        for target in targets:
            key = match_key(target.kind, latest, target.opponents)
            if key in run_state.completed_matches:
                continue
            opponents_text = ", ".join(
                f"{opponent.path.parent.name}:{opponent.step}" for opponent in target.opponents
            )
            print(
                f"[match:{target.kind}] {run_name}: "
                f"new={latest.step} opponents=[{opponents_text}]",
                flush=True,
            )
            output_path = run_match(
                newer=latest,
                opponents=target.opponents,
                match_kind=target.kind,
                output_dir=config.output_dir,
                num_games=config.num_games,
                num_envs=config.num_envs,
                cuda_device=config.cuda_device,
                progress_interval=config.progress_interval,
            )
            print(f"[done] wrote {output_path}", flush=True)
            run_state.completed_matches.append(key)
            refresh_elo_ranking(
                logs_dir=config.output_dir,
                run_dirs=config.run_dirs,
                initial_elo=config.initial_elo,
                k_factor=config.k_factor,
                k_base_score=config.k_base_score,
                k_scale_cap=config.k_scale_cap,
            )

        run_state.last_evaluated_step = latest.step
        changed = True

    return changed


def config_from_args(args) -> SchedulerConfig:
    run_dirs = parse_run_dirs(args.run_dirs)
    return SchedulerConfig(
        run_dirs=run_dirs,
        step_interval=args.step_interval,
        random_seed=args.random_seed,
        initial_elo=args.initial_elo,
        k_factor=args.k_factor,
        k_base_score=args.k_base_score,
        k_scale_cap=args.k_scale_cap,
        num_games=args.num_games,
        num_envs=args.num_envs,
        cuda_device=args.cuda_device,
        progress_interval=args.progress_interval,
        ranking_random_min_matches=args.ranking_random_min_matches,
        ranking_random_max_matches=args.ranking_random_max_matches,
        output_dir=Path(args.output_dir),
    )
