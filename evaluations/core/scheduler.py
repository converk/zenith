"""定时 checkpoint 对局调度核心逻辑。

功能：
    从配置的 checkpoint run 目录中发现 checkpoint，轮流执行 batch 评测。
    每个 batch 内会多次重新随机抽取 4 个模型，每次小 match 只打少量局数。
    batch log 使用 MATCH_RESULT JSON 行记录每一次小 match，Elo 排名会读取
    这些结构化结果。

使用方法：
    from evaluations.core.scheduler import SchedulerConfig, maybe_schedule_matches
    changed = maybe_schedule_matches(config, state)
"""

from __future__ import annotations

import json
import os
import random
import time
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
from evaluations.core.match import MatchResult, run_multi_checkpoint_match


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    step: int


@dataclass
class RunState:
    last_evaluated_step: int
    completed_matches: list[str]


@dataclass(frozen=True)
class SchedulerConfig:
    run_dirs: tuple[Path, ...] = DEFAULT_RUN_DIRS
    random_seed: int = config.RANDOM_SEED
    initial_elo: float = config.INITIAL_ELO
    k_factor: float = config.K_FACTOR
    k_base_score: int = config.K_BASE_SCORE
    k_scale_cap: float = config.K_SCALE_CAP
    num_games: int = config.NUM_GAMES
    num_envs: int = config.NUM_ENVS
    cuda_device: str | None = config.CUDA_DEVICE
    progress_interval: int = config.PROGRESS_INTERVAL
    regular_self_matches: int = config.REGULAR_SELF_MATCHES
    regular_ranking_matches: int = config.REGULAR_RANKING_MATCHES
    new_self_matches: int = config.NEW_SELF_MATCHES
    new_ranking_matches: int = config.NEW_RANKING_MATCHES
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


def refresh_ranking_paths(config: SchedulerConfig) -> list[Path]:
    records = ranked_checkpoints(
        logs_dir=config.output_dir,
        run_dirs=config.run_dirs,
        initial_elo=config.initial_elo,
        k_factor=config.k_factor,
        k_base_score=config.k_base_score,
        k_scale_cap=config.k_scale_cap,
    )
    return [Path(record.checkpoint) for record in records]


def sample_paths(
    candidates: list[Path],
    count: int,
    rng: random.Random,
    required: Path | None = None,
) -> tuple[Path, ...]:
    excluded = {required} if required is not None else set()
    pool = [path for path in candidates if path not in excluded]
    needed = count - len(excluded)
    if needed < 0 or len(pool) < needed:
        return ()
    paths = list(excluded) + rng.sample(pool, needed)
    rng.shuffle(paths)
    return tuple(paths)


def result_payload(
    batch_kind: str,
    batch_index: int,
    match_index: int,
    checkpoints: tuple[Path, ...],
    result: MatchResult,
) -> dict:
    return {
        "batch_kind": batch_kind,
        "batch_index": batch_index,
        "match_index": match_index,
        "games": result.games,
        "draws": result.draws,
        "checkpoints": [str(path) for path in checkpoints],
        "scores": list(result.checkpoint_scores),
    }


def run_batch(
    run_dir: Path,
    batch_kind: str,
    match_count: int,
    games_per_match: int,
    num_envs: int,
    cuda_device: str | None,
    output_dir: Path,
    rng: random.Random,
    selector,
    batch_index: int,
) -> Path | None:
    if match_count <= 0:
        return None
    if cuda_device is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_device)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{run_dir.name}__{batch_kind}__batch_{batch_index}__{timestamp}.log"
    completed = 0

    with output_path.open("w") as output:
        output.write(f"batch kind: {batch_kind}\n")
        output.write(f"run dir: {run_dir}\n")
        output.write(f"match count: {match_count}\n")
        output.write(f"games per match: {games_per_match}\n\n")
        output.flush()

        for match_index in range(1, match_count + 1):
            checkpoints = selector(rng)
            if len(checkpoints) != 4:
                print(
                    f"[skip:{batch_kind}] not enough candidates for match {match_index}",
                    flush=True,
                )
                continue
            result = run_multi_checkpoint_match(
                checkpoints=checkpoints,
                num_games=games_per_match,
                num_envs=num_envs,
                seed=rng.randrange(1, 2**31),
                use_cuda=True,
                progress_interval=0,
                log=lambda _text: None,
            )
            payload = result_payload(
                batch_kind=batch_kind,
                batch_index=batch_index,
                match_index=match_index,
                checkpoints=checkpoints,
                result=result,
            )
            output.write("MATCH_RESULT " + json.dumps(payload, ensure_ascii=False) + "\n")
            output.flush()
            completed += 1

            if completed % 10 == 0 or completed == match_count:
                print(
                    f"[batch:{batch_kind}] {run_dir.name} "
                    f"{completed}/{match_count} matches complete",
                    flush=True,
                )

    if completed == 0:
        output_path.unlink(missing_ok=True)
        return None
    return output_path


def make_self_selector(run_dir: Path):
    candidates = [checkpoint.path for checkpoint in list_checkpoints(run_dir)]

    def selector(rng: random.Random) -> tuple[Path, ...]:
        return sample_paths(candidates, count=4, rng=rng)

    return selector


def make_ranking_selector(ranking_paths: list[Path]):
    candidates = list(ranking_paths)

    def selector(rng: random.Random) -> tuple[Path, ...]:
        return sample_paths(candidates, count=4, rng=rng)

    return selector


def make_new_self_selector(run_dir: Path, new_checkpoint: Path):
    candidates = [checkpoint.path for checkpoint in list_checkpoints(run_dir)]

    def selector(rng: random.Random) -> tuple[Path, ...]:
        return sample_paths(candidates, count=4, rng=rng, required=new_checkpoint)

    return selector


def make_new_ranking_selector(ranking_paths: list[Path], new_checkpoint: Path):
    candidates = list(dict.fromkeys([new_checkpoint, *ranking_paths]))

    def selector(rng: random.Random) -> tuple[Path, ...]:
        return sample_paths(candidates, count=4, rng=rng, required=new_checkpoint)

    return selector


def run_and_refresh(
    config: SchedulerConfig,
    run_dir: Path,
    batch_kind: str,
    match_count: int,
    selector,
    rng: random.Random,
    batch_index: int,
) -> Path | None:
    print(
        f"[batch:{batch_kind}] {run_dir.name} "
        f"matches={match_count} games_per_match={config.num_games}",
        flush=True,
    )
    output_path = run_batch(
        run_dir=run_dir,
        batch_kind=batch_kind,
        match_count=match_count,
        games_per_match=config.num_games,
        num_envs=config.num_envs,
        cuda_device=config.cuda_device,
        output_dir=config.output_dir,
        rng=rng,
        selector=selector,
        batch_index=batch_index,
    )
    if output_path is not None:
        print(f"[done] wrote {output_path}", flush=True)
        refresh_elo_ranking(
            logs_dir=config.output_dir,
            run_dirs=config.run_dirs,
            initial_elo=config.initial_elo,
            k_factor=config.k_factor,
            k_base_score=config.k_base_score,
            k_scale_cap=config.k_scale_cap,
        )
    return output_path


def maybe_schedule_matches(config: SchedulerConfig, state: dict[str, RunState]) -> bool:
    changed = False
    batch_index = int(time.time())

    for run_position, run_dir in enumerate(config.run_dirs):
        latest = latest_checkpoint_in_run_dir(run_dir)
        if latest is None:
            print(f"[skip] no checkpoint_*.pt found in {run_dir}", flush=True)
            continue

        rng = random.Random(config.random_seed + batch_index + run_position)
        run_name = run_dir.name
        run_state = state.get(run_name)
        if run_state is None:
            run_state = RunState(last_evaluated_step=latest.step, completed_matches=[])
            state[run_name] = run_state
            changed = True

        run_and_refresh(
            config=config,
            run_dir=run_dir,
            batch_kind="self_random",
            match_count=config.regular_self_matches,
            selector=make_self_selector(run_dir),
            rng=rng,
            batch_index=batch_index,
        )

        ranking_paths = refresh_ranking_paths(config)
        run_and_refresh(
            config=config,
            run_dir=run_dir,
            batch_kind="ranking_random",
            match_count=config.regular_ranking_matches,
            selector=make_ranking_selector(ranking_paths),
            rng=rng,
            batch_index=batch_index,
        )

        latest = latest_checkpoint_in_run_dir(run_dir)
        if latest is None:
            continue
        if latest.step <= run_state.last_evaluated_step:
            continue

        print(
            f"[new] {run_name}: {run_state.last_evaluated_step} -> {latest.step}",
            flush=True,
        )
        run_and_refresh(
            config=config,
            run_dir=run_dir,
            batch_kind=f"new_self_{latest.step}",
            match_count=config.new_self_matches,
            selector=make_new_self_selector(run_dir, latest.path),
            rng=rng,
            batch_index=batch_index,
        )

        ranking_paths = refresh_ranking_paths(config)
        run_and_refresh(
            config=config,
            run_dir=run_dir,
            batch_kind=f"new_ranking_{latest.step}",
            match_count=config.new_ranking_matches,
            selector=make_new_ranking_selector(ranking_paths, latest.path),
            rng=rng,
            batch_index=batch_index,
        )

        run_state.last_evaluated_step = latest.step
        changed = True

    return changed


def config_from_args(args) -> SchedulerConfig:
    run_dirs = parse_run_dirs(args.run_dirs)
    return SchedulerConfig(
        run_dirs=run_dirs,
        random_seed=args.random_seed,
        initial_elo=args.initial_elo,
        k_factor=args.k_factor,
        k_base_score=args.k_base_score,
        k_scale_cap=args.k_scale_cap,
        num_games=args.num_games,
        num_envs=args.num_envs,
        cuda_device=args.cuda_device,
        progress_interval=args.progress_interval,
        regular_self_matches=args.regular_self_matches,
        regular_ranking_matches=args.regular_ranking_matches,
        new_self_matches=args.new_self_matches,
        new_ranking_matches=args.new_ranking_matches,
        output_dir=Path(args.output_dir),
    )
