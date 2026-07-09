"""Elo 排名核心逻辑。

功能：
    扫描对局日志，把每个 checkpoint 当成独立选手，按 A/B 胡牌得分占比
    更新 Elo，并输出 JSON/CSV 排名。这里不解析命令行，方便定时评测脚本
    和手动排名脚本复用同一套逻辑。

使用方法：
    from evaluations.core.elo import refresh_elo_ranking, best_checkpoint
    records = refresh_elo_ranking()
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from evaluations import config


DEFAULT_RUN_DIRS = config.RUN_DIRS

MODEL_A_RE = re.compile(r"^model A:\s+(.+)$", re.MULTILINE)
MODEL_B_RE = re.compile(r"^model B:\s+(.+)$", re.MULTILINE)
SCORE_A_RE = re.compile(r"^model A score:\s+(\d+)$", re.MULTILINE)
SCORE_B_RE = re.compile(r"^model B score:\s+(\d+)$", re.MULTILINE)
SEAT_CHECKPOINT_RE = re.compile(r"^seat (\d+) checkpoint:\s+(.+)$", re.MULTILINE)
SEAT_SCORE_RE = re.compile(r"^seat (\d+) score:\s+(\d+)$", re.MULTILINE)
MATCH_RESULT_RE = re.compile(r"^MATCH_RESULT\s+({.*})$", re.MULTILINE)
TIMESTAMP_RE = re.compile(r"__(\d{8}_\d{6})\.log$")


@dataclass
class EloRecord:
    checkpoint: str
    elo: float
    matches: int = 0
    score_for: int = 0
    score_against: int = 0

    @property
    def step(self) -> int:
        return checkpoint_step(Path(self.checkpoint))

    @property
    def run_name(self) -> str:
        return Path(self.checkpoint).parent.name

    @property
    def score_rate(self) -> float:
        total_score = self.score_for + self.score_against
        if total_score <= 0:
            return 0.0
        return self.score_for / total_score


@dataclass(frozen=True)
class MatchRecord:
    log_path: Path
    checkpoint_scores: tuple[tuple[str, int], ...]
    timestamp: float

    @property
    def checkpoint_a(self) -> str:
        return self.checkpoint_scores[0][0]

    @property
    def checkpoint_b(self) -> str:
        return self.checkpoint_scores[1][0]

    @property
    def score_a(self) -> int:
        return self.checkpoint_scores[0][1]

    @property
    def score_b(self) -> int:
        return self.checkpoint_scores[1][1]


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def parse_run_dirs(values: list[str] | tuple[str, ...] | None) -> tuple[Path, ...]:
    if not values:
        return DEFAULT_RUN_DIRS
    return tuple(Path(value) for value in values)


def checkpoint_allowed(checkpoint: str, run_dirs: tuple[Path, ...]) -> bool:
    path = Path(checkpoint)
    return any(path.parent == run_dir for run_dir in run_dirs)


def log_timestamp(path: Path) -> float:
    match = TIMESTAMP_RE.search(path.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").timestamp()
    return path.stat().st_mtime


def parse_batch_match_logs(path: Path, text: str, run_dirs: tuple[Path, ...]) -> list[MatchRecord]:
    records: list[MatchRecord] = []
    for match in MATCH_RESULT_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        checkpoints = payload.get("checkpoints")
        scores = payload.get("scores")
        if not isinstance(checkpoints, list) or not isinstance(scores, list):
            continue
        if len(checkpoints) != len(scores) or len(checkpoints) < 2:
            continue
        checkpoint_scores = tuple(
            (str(checkpoint), int(score)) for checkpoint, score in zip(checkpoints, scores)
        )
        if not all(checkpoint_allowed(checkpoint, run_dirs) for checkpoint, _score in checkpoint_scores):
            continue
        records.append(
            MatchRecord(
                log_path=path,
                checkpoint_scores=checkpoint_scores,
                timestamp=log_timestamp(path),
            )
        )
    return records


def parse_single_match_log(path: Path, text: str, run_dirs: tuple[Path, ...]) -> MatchRecord | None:
    seat_checkpoints = {
        int(match.group(1)): match.group(2).strip()
        for match in SEAT_CHECKPOINT_RE.finditer(text)
    }
    seat_scores = {
        int(match.group(1)): int(match.group(2))
        for match in SEAT_SCORE_RE.finditer(text)
    }
    seat_indexes = sorted(set(seat_checkpoints) & set(seat_scores))
    if len(seat_indexes) >= 2:
        checkpoint_scores = tuple(
            (seat_checkpoints[seat], seat_scores[seat]) for seat in seat_indexes
        )
        if all(checkpoint_allowed(checkpoint, run_dirs) for checkpoint, _score in checkpoint_scores):
            return MatchRecord(
                log_path=path,
                checkpoint_scores=checkpoint_scores,
                timestamp=log_timestamp(path),
            )

    model_a = MODEL_A_RE.search(text)
    model_b = MODEL_B_RE.search(text)
    score_a = SCORE_A_RE.search(text)
    score_b = SCORE_B_RE.search(text)
    if not (model_a and model_b and score_a and score_b):
        return None

    checkpoint_a = model_a.group(1).strip()
    checkpoint_b = model_b.group(1).strip()
    if not checkpoint_allowed(checkpoint_a, run_dirs):
        return None
    if not checkpoint_allowed(checkpoint_b, run_dirs):
        return None

    return MatchRecord(
        log_path=path,
        checkpoint_scores=(
            (checkpoint_a, int(score_a.group(1))),
            (checkpoint_b, int(score_b.group(1))),
        ),
        timestamp=log_timestamp(path),
    )


def parse_match_logs(path: Path, run_dirs: tuple[Path, ...]) -> list[MatchRecord]:
    text = path.read_text(errors="replace")
    return parse_batch_match_logs(path, text, run_dirs)


def load_match_logs(logs_dir: Path, run_dirs: tuple[Path, ...]) -> list[MatchRecord]:
    records = [
        record
        for path in logs_dir.glob("*.log")
        for record in parse_match_logs(path, run_dirs)
    ]
    return sorted(records, key=lambda record: (record.timestamp, record.log_path.name))


def build_elo_records(
    matches: list[MatchRecord],
    initial_elo: float,
    k_factor: float,
    k_base_score: int,
    k_scale_cap: float,
) -> dict[str, EloRecord]:
    records: dict[str, EloRecord] = {}

    def get_record(checkpoint: str) -> EloRecord:
        if checkpoint not in records:
            records[checkpoint] = EloRecord(checkpoint=checkpoint, elo=initial_elo)
        return records[checkpoint]

    for match in matches:
        total_score = sum(score for _checkpoint, score in match.checkpoint_scores)
        if total_score <= 0:
            continue

        match_records = [
            (get_record(checkpoint), score)
            for checkpoint, score in match.checkpoint_scores
        ]
        for record, score in match_records:
            record.matches += 1
            record.score_for += score
            record.score_against += total_score - score

        pair_scale = max(len(match_records) - 1, 1)
        for index, (record_a, score_a) in enumerate(match_records):
            for record_b, score_b in match_records[index + 1:]:
                pair_total = score_a + score_b
                if pair_total <= 0:
                    continue
                expected_a = 1.0 / (1.0 + 10.0 ** ((record_b.elo - record_a.elo) / 400.0))
                actual_a = score_a / pair_total
                score_scale = math.sqrt(pair_total / k_base_score)
                effective_k = k_factor * min(k_scale_cap, score_scale) / pair_scale
                delta = effective_k * (actual_a - expected_a)

                record_a.elo += delta
                record_b.elo -= delta

    return records


def sorted_records(records: dict[str, EloRecord]) -> list[EloRecord]:
    return sorted(records.values(), key=lambda record: (record.elo, record.step), reverse=True)


def write_outputs(
    records: dict[str, EloRecord],
    output_json: Path,
    output_csv: Path,
) -> None:
    ranking = sorted_records(records)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps([asdict(record) for record in ranking], indent=2, ensure_ascii=False) + "\n"
    )
    # CSV 字段含义见 evaluations/elo_ranking.py 顶部说明。
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "elo",
                "matches",
                "score_for",
                "score_against",
                "score_rate",
                "run_name",
                "step",
                "checkpoint",
            ],
        )
        writer.writeheader()
        for rank, record in enumerate(ranking, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "elo": f"{record.elo:.2f}",
                    "matches": record.matches,
                    "score_for": record.score_for,
                    "score_against": record.score_against,
                    "score_rate": f"{record.score_rate:.4f}",
                    "run_name": record.run_name,
                    "step": record.step,
                    "checkpoint": record.checkpoint,
                }
            )


def refresh_elo_ranking(
    logs_dir: Path = config.MATCH_LOGS_DIR,
    run_dirs: tuple[Path, ...] = DEFAULT_RUN_DIRS,
    initial_elo: float = config.INITIAL_ELO,
    k_factor: float = config.K_FACTOR,
    k_base_score: int = config.K_BASE_SCORE,
    k_scale_cap: float = config.K_SCALE_CAP,
) -> dict[str, EloRecord]:
    matches = load_match_logs(logs_dir, run_dirs)
    records = build_elo_records(
        matches,
        initial_elo=initial_elo,
        k_factor=k_factor,
        k_base_score=k_base_score,
        k_scale_cap=k_scale_cap,
    )
    write_outputs(
        records,
        output_json=logs_dir / "elo_ranking.json",
        output_csv=logs_dir / "elo_ranking.csv",
    )
    return records


def best_checkpoint(
    logs_dir: Path,
    run_dirs: tuple[Path, ...],
    initial_elo: float,
    k_factor: float,
    k_base_score: int,
    k_scale_cap: float,
    exclude: set[Path] | None = None,
) -> EloRecord | None:
    records = refresh_elo_ranking(
        logs_dir=logs_dir,
        run_dirs=run_dirs,
        initial_elo=initial_elo,
        k_factor=k_factor,
        k_base_score=k_base_score,
        k_scale_cap=k_scale_cap,
    )
    excluded = {str(path) for path in (exclude or set())}
    for record in sorted_records(records):
        if record.checkpoint not in excluded:
            return record
    return None


def ranked_checkpoints(
    logs_dir: Path,
    run_dirs: tuple[Path, ...],
    initial_elo: float,
    k_factor: float,
    k_base_score: int,
    k_scale_cap: float,
    exclude: set[Path] | None = None,
) -> list[EloRecord]:
    records = refresh_elo_ranking(
        logs_dir=logs_dir,
        run_dirs=run_dirs,
        initial_elo=initial_elo,
        k_factor=k_factor,
        k_base_score=k_base_score,
        k_scale_cap=k_scale_cap,
    )
    excluded = {str(path) for path in (exclude or set())}
    return [record for record in sorted_records(records) if record.checkpoint not in excluded]


def print_ranking(records: dict[str, EloRecord], limit: int) -> None:
    print("rank  elo      matches  score_for  score_against  score_rate  step        run")
    for rank, record in enumerate(sorted_records(records)[:limit], start=1):
        print(
            f"{rank:>4}  {record.elo:>7.2f}  {record.matches:>7}  "
            f"{record.score_for:>9}  {record.score_against:>13}  "
            f"{record.score_rate:>10.4f}  "
            f"{record.step:>10}  {record.run_name}"
        )
