"""Filesystem-backed, bounded historical checkpoint population."""

from __future__ import annotations

from pathlib import Path
import random
import re


_ITERATION = re.compile(r"^iteration_(\d+)\.pt$")


def compatible_history(checkpoint_dir: str | Path, *, max_entries: int = 48) -> tuple[str, ...]:
    """Select latest, uniform and stage-anchor snapshots without mutating disk."""
    rows = []
    for path in Path(checkpoint_dir).glob("iteration_*.pt"):
        match = _ITERATION.match(path.name)
        if match:
            rows.append((int(match.group(1)), str(path)))
    rows.sort()
    if len(rows) <= max_entries:
        return tuple(path for _iteration, path in rows)
    latest = rows[-24:]
    remaining = rows[:-24]
    uniform_indices = {round(index * (len(remaining) - 1) / 15) for index in range(16)}
    uniform = [remaining[index] for index in sorted(uniform_indices)]
    anchors = []
    for fraction in (.15, .35, .60, .80):
        target = rows[-1][0] * fraction
        anchors.append(min(rows, key=lambda item: abs(item[0] - target)))
    selected = {path for _iteration, path in latest + uniform + anchors}
    return tuple(path for _iteration, path in rows if path in selected)[-max_entries:]


def rollout_cohort(history: tuple[str, ...], *, seed: int, update: int, size: int = 2) -> tuple[str, ...]:
    """Choose one deterministic rollout-wide cohort, without replacement.

    Restricting a rollout to two historical policies lets every inference actor
    prefetch and pin exactly the models its six workers will request.  The
    deterministic update-specific draw still rotates through the retained pool
    across training rather than biasing toward only the latest snapshots.
    """
    if len(history) <= int(size):
        return tuple(history)
    rng = random.Random(int(seed) * 1_000_003 + int(update))
    return tuple(sorted(rng.sample(list(history), int(size))))
