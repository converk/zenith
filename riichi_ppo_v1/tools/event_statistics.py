"""按 step 作用域的 MJAI 事件身份与去重原语(供指标统计复用)。"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable


def canonical_step_events(
    per_seat_events: Iterable[Iterable[dict]],
    *,
    environment_id: int,
    hanchan_id: int,
    kyoku_id: int,
    step: int,
) -> list[tuple[tuple[int, int, int, int, int], dict]]:
    """Deduplicate observer copies without collapsing repeated game events.

    Each observer normally receives the same public event. We retain the
    maximum multiplicity seen by any one observer and assign a step-local
    ordinal. Thus identical reaches/hora in later steps or kyokus remain
    distinct while four observation-seat copies count once.
    """
    rows = [list(events) for events in per_seat_events]
    counts_by_seat: list[Counter[str]] = []
    values: dict[str, dict] = {}
    order: list[str] = []
    for events in rows:
        counts: Counter[str] = Counter()
        for event in events:
            if not isinstance(event, dict):
                continue
            normalized = json.dumps(event, sort_keys=True, separators=(",", ":"))
            counts[normalized] += 1
            if normalized not in values:
                order.append(normalized)
            values[normalized] = event
        counts_by_seat.append(counts)
    result: list[tuple[tuple[int, int, int, int, int], dict]] = []
    ordinal = 0
    for normalized in order:
        copies = max((counts[normalized] for counts in counts_by_seat), default=0)
        for _ in range(copies):
            key = (int(environment_id), int(hanchan_id), int(kyoku_id), int(step), ordinal)
            result.append((key, values[normalized]))
            ordinal += 1
    return result
