"""V18 Action Query 可区分性永久化测试（action_id 进入嵌入后不同动作必须可区分）。"""

from __future__ import annotations

import json
from pathlib import Path

from riichi_ppo_v1.model.action_groups import action_id
from riichi_ppo_v1.model.current_state import encode_batch

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "RiichiEnv/tests/data/126_204_0_mjai.jsonl"


def _records(limit: int = 3) -> list[str]:
    import riichienv  # noqa: F401
    lines = [line for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    records, current = [], None
    for line in lines:
        if '"start_kyoku"' in line:
            current = [line]
        elif current is not None:
            current.append(line)
            if '"end_kyoku"' in line:
                records.append("\n".join(current) + "\n" + json.dumps({"type": "end_game"}) + "\n")
                current = None
                if len(records) >= limit:
                    break
    return records


def test_different_action_ids_produce_different_od_features() -> None:
    from riichi_ppo_v1.model.encoding_protocol import KIND_ACTION_DEFENSE_QUERY, KIND_ACTION_OFFENSE_QUERY
    from riichienv import MjaiReplay

    collisions: list[str] = []
    for ridx, record in enumerate(_records()):
        replay = MjaiReplay.from_jsonl_string(record, rule="tenhou")
        kyoku = list(replay.take_kyokus())[0]
        for seat in range(4):
            for step, (obs, _act) in enumerate(kyoku.steps(seat=seat, skip_single_action=False)):
                by_id = {}
                for action in obs.legal_actions():
                    aid = action_id(action, obs)
                    if aid is not None:
                        by_id.setdefault(aid, action)
                if len(by_id) < 2:
                    continue
                enc = encode_batch([(obs, [(a, aid) for aid, a in sorted(by_id.items())])])
                rows = enc.actor_factors[0][: int(enc.actor_lengths[0])]
                act_rows = [r for r in rows if int(r[1]) in (11, 12)]
                ids = sorted(by_id.keys())
                off = [tuple(int(v) for v in act_rows[2 * i][2:17]) for i in range(len(ids))]
                deff = [tuple(int(v) for v in act_rows[2 * i + 1][2:17]) for i in range(len(ids))]
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        if off[i] == off[j] and deff[i] == deff[j]:
                            collisions.append(
                                f"rec{ridx} seat{seat} step{step} ids=({ids[i]},{ids[j]})"
                            )
    assert not collisions, f"found {len(collisions)} action O/D collisions: {collisions[:5]}"
