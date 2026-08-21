"""审计线上 validate 记录中的 V16 Actor 输入语义。"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_v16_token_decoder import (  # noqa: E402
    _canonical_template,
    decode_event_rows,
    decode_query_rows,
    decode_snapshot,
    decode_state_rows,
)

import riichi  # noqa: E402
from riichienv import Observation  # noqa: E402

from riichi_lab_bot.bridge import _MODEL_EVENT_TYPES  # noqa: E402
from riichi_lab_bot.observation import (  # noqa: E402
    ObservationView,
    missing_observation_fields,
    normalize_observation_base64,
)
from riichi_ppo_v1.model.action_query import (  # noqa: E402
    analyze_action_queries,
    encode_query_row,
)
from riichi_ppo_v1.model.bridge import (  # noqa: E402
    action_jsons_and_decision_flag,
    snapshot_json,
)
from riichi_ppo_v1.model.encoding_protocol import (  # noqa: E402
    QUERY_ROW_ANSWER_START,
)

NUM_PLAYERS = 4
RED_PHYSICAL = {"5mr": 16, "5pr": 52, "5sr": 88}
HONOR_PHYSICAL = {
    "E": 108,
    "S": 112,
    "W": 116,
    "N": 120,
    "P": 124,
    "F": 128,
    "C": 132,
}


class IndependentFieldTracker:
    """独立重建线上可能缺失、但 V16 Actor 输入需要的公开字段。"""

    def __init__(self, seat: int) -> None:
        self.seat = int(seat)
        self.reset()

    def reset(self) -> None:
        self.riichi_declared = [False] * NUM_PLAYERS
        self.riichi_accepted = [False] * NUM_PLAYERS
        self.riichi_declaration_indices = [None] * NUM_PLAYERS
        self.riichi_sutehais = [None] * NUM_PLAYERS
        self.last_tedashis = [None] * NUM_PLAYERS
        self.tsumogiri_flags: list[list[bool]] = [
            [] for _ in range(NUM_PLAYERS)
        ]
        self.discard_counts = [0] * NUM_PLAYERS
        self.tsumo_count = 0
        self.missed_agari_doujun = False
        self.missed_agari_riichi = False
        self.drawn_tile = None
        self.drawn_pai = None

    def apply_event(self, event: dict[str, Any]) -> None:
        kind = str(event.get("type", ""))
        if kind == "start_kyoku":
            self.reset()
            return
        actor_raw = event.get("actor")
        actor = int(actor_raw) if isinstance(actor_raw, int) else -1
        if not 0 <= actor < NUM_PLAYERS:
            return
        if kind == "tsumo":
            self.tsumo_count += 1
            if actor == self.seat:
                pai = event.get("pai")
                self.drawn_pai = pai if isinstance(pai, str) else None
                self.drawn_tile = mjai_to_physical_id(self.drawn_pai)
        elif kind == "reach":
            self.riichi_declared[actor] = True
            self.riichi_declaration_indices[actor] = self.discard_counts[actor]
        elif kind == "reach_accepted":
            self.riichi_accepted[actor] = True
        elif kind in {"chi", "pon", "daiminkan"} and actor == self.seat:
            self.missed_agari_doujun = False
            self.drawn_tile = None
            self.drawn_pai = None
        elif kind == "dahai":
            pai = event.get("pai")
            tsumogiri = bool(event.get("tsumogiri", False))
            self.tsumogiri_flags[actor].append(tsumogiri)
            physical = mjai_to_physical_id(pai if isinstance(pai, str) else None)
            if not tsumogiri and physical is not None:
                self.last_tedashis[actor] = physical
            if (
                self.riichi_declared[actor]
                and self.riichi_sutehais[actor] is None
                and physical is not None
            ):
                self.riichi_sutehais[actor] = physical
            self.discard_counts[actor] += 1
            if actor == self.seat:
                self.missed_agari_doujun = False
                self.drawn_tile = None
                self.drawn_pai = None

    def refine_drawn_tile(self, observation: object) -> None:
        if self.drawn_tile is None:
            return
        hand = getattr(observation, "hands", None)
        if hand is None:
            return
        candidates = [
            int(tile)
            for tile in hand[self.seat]
            if physical_matches_pai(int(tile), self.drawn_pai)
        ]
        if candidates:
            self.drawn_tile = candidates[-1]

    def apply_current_window(
        self,
        observation: object,
        *,
        last_type: str | None,
        actor: int | None,
        pai: str | None,
    ) -> None:
        if actor is None or actor == self.seat:
            return
        if last_type not in {"dahai", "kakan", "ankan"}:
            return
        if self.missed_agari_doujun or self.missed_agari_riichi:
            return
        if has_legal_hora(observation):
            return
        win_tile = mjai_to_physical_id(pai)
        if win_tile is None:
            return
        own_discards = getattr(observation, "discards", ([], [], [], []))
        if any(int(tile) // 4 == win_tile // 4 for tile in own_discards[self.seat]):
            return
        if has_win_shape_without_legal_ron(observation, self.seat, win_tile):
            self.missed_agari_doujun = True

    def record_response(
        self,
        observation: object,
        *,
        last_type: str | None,
        actor: int | None,
        legal_jsons: list[dict[str, Any]],
        response: dict[str, Any] | None,
    ) -> None:
        response_type = response.get("type") if isinstance(response, dict) else None
        if (
            actor is not None
            and actor != self.seat
            and last_type in {"dahai", "kakan", "ankan"}
            and response_type != "hora"
            and any(item.get("type") == "hora" for item in legal_jsons)
        ):
            self.missed_agari_doujun = True
            declared = getattr(observation, "riichi_declared", (False,) * NUM_PLAYERS)
            if bool(declared[self.seat]):
                self.missed_agari_riichi = True
        if response_type in {"chi", "pon", "daiminkan", "dahai"}:
            self.missed_agari_doujun = False

    def fields(self) -> dict[str, Any]:
        declared = [
            bool(self.riichi_declared[seat] and self.riichi_sutehais[seat] is not None)
            or bool(self.riichi_accepted[seat])
            for seat in range(NUM_PLAYERS)
        ]
        return {
            "riichi_declared": declared,
            "riichi_accepted": list(self.riichi_accepted),
            "riichi_declaration_indices": list(self.riichi_declaration_indices),
            "last_tedashis": list(self.last_tedashis),
            "tsumogiri_flags": [list(row) for row in self.tsumogiri_flags],
            "riichi_sutehais": list(self.riichi_sutehais),
            "missed_agari_doujun": self.missed_agari_doujun,
            "missed_agari_riichi": self.missed_agari_riichi,
            "tiles_left": max(70 - self.tsumo_count, 0),
            "waits": [],
            "is_tenpai": False,
            "drawn_tile": self.drawn_tile,
        }


def mjai_to_physical_id(pai: str | None) -> int | None:
    if not pai or pai == "?":
        return None
    if pai in RED_PHYSICAL:
        return RED_PHYSICAL[pai]
    if len(pai) == 2 and pai[1] in "mps":
        rank = int(pai[0])
        suit = "mps".index(pai[1])
        if 1 <= rank <= 9:
            return suit * 36 + (rank - 1) * 4
    if pai in HONOR_PHYSICAL:
        return HONOR_PHYSICAL[pai]
    return None


def physical_matches_pai(tile: int, pai: str | None) -> bool:
    reference = mjai_to_physical_id(pai)
    if reference is None:
        return False
    if pai in RED_PHYSICAL:
        return int(tile) == reference
    return int(tile) // 4 == reference // 4


def has_legal_hora(observation: object) -> bool:
    for action in observation.legal_actions():
        try:
            payload = json.loads(action.to_mjai())
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("type") == "hora":
            return True
    return False


def has_win_shape_without_legal_ron(
    observation: object, seat: int, win_tile: int,
) -> bool:
    try:
        from riichienv import Conditions, HandEvaluator

        declared = getattr(observation, "riichi_declared", (False,) * NUM_PLAYERS)
        conditions = Conditions(
            tsumo=False,
            riichi=bool(declared[seat]),
            player_wind=(seat - int(getattr(observation, "oya", 0))) % NUM_PLAYERS,
            round_wind=int(getattr(observation, "round_wind", 0)),
            houtei=int(getattr(observation, "tiles_left", 1)) == 0,
            riichi_sticks=int(getattr(observation, "riichi_sticks", 0)),
            honba=int(getattr(observation, "honba", 0)),
        )
        evaluator = HandEvaluator(
            sorted(int(tile) for tile in observation.hands[seat]),
            list(observation.melds[seat]),
        )
        result = evaluator.calc(
            int(win_tile),
            [int(tile) for tile in getattr(observation, "dora_indicators", ())],
            conditions,
        )
    except Exception:
        return False
    return bool(getattr(result, "has_win_shape", False)) and not bool(
        getattr(result, "is_win", False)
    )


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_without_tsumogiri(value: Any) -> str:
    """服务器 possible_actions 不承诺 dahai.tsumogiri,集合比较时忽略该位。"""
    if isinstance(value, dict):
        normalized = dict(value)
        if normalized.get("type") == "dahai":
            normalized.pop("tsumogiri", None)
        return canonical(normalized)
    return canonical(value)


def raw_observation_json(encoded: str) -> dict[str, Any]:
    value = json.loads(base64.b64decode(encoded))
    if not isinstance(value, dict):
        raise ValueError("observation_base64 must contain a JSON object")
    return value


def deserialize_view(record: dict[str, Any]) -> ObservationView:
    normalized = normalize_observation_base64(str(record["observation_base64"]))
    observation = Observation.deserialize_from_base64(normalized)
    return ObservationView(
        observation,
        missing_fields=missing_observation_fields(str(record["observation_base64"])),
    )


def accepted_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in record.get("new_events", [])
        if isinstance(event, dict) and event.get("type") in _MODEL_EVENT_TYPES
    ]


def parse_context(events: list[dict[str, Any]]) -> tuple[str | None, int | None, str | None]:
    context: tuple[str | None, int | None, str | None] = (None, None, None)
    for event in events:
        event_type = event.get("type")
        if event_type in {"tsumo", "dahai", "kakan", "ankan"}:
            actor = event.get("actor")
            pai = event.get("pai")
            context = (
                str(event_type),
                int(actor) if isinstance(actor, int) else None,
                pai if isinstance(pai, str) else None,
            )
    return context


def compare_array(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
    request_id: int,
    atol: float = 0.0,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"request={request_id} {label} shape 不一致: "
            f"actual={actual.shape} expected={expected.shape}"
        )
    if atol == 0.0:
        ok = np.array_equal(actual, expected)
    else:
        ok = np.allclose(actual, expected, rtol=0.0, atol=atol)
    if ok:
        return
    diff = np.argwhere(np.abs(actual.astype(float) - expected.astype(float)) > atol)
    first = diff[0].tolist() if diff.size else []
    raise AssertionError(
        f"request={request_id} {label} 不一致 first_diff={first} "
        f"actual={actual[tuple(first)] if first else None} "
        f"expected={expected[tuple(first)] if first else None}"
    )


def action_for_id(
    observation: object,
    record: dict[str, Any],
    action_id: int,
) -> object:
    _legal, decision_flag = action_jsons_and_decision_flag(observation)
    manager = riichi.MjaiKyokuStateMachineManager(1)
    legal_jsons = [
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        for value in record["legal_jsons"]
    ]
    manager.prepare_decisions(
        [int(record["seat"])],
        [legal_jsons],
        [snapshot_json(observation, decision_flag)],
    )
    decoded = json.loads(manager.decode_actions([int(record["seat"])], [int(action_id)])[0])
    decoded_key = canonical(decoded)
    representatives: dict[str, object] = {}
    for action in observation.legal_actions():
        representatives.setdefault(_canonical_template(action, observation), action)
    if decoded_key not in representatives:
        raise AssertionError(
            f"request={record['request_id']} action_id={action_id} decoded action 不在合法动作中"
        )
    return representatives[decoded_key]


def compare_possible_actions(record: dict[str, Any]) -> None:
    possible = {canonical_without_tsumogiri(value) for value in record.get("possible_actions", [])}
    legal = {canonical_without_tsumogiri(value) for value in record.get("legal_jsons", [])}
    if possible != legal:
        missing = sorted(legal - possible)[:3]
        extra = sorted(possible - legal)[:3]
        raise AssertionError(
            f"request={record['request_id']} server possible_actions 与本地合法动作不一致 "
            f"missing={missing} extra={extra}"
        )


def audit(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if record.get("event") == "v16_input_audit"]
    if not records:
        raise RuntimeError(f"{path} 没有 v16_input_audit 记录")

    seat = int(records[0]["seat"])
    tracker = IndependentFieldTracker(seat)
    event_buffer: list[dict[str, Any]] = []
    last_context: tuple[str | None, int | None, str | None] = (None, None, None)
    counters: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    rebuilt_mismatches: list[str] = []
    raw_field_mismatches: list[str] = []
    action_type_counts: Counter[str] = Counter()
    query_pairs = 0
    max_history = 0
    max_snapshot = 0

    for index, record in enumerate(records):
        request_id = int(record["request_id"])
        view = deserialize_view(record)
        raw_json = raw_observation_json(str(record["observation_base64"]))
        for field in record.get("missing_fields", []):
            missing_fields[str(field)] += 1

        events = accepted_events(record)
        if any(event.get("type") == "start_kyoku" for event in events):
            event_buffer = []
            last_context = (None, None, None)
        for event in events:
            if event.get("type") == "start_kyoku":
                event_buffer = []
            event_buffer.append(event)
            tracker.apply_event(event)
        tracker.refine_drawn_tile(view)
        new_context = parse_context(events)
        if new_context[0] is not None:
            last_context = new_context
        tracker.apply_current_window(
            view,
            last_type=last_context[0],
            actor=last_context[1],
            pai=last_context[2],
        )
        expected_fields = tracker.fields()
        rebuilt_fields = record.get("rebuilt_fields", {})
        for field, expected in expected_fields.items():
            if field in rebuilt_fields and rebuilt_fields[field] != expected:
                rebuilt_mismatches.append(
                    f"request={request_id} field={field} expected={expected} actual={rebuilt_fields[field]}"
                )
            if field in raw_json and raw_json[field] != expected:
                raw_field_mismatches.append(
                    f"request={request_id} field={field} expected={expected} raw={raw_json[field]}"
                )
        view.set_fields(expected_fields)

        compare_possible_actions(record)

        event_factors, event_numeric = decode_event_rows(event_buffer, seat)
        state_factors, state_numeric = decode_state_rows(
            view,
            live_wall=expected_fields["tiles_left"],
        )
        expected_history = np.stack(event_factors + state_factors, axis=0)
        expected_history_num = np.stack(event_numeric + state_numeric, axis=0)
        actual_history = np.asarray(record["history"]["factors"], dtype=np.uint8)
        actual_history_num = np.asarray(record["history"]["numeric"], dtype=np.float32)
        compare_array(
            actual_history,
            expected_history,
            label="history_factors",
            request_id=request_id,
        )
        compare_array(
            actual_history_num,
            expected_history_num,
            label="history_numeric",
            request_id=request_id,
            atol=1e-3,
        )
        max_history = max(max_history, int(actual_history.shape[0]))

        expected_kinds, expected_cat, expected_num = decode_snapshot(view)
        compare_array(
            np.asarray(record["snapshot"]["kinds"], dtype=np.uint8),
            expected_kinds,
            label="snapshot_kinds",
            request_id=request_id,
        )
        compare_array(
            np.asarray(record["snapshot"]["cat"], dtype=np.uint8),
            expected_cat,
            label="snapshot_cat",
            request_id=request_id,
        )
        compare_array(
            np.asarray(record["snapshot"]["num"], dtype=np.float32),
            expected_num,
            label="snapshot_num",
            request_id=request_id,
            atol=1e-5,
        )
        max_snapshot = max(max_snapshot, int(expected_kinds.shape[0]))

        actual_ids = [int(value) for value in record["query"]["action_ids"]]
        if actual_ids != [int(value) for value in record["legal_mask"]]:
            raise AssertionError(
                f"request={request_id} query_action_ids 与 legal_mask 不一致"
            )
        actual_rows = np.asarray(record["query"]["rows"], dtype=np.int32)
        if actual_rows.shape[0] != 2 * len(actual_ids):
            raise AssertionError(f"request={request_id} query 行数不是 action 对数的两倍")
        query_pairs += len(actual_ids)
        for offset, action_id in enumerate(actual_ids):
            action = action_for_id(view, record, action_id)
            actual_pair = actual_rows[2 * offset : 2 * offset + 2]
            independent_pair, _offense_state = decode_query_rows(
                view, action, action_id,
            )
            compare_array(
                actual_pair[:, :QUERY_ROW_ANSWER_START],
                independent_pair[:, :QUERY_ROW_ANSWER_START],
                label=f"query[{action_id}]_header",
                request_id=request_id,
            )
            compare_array(
                actual_pair[1, QUERY_ROW_ANSWER_START:],
                independent_pair[1, QUERY_ROW_ANSWER_START:],
                label=f"query[{action_id}]_defense",
                request_id=request_id,
            )
            offense, defense = analyze_action_queries(view, action, action_id)
            expected_pair = np.stack(
                [encode_query_row(offense), encode_query_row(defense)],
                axis=0,
            )
            compare_array(
                actual_pair,
                expected_pair,
                label=f"query[{action_id}]_full",
                request_id=request_id,
            )
            action_type_counts[str(json.loads(action.to_mjai()).get("type"))] += 1

        selected = record.get("selected", {})
        tracker.record_response(
            view,
            last_type=last_context[0],
            actor=last_context[1],
            legal_jsons=list(record.get("legal_jsons", [])),
            response=selected.get("payload"),
        )
        counters["requests"] += 1
        counters["with_missing_fields"] += int(bool(record.get("missing_fields")))
        counters["model_actions"] += int(selected.get("source") == "model")
        counters["fallback_actions"] += int(selected.get("source") != "model")
        counters["history_tokens"] += int(actual_history.shape[0])
        counters["snapshot_rows"] += int(expected_kinds.shape[0])
        counters["query_rows"] += int(actual_rows.shape[0])
        counters["record_index_max"] = index

    if rebuilt_mismatches:
        raise AssertionError("重建字段进入模型前已错误:\n" + "\n".join(rebuilt_mismatches[:10]))
    return {
        "audit_log": str(path),
        "requests": counters["requests"],
        "model_actions": counters["model_actions"],
        "fallback_actions": counters["fallback_actions"],
        "history_tokens_checked": counters["history_tokens"],
        "snapshot_rows_checked": counters["snapshot_rows"],
        "query_pairs_checked": query_pairs,
        "query_rows_checked": counters["query_rows"],
        "max_history_tokens": max_history,
        "max_snapshot_rows": max_snapshot,
        "requests_with_missing_fields": counters["with_missing_fields"],
        "missing_fields_seen": dict(sorted(missing_fields.items())),
        "raw_field_mismatch_count": len(raw_field_mismatches),
        "raw_field_mismatch_examples": raw_field_mismatches[:10],
        "action_type_counts": dict(sorted(action_type_counts.items())),
        "input_errors": [],
        "unavailable_actor_inputs": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_log", type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()
    summary = audit(args.audit_log)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("online validate v16 input audit: all checks passed", flush=True)


if __name__ == "__main__":
    main()
