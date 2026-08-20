"""V16 SFT 输入编码与预计算样本读取。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..model.action_groups import action_id
from ..model.action_query import analyze_action_queries, encode_query_row
from ..model.bridge import (
    action_jsons,
    action_jsons_and_decision_flag,
    snapshot_json,
)
from ..model.encoding_protocol import ENCODED_FORMAT as V16_ENCODED_FORMAT
from ..model.snapshot import build_snapshot_facts, encode_snapshot_rows


@dataclass(slots=True)
class V16Sample:
    """V16 决策样本:Objective Facts + Snapshot + 每动作一对 Query。"""

    history_factors: np.ndarray
    history_numeric: np.ndarray
    snapshot_kinds: np.ndarray
    snapshot_cat: np.ndarray
    snapshot_num: np.ndarray
    query_rows: np.ndarray
    action_ids: np.ndarray
    legal_mask: np.ndarray
    action: int
    year: int
    game_id: str
    kyoku_index: int
    seat: int
    decision_index: int = 0

    @property
    def history_length(self) -> int:
        return int(self.history_factors.shape[0])

    @property
    def snapshot_length(self) -> int:
        return int(self.snapshot_kinds.shape[0])

    @property
    def query_pair_count(self) -> int:
        return int(self.action_ids.shape[0])

    @property
    def token_length(self) -> int:
        """长度分桶键:完整目标序列的 token 数。"""
        return self.history_length + self.snapshot_length + 2 * self.query_pair_count


def _member_metadata(name: str) -> tuple[int, str, int]:
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return 0, stem, 0
    try:
        return int(parts[0]), "-".join(parts[1:-1]), int(parts[-1])
    except ValueError:
        return 0, stem, 0


def encode_kyoku_v16(
    content: str,
    *,
    year: int = 0,
    game_id: str = "",
    kyoku_index: int = 0,
) -> list[V16Sample]:
    """Replay 一局并编码 V16 输入(Objective Facts + Snapshot + Query 对)。"""
    from .contract import assert_runtime_contract

    assert_runtime_contract()
    import riichi
    from riichienv import MjaiReplay

    replay = MjaiReplay.from_jsonl_string(content, rule="tenhou")
    kyokus = list(replay.take_kyokus())
    if len(kyokus) != 1:
        raise ValueError(f"SFT record must contain exactly one kyoku, got {len(kyokus)}")
    kyoku = kyokus[0]
    manager = riichi.MjaiKyokuStateMachineManager(4)
    streams = [iter(kyoku.steps(seat=seat, skip_single_action=False)) for seat in range(4)]
    pending: list[
        tuple[
            int, object,
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, np.ndarray, np.ndarray, int,
        ]
    ] = []
    active = set(range(4))
    while active:
        batch: list[tuple[int, object, object]] = []
        for seat in sorted(active):
            try:
                observation, expert_action = next(streams[seat])
            except StopIteration:
                active.remove(seat)
            else:
                batch.append((seat, observation, expert_action))
        if not batch:
            continue
        env_indices = [seat for seat, _observation, _action in batch]
        events_by_env: list[list[list[str]]] = []
        action_rows: list[list[str]] = []
        snapshots: list[str] = []
        for seat, observation, _expert_action in batch:
            events = [[], [], [], []]
            events[seat] = list(observation.new_events())
            events_by_env.append(events)
            actions, decision_flag = action_jsons_and_decision_flag(observation)
            action_rows.append(actions)
            snapshots.append(snapshot_json(observation, decision_flag))
        manager.apply_events_batch(env_indices, events_by_env)
        batch_indices = [seat * 4 + seat for seat, _observation, _action in batch]
        try:
            prepared = manager.prepare_decisions(batch_indices, action_rows, snapshots)
        except Exception as exc:
            raise RuntimeError(
                f"failed to encode legal actions: game={game_id} kyoku={kyoku_index} seats={env_indices}"
            ) from exc
        prepared_factors = np.asarray(prepared[0], dtype=np.uint8)
        prepared_numeric = np.asarray(prepared[1], dtype=np.float32)
        prepared_lengths = np.asarray(prepared[2], dtype=np.int64)
        prepared_legal = np.asarray(prepared[3], dtype=np.bool_)
        for row, (seat, observation, expert_action) in enumerate(batch):
            length = int(prepared_lengths[row])
            legal = prepared_legal[row]
            target_action = action_id(expert_action, observation)
            if target_action is None:
                raise RuntimeError(f"expert action cannot be mapped: seat={seat} action={expert_action}")
            if not legal[int(target_action)]:
                raise RuntimeError(
                    f"expert action is outside legal mask: seat={seat} action_id={target_action}"
                )
            ids = [int(value) for value in np.flatnonzero(legal).tolist()]
            if not ids:
                raise RuntimeError(f"empty legal mask: game={game_id} seat={seat}")
            batch_index = seat * 4 + seat
            decoded = manager.decode_actions([batch_index] * len(ids), ids)
            legal_actions = list(observation.legal_actions())
            templates = action_jsons(observation)
            if len(legal_actions) != len(templates):
                raise RuntimeError(f"legal/template length mismatch: game={game_id} seat={seat}")
            representative: dict[str, object] = {}
            for action, template in zip(legal_actions, templates, strict=True):
                key = json.dumps(json.loads(template), separators=(",", ":"), sort_keys=True)
                representative.setdefault(key, action)
            query_rows: list[np.ndarray] = []
            action_ids: list[int] = []
            for action_id_value, raw in zip(ids, decoded, strict=True):
                key = json.dumps(json.loads(raw), separators=(",", ":"), sort_keys=True)
                action = representative.get(key)
                if action is None:
                    raise RuntimeError(
                        f"decoded action_id={action_id_value} has no legal representative: "
                        f"game={game_id} seat={seat} mjai={raw}"
                    )
                offense, defense = analyze_action_queries(
                    observation, action, action_id_value,
                )
                query_rows.append(encode_query_row(offense))
                query_rows.append(encode_query_row(defense))
                action_ids.append(action_id_value)
            kinds, categorical, numeric_snapshot = encode_snapshot_rows(
                build_snapshot_facts(observation),
            )
            pending.append((
                seat, observation,
                prepared_factors[row, :length].copy(),
                prepared_numeric[row, :length].copy(),
                kinds, categorical, numeric_snapshot,
                np.asarray(query_rows, dtype=np.int32),
                np.asarray(action_ids, dtype=np.int32),
                legal.copy(), int(target_action),
            ))
    if not pending:
        return []
    seat_samples: list[list[V16Sample]] = [[] for _ in range(4)]
    for (
        seat, _observation, factors, numeric, kinds, categorical,
        numeric_snapshot, rows, action_ids_array, legal, target_action,
    ) in pending:
        seat_samples[seat].append(V16Sample(
            history_factors=factors,
            history_numeric=numeric,
            snapshot_kinds=kinds,
            snapshot_cat=categorical,
            snapshot_num=numeric_snapshot,
            query_rows=rows,
            action_ids=action_ids_array,
            legal_mask=legal,
            action=target_action,
            year=int(year),
            game_id=str(game_id),
            kyoku_index=int(kyoku_index),
            seat=seat,
            decision_index=len(seat_samples[seat]),
        ))
    return [
        sample
        for rows in seat_samples
        for sample in rows
    ]


def iter_split_samples(
    dataset: Path,
    split: str,
    *,
    gamma: float = 0.99,
    seed: int = 1,
    shuffle: bool = True,
    shuffle_buffer_kyokus: int = 8192,
    rank: int = 0,
    world_size: int = 1,
    include_critic: bool = True,
) -> Iterator[V16Sample]:
    """读取 V16 预计算数据集 split。"""
    del gamma, shuffle_buffer_kyokus
    manifest_path = dataset / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("SFT training requires a v16 encoded dataset manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("format", "")) != V16_ENCODED_FORMAT:
        raise RuntimeError("only the v16 encoded SFT format is supported")
    from .contract import validate_v16_manifest

    validate_v16_manifest(manifest)
    if include_critic:
        raise ValueError("the v16 encoded subset is actor-only; set train_critic: false")
    from .precompute import iter_precomputed_v16_samples

    yield from iter_precomputed_v16_samples(
        dataset,
        split,
        seed=seed,
        shuffle=shuffle,
        rank=rank,
        world_size=world_size,
    )
