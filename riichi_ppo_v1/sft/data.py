"""V18 当前局面 SFT 输入编码与预计算样本读取。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..model.action_groups import action_id
from ..model.current_state import EncodedStateBatch, encode_batch
from ..model.bridge import (
    action_jsons,
)
from ..model.encoding_protocol import ENCODED_FORMAT


@dataclass(slots=True)
class EncodedSample:
    """V18 决策样本：完整 Actor 序列 + Query 元数据 + 监督动作。"""

    actor_factors: np.ndarray  # [T, 32] int32
    actor_numeric: np.ndarray  # [T, 8] float32
    query_rows: np.ndarray  # [2Q, 15] int32
    action_ids: np.ndarray  # [Q] int32
    legal_mask: np.ndarray  # [241] bool
    action: int
    year: int
    game_id: str
    kyoku_index: int
    seat: int
    decision_index: int = 0

    @property
    def token_length(self) -> int:
        """完整 Actor 序列的 token 数。"""
        return int(self.actor_factors.shape[0])

    @property
    def query_pair_count(self) -> int:
        return int(self.action_ids.shape[0])


def _member_metadata(name: str) -> tuple[int, str, int]:
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return 0, stem, 0
    try:
        return int(parts[0]), "-".join(parts[1:-1]), int(parts[-1])
    except ValueError:
        return 0, stem, 0


def encode_kyoku(
    content: str,
    *,
    year: int = 0,
    game_id: str = "",
    kyoku_index: int = 0,
) -> list[EncodedSample]:
    """Replay 一局并按当前局面协议编码每个决策。"""
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
            np.ndarray, np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, int,
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
        for seat, observation, _expert_action in batch:
            events = [[], [], [], []]
            events[seat] = list(observation.new_events())
            events_by_env.append(events)
            action_rows.append(action_jsons(observation))
        manager.apply_events_batch(env_indices, events_by_env)
        batch_indices = [seat * 4 + seat for seat, _observation, _action in batch]
        try:
            prepared = manager.prepare_decisions(batch_indices, action_rows)
        except Exception as exc:
            raise RuntimeError(
                f"failed to encode legal actions: game={game_id} kyoku={kyoku_index} seats={env_indices}"
            ) from exc
        prepared_legal = np.asarray(prepared, dtype=np.bool_)
        # Rust 直接映射 action_id → 调用方合法动作列表下标，避免 Python 侧
        # decode_actions + JSON 规范匹配（与 model/bridge.prepare 同一路径）。
        index_rows = manager.action_ids_with_source_indices(batch_indices)
        decisions: list[tuple[object, list[tuple[object, int]]]] = []
        seat_list: list[int] = []
        target_actions: list[int] = []
        for row, (seat, observation, expert_action) in enumerate(batch):
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
            legal_actions = list(observation.legal_actions())
            mappings = index_rows[row]
            actions_by_id: list[tuple[object, int]] = []
            for action_id_value, source_index in mappings:
                action_id_value = int(action_id_value)
                source_index = int(source_index)
                if not 0 <= source_index < len(legal_actions):
                    raise RuntimeError(
                        f"state machine returned invalid legal action index {source_index}: "
                        f"game={game_id} seat={seat}"
                    )
                actions_by_id.append((legal_actions[source_index], action_id_value))
            if [int(action_id_value) for _action, action_id_value in actions_by_id] != ids:
                raise RuntimeError(
                    f"state machine action-id mapping disagrees with legal mask: "
                    f"game={game_id} seat={seat}"
                )
            decisions.append((observation, actions_by_id))
            seat_list.append(seat)
            target_actions.append(int(target_action))
        encoded: EncodedStateBatch = encode_batch(decisions)
        for row, (seat, observation, _expert_action) in enumerate(batch):
            count = int(encoded.query_pair_counts[row])
            pending.append((
                seat, observation,
                encoded.actor_factors[row, : int(encoded.actor_lengths[row])].copy(),
                encoded.actor_numeric[row, : int(encoded.actor_lengths[row])].copy(),
                encoded.query_rows[row, : 2 * count].copy(),
                encoded.action_ids[row, :count].copy(),
                encoded.legal_mask[row].copy(),
                target_actions[row],
            ))
    if not pending:
        return []
    seat_samples: list[list[EncodedSample]] = [[] for _ in range(4)]
    for (
        seat, _observation, actor_factors, actor_numeric, query_rows,
        action_ids_array, legal, target_action,
    ) in pending:
        seat_samples[seat].append(EncodedSample(
            actor_factors=actor_factors,
            actor_numeric=actor_numeric,
            query_rows=query_rows,
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
    seed: int = 1,
    shuffle: bool = True,
    rank: int = 0,
    world_size: int = 1,
    include_critic: bool = True,
) -> Iterator[EncodedSample]:
    """读取 V18 预计算数据集 split。"""
    manifest_path = dataset / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("SFT training requires a V18 encoded dataset manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("format", "")) != ENCODED_FORMAT:
        raise RuntimeError("only the V18 encoded SFT format is supported")
    from .contract import validate_manifest

    validate_manifest(manifest)
    if include_critic:
        raise ValueError("the V18 encoded subset is actor-only; set train_critic: false")
    from .precompute import iter_precomputed_samples

    yield from iter_precomputed_samples(
        dataset,
        split,
        seed=seed,
        shuffle=shuffle,
        rank=rank,
        world_size=world_size,
    )
