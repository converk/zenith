"""Online kyoku replay and model-input encoding for SFT."""

from __future__ import annotations

import gzip
import json
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..model.encoding_protocol import ENCODED_FORMAT as V16_ENCODED_FORMAT
from ..model.bridge import (
    Decision,
    action_jsons,
    action_jsons_and_decision_flag,
    snapshot_json,
)
from ..model.action_query import analyze_action_queries, encode_query_row
from ..model.critic_features import (
    collect_replay_table_state,
    encode_critic_features,
    encode_public_summary,
)
from ..model.snapshot import build_snapshot_facts, encode_snapshot_rows
from ..model.semantic_validation import (
    assert_actor_token_semantics,
    assert_critic_token_semantics,
)
from ..training.rewards.decision import DecisionAnalysisBatch, action_id
from ..training.rewards.efficiency import EfficiencyAnalyzer


@dataclass(slots=True)
class SftSample:
    token_factors: np.ndarray
    token_numeric: np.ndarray
    legal_mask: np.ndarray
    action: int
    value_target: float
    critic_factors: np.ndarray
    year: int
    game_id: str
    kyoku_index: int
    seat: int
    decision_index: int = 0
    teacher_mask: np.ndarray | None = None

    @property
    def token_length(self) -> int:
        return int(self.token_factors.shape[0])

    @property
    def critic_length(self) -> int:
        return int(self.critic_factors.shape[0])


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


def _append_rows(
    factors: np.ndarray,
    numeric: np.ndarray,
    extra_factors: np.ndarray,
    extra_numeric: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(extra_factors):
        return factors, numeric
    result_factors = np.concatenate((factors, np.asarray(extra_factors, dtype=np.uint8)), axis=0)
    if extra_numeric is None:
        extra_numeric = np.zeros((len(extra_factors), 8), dtype=np.float32)
    result_numeric = np.concatenate((numeric, np.asarray(extra_numeric, dtype=np.float32)), axis=0)
    return result_factors, result_numeric


def encode_kyoku(
    content: str,
    *,
    year: int = 0,
    game_id: str = "",
    kyoku_index: int = 0,
    gamma: float = 0.99,
    analyzer: EfficiencyAnalyzer | None = None,
    include_critic: bool = True,
) -> list[SftSample]:
    """Replay one JSONL kyoku and encode all four players' decisions."""
    from .contract import assert_runtime_contract

    assert_runtime_contract()
    import riichi
    from riichienv import MjaiReplay

    replay = MjaiReplay.from_jsonl_string(content, rule="tenhou")
    kyokus = list(replay.take_kyokus())
    if len(kyokus) != 1:
        raise ValueError(f"SFT record must contain exactly one kyoku, got {len(kyokus)}")
    kyoku = kyokus[0]
    point_deltas = [int(value) for value in kyoku.grp_features()["delta_scores"]]
    if len(point_deltas) != 4:
        raise ValueError("SFT kyoku must contain four point deltas")
    analyzer = analyzer or EfficiencyAnalyzer()
    # Each seat has an independent chronological replay stream.  Keep that
    # ordering intact, but place the four streams in separate state-machine
    # tables so each time slice crosses the Python/Rust boundary once.
    manager = riichi.MjaiKyokuStateMachineManager(4)
    streams = [iter(kyoku.steps(seat=seat, skip_single_action=False)) for seat in range(4)]
    pending: list[tuple[int, object, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
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
        events_by_env = []
        action_rows = []
        snapshots = []
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
            pending.append((
                seat, observation, expert_action,
                prepared_factors[row, :length].copy(),
                prepared_numeric[row, :length].copy(),
                legal.copy(), int(target_action),
            ))

    # Candidate analysis is pure with respect to the recorded observation, so
    # one kyoku-wide batch avoids rebuilding the analyzer per decision.
    if not pending:
        return []
    decisions = [Decision(0, seat, observation) for seat, observation, _expert, _factors, _numeric, _legal, _action in pending]
    analysis = DecisionAnalysisBatch.build(decisions, analyzer=analyzer)
    legal_batch = np.stack([legal for _seat, _observation, _expert, _factors, _numeric, legal, _action in pending])
    candidate_factors, candidate_numeric = analysis.candidate_tokens(decisions, legal_batch)
    state_factors, state_numeric = analysis.state_tokens(decisions)
    teacher_masks = analysis.teacher_masks(decisions)
    seat_samples: list[list[SftSample]] = [[] for _ in range(4)]
    for row, (seat, observation, _expert_action, factors, numeric, legal, target_action) in enumerate(pending):
        table_state = collect_replay_table_state(observation)
        public = encode_public_summary(table_state, seat)
        factors, numeric = _append_rows(factors, numeric, public.factors)
        factors, numeric = _append_rows(factors, numeric, state_factors[row], state_numeric[row])
        factors, numeric = _append_rows(factors, numeric, candidate_factors[row], candidate_numeric[row])
        critic = encode_critic_features(table_state, seat) if include_critic else None
        seat_samples[seat].append(SftSample(
            token_factors=factors,
            token_numeric=numeric,
            legal_mask=legal,
            action=target_action,
            value_target=0.0,
            critic_factors=(
                critic.factors.copy() if critic is not None else np.zeros((0, 10), dtype=np.uint8)
            ),
            year=int(year), game_id=str(game_id), kyoku_index=int(kyoku_index), seat=seat,
            decision_index=len(seat_samples[seat]),
            teacher_mask=teacher_masks[row].copy(),
        ))
    all_samples: list[SftSample] = []
    for seat in range(4):
        seat_samples_for_seat = seat_samples[seat]
        terminal = float(np.clip(point_deltas[seat], -12_000, 12_000) / 1_000.0)
        remaining = len(seat_samples_for_seat) - 1
        for index, sample in enumerate(seat_samples_for_seat):
            sample.value_target = float(terminal * float(gamma) ** (remaining - index))
        all_samples.extend(seat_samples_for_seat)
    if all_samples:
        max_tokens = max(sample.token_length for sample in all_samples)
        actor_factors = np.zeros((len(all_samples), max_tokens, 10), dtype=np.uint8)
        actor_numeric = np.zeros((len(all_samples), max_tokens, 8), dtype=np.float32)
        actor_lengths = np.asarray([sample.token_length for sample in all_samples], dtype=np.int64)
        for row, sample in enumerate(all_samples):
            actor_factors[row, :sample.token_length] = sample.token_factors
            actor_numeric[row, :sample.token_length] = sample.token_numeric
        assert_actor_token_semantics(actor_factors, actor_numeric, actor_lengths)
        if include_critic:
            max_critic = max(sample.critic_length for sample in all_samples)
            critic_factors = np.zeros((len(all_samples), max_critic, 10), dtype=np.uint8)
            critic_lengths = np.asarray([sample.critic_length for sample in all_samples], dtype=np.int64)
            for row, sample in enumerate(all_samples):
                critic_factors[row, :sample.critic_length] = sample.critic_factors
            assert_critic_token_semantics(critic_factors, critic_lengths)
    return all_samples


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
                offense, defense = analyze_action_queries(observation, action, action_id_value)
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
    for seat, _observation, factors, numeric, kinds, categorical, numeric_snapshot, rows, action_ids, legal, target_action in pending:
        seat_samples[seat].append(V16Sample(
            history_factors=factors,
            history_numeric=numeric,
            snapshot_kinds=kinds,
            snapshot_cat=categorical,
            snapshot_num=numeric_snapshot,
            query_rows=rows,
            action_ids=action_ids,
            legal_mask=legal,
            action=target_action,
            year=int(year), game_id=str(game_id), kyoku_index=int(kyoku_index),
            seat=seat, decision_index=len(seat_samples[seat]),
        ))
    return [
        sample
        for rows in seat_samples
        for sample in rows
    ]


def _member_metadata(name: str) -> tuple[int, str, int]:
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return 0, stem, 0
    try:
        return int(parts[0]), "-".join(parts[1:-1]), int(parts[-1])
    except ValueError:
        return 0, stem, 0


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
) -> Iterator[SftSample]:
    manifest_path = dataset / "manifest.json"
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("format", "")) == V16_ENCODED_FORMAT:
            from .contract import validate_v16_manifest

            validate_v16_manifest(manifest)
            if include_critic:
                raise ValueError("the encoded subset is actor-only; set train_critic: false")
            from .precompute import iter_precomputed_v16_samples
            yield from iter_precomputed_v16_samples(
                dataset,
                split,
                seed=seed,
                shuffle=shuffle,
                rank=rank,
                world_size=world_size,
            )
            return
        if str(manifest.get("format", "")).startswith("riichi-sft-encoded-v"):
            from .contract import validate_v13_manifest

            validate_v13_manifest(manifest)
            if include_critic:
                raise ValueError("the encoded subset is actor-only; set train_critic: false")
            from .precompute import iter_precomputed_samples
            yield from iter_precomputed_samples(
                dataset,
                split,
                seed=seed,
                shuffle=shuffle,
                rank=rank,
                world_size=world_size,
            )
            return
    shards = sorted((dataset / split).glob(f"{split}-*.tar"))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(shards)
    shards = shards[int(rank)::int(world_size)]
    analyzer = EfficiencyAnalyzer()
    buffer: list[list[SftSample]] = []
    for shard in shards:
        with tarfile.open(shard, "r") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            if shuffle:
                rng.shuffle(members)
            for member in members:
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"cannot read {shard}:{member.name}")
                payload = extracted.read()
                content = gzip.decompress(payload).decode("utf-8") if payload[:2] == b"\x1f\x8b" else payload.decode("utf-8")
                year, game_id, kyoku_index = _member_metadata(member.name)
                samples = encode_kyoku(
                    content,
                    year=year,
                    game_id=game_id,
                    kyoku_index=kyoku_index,
                    gamma=gamma,
                    analyzer=analyzer,
                    include_critic=include_critic,
                )
                if not shuffle:
                    yield from samples
                    continue
                buffer.append(samples)
                if len(buffer) >= shuffle_buffer_kyokus:
                    chosen = rng.randrange(len(buffer))
                    yield from buffer.pop(chosen)
    while buffer:
        chosen = rng.randrange(len(buffer)) if shuffle else 0
        yield from buffer.pop(chosen)
