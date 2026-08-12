"""Top4 paired counterfactual rollout + GRP V2 teacher (GOAL_PROMPT_STEP2).

For every selected decision state ``I`` (seat ``s``):

1. reconstruct the full game state by replaying the raw MJAI kyoku log to the
   target decision index (per-seat);
2. sample ``N`` hidden worlds consistent with public information (own hand +
   melds + discards known; opponents' hands and wall resampled from the
   remaining tile multiset);
3. in each world, force Policy Top1..Top4 (shared worlds, four branches);
4. continue greedily with the configured policy to the end of the kyoku;
5. evaluate the kyoku-ending state with GRP V2 (target seat's expected
   final-rank utility) and record paired differences vs Policy Top1.

The world budget is adaptive: start at ``min_worlds``, add ``increment`` until
every challenger-vs-Top1 pair is statistically determined at the configured
z-level, or the per-decision cap is reached.  Stability rows are written at
each intermediate world count for the convergence analysis.

Run two shards with ``CUDA_DEVICE=0`` and ``CUDA_DEVICE=2`` (physical GPUs 0
and 3) on this host.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import sys
import tarfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from riichienv import RiichiEnv  # noqa: E402
from riichienv import MjaiReplay  # noqa: E402
import riichi  # noqa: E402

from riichi_ppo_v1.grp.model import (  # noqa: E402
    RankPredictor,
    grp_features_from_scores,
    reward_from_rank_probs,
)
from riichi_ppo_v1.model.bridge import BatchedStateBridge  # noqa: E402
from riichi_ppo_v1.sft.policy_adapter import load_policy_adapter  # noqa: E402
from riichi_ppo_v1.training.rewards import (  # noqa: E402
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)
import riichi_ppo_v1.training.rewards.decision as _decision_module  # noqa: E402
from riichi_ppo_v1.training.rewards.efficiency import HandAnalysis  # noqa: E402
from riichi_ppo_v1.training.rewards.decision import (  # noqa: E402
    action_id,
    action_kind,
)
from riichi_ppo_v1.training.worker import active_decisions  # noqa: E402


@dataclass
class ExperimentConfig:
    seed_base: int = 20260812
    continuation_policy: str = (
        "checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt"
    )
    grp_checkpoint: str = (
        "checkpoints/train_riichi_ppo_goal_grp_v2/grp_rank_predictor.pt"
    )
    grp_pts_weight: tuple[float, float, float, float] = (10.0, 4.0, -4.0, -10.0)
    game_mode: str = "4p-red-half"
    raw_index: str = "datasets/tenhou_sft_2024_2025/index.jsonl.gz"
    raw_validation_dir: str = "datasets/tenhou_sft_2024_2025/validation"

    min_worlds: int = 16
    world_increment: int = 16
    max_worlds: int = 64
    max_worlds_stability_subset: int = 128
    stability_subset_count: int = 40
    z_determined: float = 1.96
    z_secondary: float = 1.28
    analysis_cache_capacity: int = 262_144


GAP_BUCKETS: list[tuple[str, float, float]] = [
    ("lt005", -float("inf"), 0.05),
    ("05_20", 0.05, 0.20),
    ("20_50", 0.20, 0.50),
    ("50_70", 0.50, 0.70),
    ("ge70", 0.70, float("inf")),
]


def gap_bucket(gap: float) -> str:
    for name, lo, hi in GAP_BUCKETS:
        if lo <= gap < hi:
            return name
    return "ge70"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed % (2**32))


def _field(observation: Any) -> tuple[int, int, int, int]:
    return (
        int(observation.round_wind),
        int(observation.kyoku_index),
        int(observation.honba),
        int(observation.riichi_sticks),
    )


class RawKyokuStore:
    """Locate and read raw per-kyoku MJAI logs from the validation tars."""

    def __init__(self, index_path: Path, validation_dir: Path) -> None:
        self.index_path = Path(index_path)
        self.validation_dir = Path(validation_dir)
        self.locations: dict[tuple[str, int], str] = {}
        self._tars: dict[Path, tarfile.TarFile] = {}

    def load_needed(self, game_ids: set[str]) -> None:
        if not game_ids:
            return
        with gzip.open(self.index_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != "validation":
                    continue
                game_id = str(row.get("game_id", ""))
                if game_id not in game_ids:
                    continue
                self.locations[(game_id, int(row["kyoku_index"]))] = str(row["location"])

    def read(self, game_id: str, kyoku_index: int) -> str:
        location = self.locations.get((game_id, kyoku_index))
        if location is None:
            raise KeyError(f"raw kyoku not found: {game_id}/{kyoku_index}")
        tar_name, member_name = location.split(":", 1)
        tar_path = self.validation_dir / Path(tar_name).name
        archive = self._tars.get(tar_path)
        if archive is None:
            archive = tarfile.open(tar_path, "r")
            self._tars[tar_path] = archive
        member = archive.extractfile(member_name)
        if member is None:
            raise KeyError(f"member not found: {location}")
        payload = member.read()
        if payload[:2] == b"\x1f\x8b":
            payload = gzip.decompress(payload)
        return payload.decode("utf-8")

    def close(self) -> None:
        for archive in self._tars.values():
            archive.close()
        self._tars.clear()


def replay_to_decision(
    content: str,
    *,
    seat: int,
    decision_index: int,
) -> tuple[RiichiEnv, Any, list[int], dict[str, Any]]:
    """Replay a kyoku to the target per-seat decision; return (env, obs, kyoku_initial, meta)."""
    events = [json.loads(line) for line in content.splitlines()]
    decision_event_index = _find_decision_event_index(
        events, seat=seat, decision_index=decision_index
    )
    env = _build_env(events[: decision_event_index + 1])
    target_obs = env.get_observations()[seat]
    kyoku_initial, meta = _kyoku_meta(events)
    meta["decision_event_index"] = decision_event_index
    return env, target_obs, kyoku_initial, meta


def _kyoku_meta(events: list[dict[str, Any]]) -> tuple[list[int], dict[str, Any]]:
    start = next(
        (event for event in events if event.get("type") == "start_kyoku"),
        None,
    )
    if start is None:
        raise RuntimeError("start_kyoku missing from raw log")
    kyoku_initial = [int(x) for x in start["scores"]]
    bakaze = str(start.get("bakaze", "E"))
    chang = {"E": 0, "S": 1, "W": 2, "N": 3}.get(bakaze, 0)
    ju = max(0, int(start.get("kyoku", 1)) - 1)
    meta = {
        "round_wind": chang,
        "kyoku_index": ju,
        "honba": int(start.get("honba", 0)),
        "riichi_sticks": int(start.get("kyotaku", 0)),
        "kyoku_initial_scores": kyoku_initial,
    }
    return kyoku_initial, meta


def _find_decision_event_index(
    events: list[dict[str, Any]],
    *,
    seat: int,
    decision_index: int,
) -> int:
    """Return the raw-event index after which the target decision is available.

    The SFT encoder counts per-seat decisions from ``Kyoku.steps``, which
    includes synthesized implicit-pass windows that RiichiEnv's observations
    do not expose.  We therefore locate the decision through the canonical
    per-seat observation (hand multiset + legal-action set) and find the raw
    event index whose replayed environment reproduces it exactly.
    """
    jsonl = "\n".join(
        json.dumps(event, ensure_ascii=False) for event in events
    )
    replay = MjaiReplay.from_jsonl_string(jsonl, rule="tenhou")
    kyokus = list(replay.take_kyokus())
    if len(kyokus) != 1:
        raise RuntimeError("expected exactly one kyoku in the raw log")
    target_observation = None
    for index, (observation, _expert_action) in enumerate(
        kyokus[0].steps(seat=seat, skip_single_action=False)
    ):
        if index == decision_index:
            target_observation = observation
            break
    if target_observation is None:
        raise RuntimeError(
            f"decision not found: seat={seat} decision_index={decision_index}"
        )
    target_hand = Counter(int(tile) // 4 for tile in target_observation.hands[seat])
    target_drawn = getattr(target_observation, "drawn_tile", None)
    if not target_observation.new_events():
        # The target decision is a synthesized implicit-pass window.  It is
        # triggered by the previous non-empty (real) event; locate that event
        # by matching the immediately preceding canonical observation.
        steps = list(
            kyokus[0].steps(seat=seat, skip_single_action=False)
        )
        prev_index = decision_index - 1
        while prev_index >= 0 and not steps[prev_index][0].new_events():
            prev_index -= 1
        if prev_index < 0:
            raise RuntimeError("implicit-pass decision has no preceding event")
        prev_obs = steps[prev_index][0]
        target_hand = Counter(int(tile) // 4 for tile in prev_obs.hands[seat])
        target_drawn = getattr(prev_obs, "drawn_tile", None)
    env = RiichiEnv(game_mode="4p-red-half", skip_mjai_logging=False)
    for event_index, event in enumerate(events):
        env.apply_event(event)
        observations = env.get_observations()
        if seat not in observations:
            continue
        observation = observations[seat]
        if not observation.legal_actions():
            continue
        hand_now = Counter(int(tile) // 4 for tile in env.hands[seat])
        drawn_now = getattr(observation, "drawn_tile", None)
        drawn_ok = (
            target_drawn is None
            or drawn_now is None
            or int(target_drawn) == int(drawn_now)
        )
        if hand_now == target_hand and drawn_ok:
            return event_index
    raise RuntimeError(
        f"cannot align decision event for seat={seat} "
        f"decision_index={decision_index}: "
        f"target_hand={dict(target_hand)} target_drawn={target_drawn}"
    )


def _build_env(prefix_events: list[dict[str, Any]]) -> RiichiEnv:
    """Replay a prefix of MJAI events without consuming event cursors.

    MJAI logging stays enabled: the batched policy state machine reconstructs
    its view from per-player event streams, and kyoku/game boundaries are only
    surfaced to the bridge through end_kyoku/end_game events.  No observation
    is taken here so the first ``bridge.sync`` receives the full history.
    """
    env = RiichiEnv(game_mode="4p-red-half", skip_mjai_logging=False)
    for event in prefix_events:
        env.apply_event(event)
    return env


def build_pre_streams(
    events: list[dict[str, Any]],
    decision_event_index: int,
) -> list[list[str]]:
    """Build the canonical per-seat event streams used by the SFT encoder.

    The training-time pipeline encodes each decision from the per-seat
    ``Kyoku.steps`` streams (which include the reconstructed initial-hand
    draws and implicit-pass observations), not from the raw RiichiEnv logs.
    Rebuilding those streams for the (possibly rewritten) world and feeding
    them to the batched state machine makes the rollout policy see the same
    feature history as the model was trained on.
    """
    counts = [0, 0, 0, 0]
    scratch = RiichiEnv(game_mode="4p-red-half", skip_mjai_logging=False)
    for event in events[: decision_event_index + 1]:
        scratch.apply_event(event)
        observations = scratch.get_observations()
        for pid, observation in observations.items():
            if observation.legal_actions():
                counts[int(pid)] += 1
    # Only events up to (and including) the decision are part of the world:
    # later events belong to the future continuation and are not assigned.
    jsonl = "\n".join(
        json.dumps(event, ensure_ascii=False)
        for event in events[: decision_event_index + 1]
    )
    replay = MjaiReplay.from_jsonl_string(jsonl, rule="tenhou")
    kyokus = list(replay.take_kyokus())
    if len(kyokus) != 1:
        raise RuntimeError("expected exactly one kyoku in the world log")
    kyoku = kyokus[0]
    streams: list[list[str]] = []
    for seat in range(4):
        stream: list[str] = []
        for index, (observation, _expert_action) in enumerate(
            kyoku.steps(seat=seat, skip_single_action=False)
        ):
            stream.extend(observation.new_events())
            if counts[seat] > 0 and index == counts[seat] - 1:
                break
        streams.append(stream)
    return streams


def _consumed_from_hand(meld_rows: list[Any]) -> Counter:
    """Tiles removed from a player's own hand by open/closed melds."""
    result: Counter = Counter()
    for meld in meld_rows:
        tiles = [int(tile) for tile in meld.tiles]
        opened = bool(getattr(meld, "opened", True))
        if opened:
            result.update(tiles[1:])  # first tile is the called discard
        else:
            result.update(tiles)  # ankan: all four from hand
    return result


def _build_unknown_pool(env: RiichiEnv, seat: int) -> list[int]:
    own_count: Counter = Counter(int(tile) for tile in env.hands[seat])
    discard_count: Counter = Counter()
    for discards in env.discards:
        discard_count.update(int(tile) for tile in discards)
    meld_count: Counter = Counter()
    called_count: Counter = Counter()
    visible_ids: set[int] = set()
    visible_ids.update(int(tile) for tile in env.hands[seat])
    for discards in env.discards:
        visible_ids.update(int(tile) for tile in discards)
    for meld_rows in env.melds:
        for meld in meld_rows:
            tiles = [int(tile) for tile in meld.tiles]
            meld_count.update(tiles)
            if bool(getattr(meld, "opened", True)):
                called_count.update(tiles[:1])
            visible_ids.update(tiles)

    pool: list[int] = []
    for tile_type in range(34):
        visible_type = (
            sum(count for tile_id, count in own_count.items() if tile_id // 4 == tile_type)
            + sum(count for tile_id, count in discard_count.items() if tile_id // 4 == tile_type)
            + sum(count for tile_id, count in meld_count.items() if tile_id // 4 == tile_type)
            - sum(count for tile_id, count in called_count.items() if tile_id // 4 == tile_type)
        )
        remaining = 4 - visible_type
        available = [
            4 * tile_type + offset
            for offset in range(4)
            if 4 * tile_type + offset not in visible_ids
        ]
        if len(available) < remaining:
            # Replay collapses duplicate physical copies of the same tile
            # string to one id; pad with the missing ids.
            available = list(dict.fromkeys(available + [
                4 * tile_type + offset
                for offset in range(4)
            ]))
        if len(available) < remaining:
            raise RuntimeError(
                f"tile type {tile_type}: visible={visible_type} available={available}"
            )
        pool.extend(available[:remaining])
    return pool


def _sample_wall(pool: list[int], env: RiichiEnv, rng: np.random.Generator) -> list[int]:
    rest = list(pool)
    new_wall = [0] * len(env.wall)
    fixed: dict[int, int] = {}
    for index, dora_tile in enumerate(env.dora_indicators):
        slot = 4 + 2 * index - env.rinshan_draw_count
        fixed[slot] = int(dora_tile)
        tile_type = int(dora_tile) // 4
        found = next(
            (j for j, tile in enumerate(rest) if int(tile) // 4 == tile_type),
            None,
        )
        if found is not None:
            rest.pop(found)
        else:
            rest.pop()
    for slot in range(len(new_wall)):
        if slot in fixed:
            new_wall[slot] = fixed[slot]
        else:
            new_wall[slot] = int(rest.pop())
    return new_wall


def _assign_draws(
    events: list[dict[str, Any]],
    *,
    player: int,
    sampled_hand: list[int],
    discards: list[int],
    consumed: Counter,
    melds: list[Any],
    rng: np.random.Generator,
) -> tuple[dict[int, str], list[int]]:
    """Assign a valid per-tsumo draw tile string for one player.

    The sampled hand plus the public discards/calls determine the multiset of
    tiles the player has ever held.  Walking the raw event timeline, every
    removed tile (discard/call consumption) is assigned to the most recent
    unassigned tsumo slot before its removal; tiles removed before any draw
    must come from the initial 13-tile hand.  The remaining tiles fill the
    leftover tsumo slots, which keeps every public event legal in replay.
    """
    drawn_ever: Counter = Counter(int(tile) for tile in sampled_hand)
    drawn_ever.update(int(tile) for tile in discards)
    drawn_ever.update(consumed)
    held_pool = list(drawn_ever.elements())
    rng.shuffle(held_pool)
    if len(held_pool) < 13:
        raise RuntimeError(
            f"player {player}: held-tile multiset smaller than 13 "
            f"({len(held_pool)})"
        )
    draws: Counter = Counter(held_pool)
    tsumo_count = sum(
        1
        for event in events
        if event.get("type") == "tsumo" and int(event["actor"]) == player
    )
    if sum(draws.values()) != tsumo_count + 13:
        raise RuntimeError(
            f"player {player}: held tiles={sum(draws.values())} != "
            f"tsumo events + 13 ({tsumo_count + 13})"
        )
    initial_bag: list[int] = []
    slots: list[int] = []  # event indices of unassigned tsumo events
    assignment: dict[int, str] = {}
    for index, event in enumerate(events):
        kind = event.get("type")
        if kind == "tsumo" and int(event["actor"]) == player:
            slots.append(index)
        else:
            removed = _removed_tiles(event, player, melds)
            if removed is None:
                continue
            for tile_id in removed:
                if draws[tile_id] <= 0:
                    raise RuntimeError(
                        f"player {player}: removal tile {tile_id} not in held multiset"
                    )
                draws[tile_id] -= 1
                free_slot = next(
                    (slot for slot in reversed(slots) if slot not in assignment),
                    None,
                )
                if free_slot is not None:
                    assignment[free_slot] = _tile_to_mjai(tile_id)
                else:
                    initial_bag.append(tile_id)
    if len(initial_bag) > 13:
        raise RuntimeError(
            f"player {player}: initial hand needs {len(initial_bag)} tiles"
        )
    # Remaining draws fill the leftover tsumo slots; the rest of the pool
    # completes the 13-tile initial hand.
    leftover = list(draws.elements())
    rng.shuffle(leftover)
    for slot in slots:
        if slot in assignment:
            continue
        tile_id = int(leftover.pop())
        assignment[slot] = _tile_to_mjai(tile_id)
    if len(leftover) != 13 - len(initial_bag):
        raise RuntimeError(
            f"player {player}: initial hand size mismatch: "
            f"bag={len(initial_bag)} leftover={len(leftover)}"
        )
    initial13 = initial_bag + leftover
    return assignment, initial13


def _removed_tiles(
    event: dict[str, Any],
    player: int,
    melds: list[Any],
) -> list[int] | None:
    """Tiles removed from ``player``'s hand by one event (or None)."""
    if int(event.get("actor", -1)) != player:
        return None
    kind = event.get("type")
    if kind == "dahai":
        return [_mjai_to_tile_id(str(event["pai"]))]
    if kind in {"chi", "pon", "daiminkan"}:
        return [
            _mjai_to_tile_id(str(tile))
            for tile in event.get("consumed", ())
            if str(tile)
        ]
    if kind == "kakan":
        # The fourth tile of the pair is added from hand.
        return [_mjai_to_tile_id(str(event.get("pai", "")))]
    if kind == "ankan":
        consumed_tiles = event.get("consumed") or ()
        if consumed_tiles:
            return [_mjai_to_tile_id(str(tile)) for tile in consumed_tiles]
        tile_type = _mjai_to_tile_id(str(event.get("pai", ""))) // 4
        ankan_tiles = [
            int(tile)
            for meld in melds
            if not bool(getattr(meld, "opened", True))
            for tile in meld.tiles
            if int(tile) // 4 == tile_type
        ]
        if len(ankan_tiles) != 4:
            raise RuntimeError(
                f"player {player}: expected 4 ankan tiles for type {tile_type}, "
                f"got {len(ankan_tiles)}"
            )
        return ankan_tiles
    return None


def _mjai_to_tile_id(tile_str: str) -> int:
    """MJAI tile string -> physical tile id (matches Rust ``mjai_to_tid``)."""
    honors = ["E", "S", "W", "N", "P", "F", "C"]
    if tile_str in honors:
        return 108 + honors.index(tile_str) * 4
    red = {"5mr": 16, "5pr": 52, "5sr": 88}
    if tile_str in red:
        return red[tile_str]
    if len(tile_str) < 2:
        raise ValueError(f"invalid MJAI tile: {tile_str!r}")
    num = int(tile_str[0])
    suit = tile_str[1]
    suit_idx = {"m": 0, "p": 1, "s": 2}.get(suit)
    if suit_idx is None or not 1 <= num <= 9:
        raise ValueError(f"invalid MJAI tile: {tile_str!r}")
    base = suit_idx * 36 + (num - 1) * 4
    return base + 1 if num == 5 else base


def _tile_to_mjai(tile_id: int) -> str:
    from riichi_ppo_v1.model.bridge import tile_id_to_mjai

    value = tile_id_to_mjai(int(tile_id))
    if value is None:
        raise RuntimeError(f"cannot map tile id {tile_id}")
    return value


def sample_world(
    env: RiichiEnv,
    seat: int,
    rng: np.random.Generator,
    events: list[dict[str, Any]],
    *,
    decision_index: int,
    decision_event_index: int | None = None,
) -> tuple[RiichiEnv, list[dict[str, Any]]]:
    """Sample a full hidden world and replay a consistent MJAI history.

    Known: the deciding player's hand, all melds, all discards.  Unknown: the
    remaining tile multiset, which is distributed uniformly over opponents'
    concealed hands (fixed public sizes) and the remaining wall slots.
    Revealed dora slots stay at their fixed dead-wall positions.

    Because the batched policy state machine reconstructs hands from the MJAI
    event stream, the sampled world is materialized by rewriting the raw
    log's per-player tsumo tiles to a draw sequence that yields the sampled
    hands while keeping every public discard/call event legal.  The wall
    order is then injected into the replayed environment.

    The raw log does not record the wall order, so there is no single "true"
    wall to leak; every world is drawn from this public-consistent baseline
    distribution (opponent hand + wall order posterior).
    """
    pool = _build_unknown_pool(env, seat)
    expected = len(env.wall) + sum(
        len(env.hands[i]) for i in range(4) if i != seat
    )
    if len(pool) != expected:
        raise RuntimeError(
            f"world pool mismatch: pool={len(pool)} expected={expected}"
        )
    rng.shuffle(pool)
    pos = 0
    sampled_hands: list[list[int]] = []
    for i in range(4):
        if i == seat:
            sampled_hands.append(list(env.hands[i]))
        else:
            size = len(env.hands[i])
            sampled_hands.append([int(tile) for tile in pool[pos : pos + size]])
            pos += size
    new_wall = _sample_wall(pool[pos:], env, rng)

    # Rewrite the event stream with sampled initial hands and per-player draws.
    original_tehais = next(
        (event for event in events if event.get("type") == "start_kyoku"),
        None,
    )
    if original_tehais is None:
        raise RuntimeError("missing start_kyoku event")
    rewrite_tehais = list(original_tehais.get("tehais", []))
    draws: dict[int, dict[int, str]] = {}
    initial13_by_player: dict[int, list[int]] = {}
    # Only events up to (and including) the decision constrain the world.
    if decision_event_index is None:
        decision_event_index = _find_decision_event_index(
            events, seat=seat, decision_index=decision_index
        )
    prefix_events = events[: decision_event_index + 1]
    for i in range(4):
        if i == seat:
            continue
        draws[i], initial13_by_player[i] = _assign_draws(
            prefix_events,
            player=i,
            sampled_hand=sampled_hands[i],
            discards=[int(tile) for tile in env.discards[i]],
            consumed=_consumed_from_hand(env.melds[i]),
            melds=env.melds[i],
            rng=rng,
        )
        rewrite_tehais[i] = [
            _tile_to_mjai(tile) for tile in initial13_by_player[i]
        ]
    rewritten: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        row = dict(event)
        if event["type"] == "start_kyoku":
            row["tehais"] = rewrite_tehais
        elif (
            event["type"] == "tsumo"
            and int(event["actor"]) != seat
            and index in draws.get(int(event["actor"]), {})
        ):
            row["pai"] = draws[int(event["actor"])][index]
        rewritten.append(row)
    world = _build_env(rewritten[: decision_event_index + 1])
    world.wall = new_wall
    for i in range(4):
        if i == seat:
            continue
        sampled_counts = Counter(int(tile) // 4 for tile in sampled_hands[i])
        replayed_counts = Counter(int(tile) // 4 for tile in world.hands[i])
        if sampled_counts != replayed_counts:
            raise RuntimeError(
                f"world replay mismatch for player {i}: "
                f"sampled={dict(sampled_counts)} replayed={dict(replayed_counts)}"
            )
    return world, rewritten


class BatchPlayer:
    """Greedy batched continuation policy for a set of RiichiEnv tables."""

    def __init__(
        self,
        adapter: Any,
        envs: list[RiichiEnv],
        *,
        env_labels: list[str] | None = None,
        initial_streams: list[list[list[str]]] | None = None,
        analysis_cache_capacity: int = 262_144,
    ) -> None:
        self.adapter = adapter
        self.envs = envs
        self.n = len(envs)
        self.env_labels = env_labels or [str(i) for i in range(len(envs))]
        self.state_machine = riichi.MjaiKyokuStateMachineManager(self.n)
        self.bridge = BatchedStateBridge(self.state_machine, self.n)
        self.public = PublicStateTracker(self.n)
        self.analyzer = EfficiencyAnalyzer(analysis_cache_capacity)
        self._install_tolerant_analyzer()
        if initial_streams is not None:
            if len(initial_streams) != self.n:
                raise ValueError("initial_streams must have one row per env")
            self.state_machine.apply_events_batch(
                list(range(self.n)), initial_streams
            )
            # Consume the env's accumulated event cursors so the first
            # ``bridge.sync`` only sees post-decision deltas.
            for env in envs:
                observations = env.get_observations()
                for seat in range(4):
                    observations[seat].new_events()
            self.public.update(initial_streams)
            self.observations = [env.get_observations() for env in envs]
            self.bridge.observations_by_env = self.observations
            self.bridge.last_events = [[] for _ in range(self.n)]
        else:
            self.observations = [env.get_observations() for env in envs]
            self.bridge.sync(self.observations)
            self.public.update(self.bridge.last_events)

    def _install_tolerant_analyzer(self) -> None:
        """Fallback build for legal-but-edge hands the Rust analyzer rejects.

        ``riichi.analyze_hands`` enforces ``concealed + 3*open_melds == 13/14``.
        A player who has drawn more tiles than they discarded relative to
        their calls (e.g. kan-containing hands) can legally hold 15-17
        physical tiles, which this invariant cannot represent.  For those
        rare candidates we substitute a neutral analysis so the rollout can
        continue; the affected candidate features are documented as a
        limitation in the report.
        """
        original = self.analyzer.analyze

        def tolerant(hands: list[np.ndarray], open_melds: list[int]) -> list[HandAnalysis]:
            rows = list(zip(hands, open_melds))
            valid = [
                (index, hand, melds)
                for index, (hand, melds) in enumerate(rows)
                if int(np.asarray(hand).sum()) + 3 * int(melds) in (13, 14)
            ]
            result: list[HandAnalysis | None] = [None] * len(rows)
            if valid:
                values = original(
                    [hand for _index, hand, _melds in valid],
                    [melds for _index, _hand, melds in valid],
                )
                for (index, _hand, _melds), value in zip(valid, values, strict=True):
                    result[index] = value
            placeholder = HandAnalysis(6, 0, 6, 6, 6)
            for index in range(len(rows)):
                if result[index] is None:
                    result[index] = placeholder
            return [value for value in result if value is not None]

        self.analyzer.analyze = tolerant
        # The schema-13 defense kernel applies the same strict 13/14-tile
        # invariant per candidate; fall back to zero-valued defense features
        # for legal edge hands the kernel cannot represent.
        if not getattr(_decision_module, "_codex_tolerant_defense", False):
            _decision_module._codex_tolerant_defense = True
            original_defense = _decision_module._defense_feature_batch

            def tolerant_defense(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
                try:
                    return original_defense(*args, **kwargs)
                except ValueError as exc:
                    if not any(
                        marker in str(exc)
                        for marker in ("expected 13 or 14", "must describe a normalized")
                    ):
                        raise
                    n = len(args[1])
                    zeros_u8 = np.zeros((n, 10), dtype=np.uint8)
                    zeros_f32 = np.zeros((n, 8), dtype=np.float32)
                    return zeros_u8, zeros_u8, zeros_f32

            _decision_module._defense_feature_batch = tolerant_defense

    def step(self, active: set[int]) -> tuple[np.ndarray, np.ndarray, int]:
        import time as _time

        trace_env = int(os.environ.get("ROLLOUT_TRACE_ENV", "-1"))
        _t0 = _time.perf_counter()
        decisions = active_decisions(self.observations, active)
        _t1 = _time.perf_counter()
        actions_by_env: list[dict[int, Any]] = [{} for _ in range(self.n)]
        if decisions:
            try:
                analysis = DecisionAnalysisBatch.build(
                    decisions,
                    analyzer=self.analyzer,
                    public=self.public,
                )
            except Exception as exc:
                details = [
                    f"{self.env_labels[d.env_index]}-seat{d.seat_id} "
                    f"hand={[int(t) for t in d.observation.hands[d.seat_id]]}"
                    for d in decisions
                ]
                raise RuntimeError(
                    f"decision analysis failed: {exc}\n"
                    f"batch rows ({len(decisions)}):\n" + "\n".join(details)
                ) from exc
            _t2 = _time.perf_counter()
            prepared = self.adapter.prepare(self.bridge, decisions, analysis)
            _t3 = _time.perf_counter()
            logits = self.adapter.masked_logits(prepared)
            _t4 = _time.perf_counter()
            action_ids = logits.argmax(-1).tolist()
            actions = self.bridge.decode(decisions, action_ids)
            _t5 = _time.perf_counter()
            for decision, action_id_value, action in zip(
                decisions, action_ids, actions, strict=True
            ):
                # The state machine decodes a tsumogiri discard by tile type;
                # with duplicate tiles in hand, ``select_action_from_mjai`` may
                # return the wrong physical copy and desync the env's turn
                # bookkeeping.  Force tsumogiri discards onto the drawn tile.
                drawn = getattr(decision.observation, "drawn_tile", None)
                if (
                    drawn is not None
                    and int(drawn) >= 0
                    and action_id_value % 2 == 0  # discard, tsumogiri mode
                    and action_kind(action) == "dahai"
                    and getattr(action, "tile", None) is not None
                    and int(action.tile) != int(drawn)
                ):
                    preferred = next(
                        (
                            candidate
                            for candidate in decision.observation.legal_actions()
                            if getattr(candidate, "tile", None) is not None
                            and int(candidate.tile) == int(drawn)
                            and action_kind(candidate) == "dahai"
                        ),
                        None,
                    )
                    if preferred is not None:
                        action = preferred
                actions_by_env[decision.env_index][decision.seat_id] = action
            if trace_env >= 0 and any(d.env_index == trace_env for d in decisions):
                env_index = trace_env
                print(
                    f"[trace] env={env_index} label={self.env_labels[env_index]} "
                    f"pre_state=cur{self.envs[env_index].current_player} "
                    f"drawn{self.envs[env_index].drawn_tile} "
                    f"turn{self.envs[env_index].turn_count} "
                    f"hands={[[int(t) for t in h] for h in self.envs[env_index].hands]} "
                    f"decisions={[(d.seat_id, action_id(a, d.observation)) for d, a in zip(decisions, actions) if d.env_index == trace_env]} "
                    f"obs_actors={[p for p, o in self.observations[env_index].items() if o.legal_actions()]}",
                    flush=True,
                )
        for env, row in zip(self.envs, actions_by_env, strict=True):
            if row:
                env_index = self.envs.index(env)
                pre_obs = self.observations[env_index]
                for seat_id, action in row.items():
                    if int(seat_id) not in pre_obs:
                        raise RuntimeError(
                            f"stale decision: label={self.env_labels[env_index]} "
                            f"seat={seat_id} not in pre-step observations "
                            f"(obs seats={list(pre_obs)})"
                        )
                    legal_ids = [
                        action_id(a, pre_obs[int(seat_id)])
                        for a in pre_obs[int(seat_id)].legal_actions()
                    ]
                    if action_id(action, pre_obs[int(seat_id)]) not in legal_ids:
                        raise RuntimeError(
                            f"stale decision: label={self.env_labels[env_index]} "
                            f"seat={seat_id} action={action.to_mjai()} not legal in "
                            f"pre-step observation (legal={legal_ids})"
                        )
                result = env.step(row)
                if not result:
                    env_index = self.envs.index(env)
                    terminal = bool(
                        env.done()
                        or getattr(env, "round_end_scores", None) is not None
                        or bool(getattr(env, "needs_initialize_next_round", False))
                    )
                    if terminal:
                        # The env's own tenpai evaluator can reject a legal
                        # kan-heavy hand during exhaustive-draw settlement and
                        # return an empty observation dict even though the
                        # kyoku/game actually terminated correctly.  The bridge
                        # observes the boundary events and removes the env.
                        continue
                    obs_now = env.get_observations()
                    legal_preview = {
                        int(seat): [
                            action_id(a, obs_now[int(seat)])
                            for a in obs_now[int(seat)].legal_actions()
                        ]
                        for seat in row
                        if int(seat) in obs_now
                    }
                    failed_seat = int(next(iter(row)))
                    raise RuntimeError(
                        f"RiichiEnv.step failed silently: label={self.env_labels[env_index]} "
                        f"env_index={env_index} "
                        f"actions={[(int(seat), action.to_mjai()) for seat, action in row.items()]} "
                        f"env_legal_ids={legal_preview} "
                        f"hands={[[int(t) for t in hand] for hand in env.hands]} "
                        f"failed_seat_hand_len={len(env.hands[failed_seat])} "
                        f"phase={env.phase} current={env.current_player} "
                        f"drawn={env.drawn_tile} needs_tsumo={env.needs_tsumo} "
                        f"last_discard={env.last_discard} turn={env.turn_count} "
                        f"wall_len={len(env.wall)} dora={[int(t) for t in env.dora_indicators]} "
                        f"melds={[[str(m.tiles) for m in rows] for rows in env.melds]} "
                        f"round_end_scores={getattr(env, 'round_end_scores', None)} "
                        f"needs_init_next={getattr(env, 'needs_initialize_next_round', None)} "
                        f"done={env.done()}"
                    )
        _t6 = _time.perf_counter()
        self.observations = [env.get_observations() for env in self.envs]
        end_kyoku, end_game = self.bridge.sync(self.observations)
        _t7 = _time.perf_counter()
        self.public.update(self.bridge.last_events)
        _t8 = _time.perf_counter()
        if os.environ.get("ROLLOUT_PROFILE"):
            print(
                f"[profile] n_decisions={len(decisions)} "
                f"decide={_t1 - _t0:.3f}s analysis={_t2 - _t1:.3f}s "
                f"prepare={_t3 - _t2:.3f}s forward={_t4 - _t3:.3f}s "
                f"decode={_t5 - _t4:.3f}s env_step={_t6 - _t5:.3f}s "
                f"obs={_t7 - _t6:.3f}s sync={_t8 - _t7:.3f}s",
                flush=True,
            )
        return end_kyoku, end_game, len(decisions)


class GRPBatcher:
    def __init__(self, config: ExperimentConfig, device: torch.device) -> None:
        self.config = config
        self.model = RankPredictor.from_checkpoint(config.grp_checkpoint)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def evaluate(
        self,
        metas: Iterable[dict[str, Any]],
        end_scores_list: list[list[int]],
        seats: list[int],
    ) -> list[float]:
        rows = []
        for meta, end_scores, seat in zip(metas, end_scores_list, seats, strict=True):
            rows.append(
                grp_features_from_scores(
                    meta["kyoku_initial_scores"],
                    end_scores,
                    chang=int(meta["round_wind"]),
                    ju=int(meta["kyoku_index"]),
                    ben=int(meta["honba"]),
                    liqibang=int(meta["riichi_sticks"]),
                    player=int(seat),
                )
            )
        features = np.stack(rows)
        probs = self.model.predict_rank_probs(features, device=self.device)
        return [reward_from_rank_probs(p, self.config.grp_pts_weight) for p in probs]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def _delta_stats(base: list[float], challenger: list[float]) -> dict[str, float]:
    if not base or not challenger:
        return {"mean": float("nan"), "std": float("nan"), "se": float("nan"), "n": 0}
    diffs = np.asarray(challenger, dtype=float) - np.asarray(base, dtype=float)
    n = len(diffs)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "se": float("nan"), "n": 0}
    mean = float(diffs.mean())
    std = float(diffs.std(ddof=1) if n > 1 else 0.0)
    se = std / math.sqrt(n)
    return {"mean": mean, "std": std, "se": se, "n": n}


def _verdict(
    means: list[float],
    deltas: dict[str, dict[str, float]],
    z: float,
) -> dict[str, Any]:
    """Classify the teacher's verdict for one decision at confidence z."""
    finite_indices = [index for index, value in enumerate(means) if math.isfinite(value)]
    if not finite_indices:
        return {
            "best_candidate": 1,
            "verdict": "uncertain",
            "challenger_better_count": 0,
            "challenger_worse_count": 0,
            "determined": False,
        }
    best_index = max(finite_indices, key=lambda index: means[index])
    best_candidate = best_index + 1
    names = ["delta_ba", "delta_ca", "delta_da"]
    challenger_better = [
        (idx + 2, deltas[name])
        for idx, name in enumerate(names)
        if deltas[name]["mean"] > z * deltas[name]["se"]
    ]
    challenger_worse = [
        (idx + 2, deltas[name])
        for idx, name in enumerate(names)
        if deltas[name]["mean"] < -z * deltas[name]["se"]
    ]
    if best_candidate == 1 and len(challenger_better) == 0:
        verdict = "keep_top1"
    elif best_candidate > 1 and len(challenger_better) == 1 and best_candidate == challenger_better[0][0]:
        verdict = f"override_top{best_candidate}"
    else:
        verdict = "uncertain"
    return {
        "best_candidate": best_candidate,
        "verdict": verdict,
        "challenger_better_count": len(challenger_better),
        "challenger_worse_count": len(challenger_worse),
        "determined": len(challenger_better) + len(challenger_worse) == 3,
    }


def _best_finite_rank(means: list[float]) -> int:
    finite = [index for index, value in enumerate(means) if math.isfinite(value)]
    if not finite:
        return 1
    return max(finite, key=lambda index: means[index]) + 1


def decision_id(row: dict[str, Any]) -> str:
    return (
        f"{row['year']}-{row['game_id']}-{int(row['kyoku_index']):02d}"
        f"-s{int(row['seat'])}-d{int(row['decision_index'])}"
    )


def _stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def evaluate_decision(
    row: dict[str, Any],
    store: RawKyokuStore,
    player_adapter: Any,
    grp: GRPBatcher,
    config: ExperimentConfig,
    *,
    progress: dict[str, Any],
) -> dict[str, Any]:
    sid = decision_id(row)
    seat = int(row["seat"])
    content = store.read(str(row["game_id"]), int(row["kyoku_index"]))
    env, obs, _kyoku_initial, meta = replay_to_decision(
        content,
        seat=seat,
        decision_index=int(row["decision_index"]),
    )
    events = [json.loads(line) for line in content.splitlines()]
    legal_actions = obs.legal_actions()
    candidate_ids = [
        int(row["top1_action"]),
        int(row["top2_action"]),
        int(row["top3_action"]),
        int(row["top4_action"]),
    ]
    forced: list[tuple[int, Any]] = []
    skipped_candidates: list[int] = []
    for rank, candidate in enumerate(candidate_ids, start=1):
        if candidate < 0:
            skipped_candidates.append(rank)
            continue
        found = next(
            (
                action
                for action in legal_actions
                if action_id(action, obs) == candidate
            ),
            None,
        )
        if found is None:
            # The training-time legal mask can contain kuikae-forbidden
            # discards that RiichiEnv correctly rejects; those candidates are
            # not forceable in a legal world and are skipped.
            skipped_candidates.append(rank)
            continue
        forced.append((rank, found))
    if not forced:
        raise RuntimeError(
            f"{sid}: no forceable candidate among {candidate_ids} "
            f"(legal ids: {[action_id(a, obs) for a in legal_actions]})"
        )
    forced_ranks = [rank for rank, _action in forced]

    is_stability_subset = (
        int(progress["stability_assigned"]) < int(config.stability_subset_count)
    )
    if is_stability_subset:
        progress["stability_assigned"] += 1
    max_worlds = (
        int(config.max_worlds_stability_subset)
        if is_stability_subset
        else int(config.max_worlds)
    )

    sid_hash = _stable_int(sid)
    values: dict[int, list[float]] = {rank: [] for rank in (1, 2, 3, 4)}
    candidate_value_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    n_worlds = 0
    target_n = int(config.min_worlds)
    verdict95 = verdict80 = None
    while n_worlds < max_worlds:
        wave = min(target_n - n_worlds, max_worlds - n_worlds)
        world_indices = list(range(n_worlds, n_worlds + wave))
        envs: list[RiichiEnv] = []
        branch_meta: list[tuple[int, int]] = []  # (world_idx, candidate_rank)
        for world_idx in world_indices:
            world_seed = (
                int(config.seed_base) * 7919
                + sid_hash * 104729
                + world_idx * 15485863
            ) % (2**31)
            world, _rewritten_events = sample_world(
                env,
                seat,
                np.random.default_rng(world_seed),
                events,
                decision_index=int(row["decision_index"]),
                decision_event_index=int(meta["decision_event_index"]),
            )
            for rank, candidate_action in forced:
                branch = world.clone()
                branch.step({seat: candidate_action})
                envs.append(branch)
                branch_meta.append((world_idx, rank))
        player = BatchPlayer(
            player_adapter,
            envs,
            env_labels=[
                f"w{world_idx}-r{rank}"
                for world_idx, rank in branch_meta
            ],
            analysis_cache_capacity=config.analysis_cache_capacity,
        )
        active = set(range(len(envs)))
        end_scores: dict[int, list[int]] = {}
        end_fields: dict[int, tuple[int, int, int, int]] = {}
        wave_steps = 0
        max_wave_steps = 1000
        while active:
            _end_kyoku, _end_game, decisions_this_step = player.step(active)
            wave_steps += 1
            if wave_steps > max_wave_steps:
                raise RuntimeError(
                    f"{sid}: wave did not end within {max_wave_steps} steps "
                    f"(active={len(active)})"
                )
            progress["policy_decisions"] += decisions_this_step
            for env_index in list(active):
                if bool(_end_kyoku[env_index]):
                    end_scores[env_index] = list(envs[env_index].scores())
                    end_fields[env_index] = (
                        envs[env_index].honba,
                        envs[env_index].riichi_sticks,
                        envs[env_index].round_wind,
                        envs[env_index].kyoku_idx,
                    )
                    active.remove(env_index)
        # GRP for finished branches.
        metas = []
        seats = []
        scores = []
        finished: list[tuple[int, int]] = []
        for env_index, (world_idx, rank) in enumerate(branch_meta):
            if env_index not in end_scores:
                raise RuntimeError(f"{sid}: branch {env_index} did not finish kyoku")
            honba, sticks, round_wind, kyoku_idx = end_fields[env_index]
            branch_meta_row = dict(meta)
            branch_meta_row.update(
                {"honba": honba, "riichi_sticks": sticks,
                 "round_wind": round_wind, "kyoku_index": kyoku_idx}
            )
            metas.append(branch_meta_row)
            seats.append(seat)
            scores.append(end_scores[env_index])
            finished.append((world_idx, rank))
        values_wave = grp.evaluate(metas, scores, seats)
        for (world_idx, rank), value in zip(finished, values_wave, strict=True):
            values[rank].append(float(value))
            candidate_value_rows.append({
                "decision_id": sid,
                "world_idx": world_idx,
                "candidate_rank": rank,
                "action_id": candidate_ids[rank - 1],
                "grp_value": float(value),
            })
        # Paired difference vs Policy Top1 (world-matched).
        for candidate_row in candidate_value_rows:
            if candidate_row["candidate_rank"] == 1:
                continue
            base = next(
                (
                    candidate
                    for candidate in candidate_value_rows
                    if candidate["world_idx"] == candidate_row["world_idx"]
                    and candidate["candidate_rank"] == 1
                ),
                None,
            )
            if base is not None:
                candidate_row["delta_vs_top1"] = float(
                    candidate_row["grp_value"] - base["grp_value"]
                )
        n_worlds += wave

        # Cumulative stats + stability row.
        means = [_mean_std(values[rank])[0] for rank in (1, 2, 3, 4)]
        deltas = {
            name: _delta_stats(values[1], values[rank])
            for name, rank in (("delta_ba", 2), ("delta_ca", 3), ("delta_da", 4))
        }
        stability_rows.append({
            "decision_id": sid,
            "n_worlds": n_worlds,
            "best_candidate": _best_finite_rank(means),
            "mean_a": means[0],
            "mean_b": means[1],
            "mean_c": means[2],
            "mean_d": means[3],
            "delta_ba_mean": deltas["delta_ba"]["mean"],
            "delta_ba_se": deltas["delta_ba"]["se"],
            "delta_ca_mean": deltas["delta_ca"]["mean"],
            "delta_ca_se": deltas["delta_ca"]["se"],
            "delta_da_mean": deltas["delta_da"]["mean"],
            "delta_da_se": deltas["delta_da"]["se"],
        })
        verdict95 = _verdict(means, deltas, float(config.z_determined))
        verdict80 = _verdict(means, deltas, float(config.z_secondary))
        progress["rollouts"] += len(envs)
        progress["waves"] += 1
        print(
            f"[rollout] {sid} n={n_worlds} verdict95={verdict95['verdict']} "
            f"best={verdict95['best_candidate']} "
            f"db={deltas['delta_ba']['mean']:.3f}±{deltas['delta_ba']['se']:.3f} "
            f"elapsed={time.perf_counter() - progress['started']:.1f}s",
            flush=True,
        )
        if verdict95["determined"] or n_worlds >= max_worlds:
            break
        target_n += int(config.world_increment)

    means = [_mean_std(values[rank])[0] for rank in (1, 2, 3, 4)]
    stds = [_mean_std(values[rank])[1] for rank in (1, 2, 3, 4)]
    ses = [
        stds[rank - 1] / math.sqrt(len(values[rank]))
        if values[rank]
        else float("nan")
        for rank in (1, 2, 3, 4)
    ]
    deltas = {
        name: _delta_stats(values[1], values[rank])
        for name, rank in (("delta_ba", 2), ("delta_ca", 3), ("delta_da", 4))
    }
    summary_row = {
        "decision_id": sid,
        **{key: row[key] for key in (
            "year", "game_id", "kyoku_index", "seat", "decision_index",
            "expert_action", "top1_action", "top2_action", "top3_action",
            "top4_action", "pi1", "pi2", "pi3", "pi4", "gap12", "gap13",
            "gap14", "log_gap12", "policy_entropy", "top4_cum_prob",
            "expert_is_top1", "expert_in_top4", "gap_bucket",
        )},
        "n_worlds": n_worlds,
        "mean_a": means[0],
        "mean_b": means[1],
        "mean_c": means[2],
        "mean_d": means[3],
        "std_a": stds[0],
        "std_b": stds[1],
        "std_c": stds[2],
        "std_d": stds[3],
        "se_a": ses[0],
        "se_b": ses[1],
        "se_c": ses[2],
        "se_d": ses[3],
    }
    for name, key in (
        ("delta_ba", "ba"),
        ("delta_ca", "ca"),
        ("delta_da", "da"),
    ):
        stats = deltas[name]
        summary_row[f"delta_{key}_mean"] = stats["mean"]
        summary_row[f"delta_{key}_std"] = stats["std"]
        summary_row[f"delta_{key}_se"] = stats["se"]
        summary_row[f"delta_{key}_ci95_lo"] = stats["mean"] - 1.96 * stats["se"]
        summary_row[f"delta_{key}_ci95_hi"] = stats["mean"] + 1.96 * stats["se"]
    if verdict95 is not None:
        summary_row.update({
            "teacher_best": verdict95["best_candidate"],
            "verdict95": verdict95["verdict"],
            "verdict80": verdict80["verdict"],
            "determined95": verdict95["determined"],
            "determined80": verdict80["determined"],
        })
    else:
        summary_row.update({
            "teacher_best": _best_finite_rank(means),
            "verdict95": "uncertain",
            "verdict80": "uncertain",
            "determined95": False,
            "determined80": False,
        })
    return {
        "summary": summary_row,
        "stability": stability_rows,
        "candidate_values": candidate_value_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="debug: only N decisions")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip decisions already present in this shard's summary CSV",
    )
    parser.add_argument(
        "--only-ids", default=None,
        help="comma-separated decision ids to process (after sharding)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="append to existing shard outputs instead of overwriting them",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir or (Path(__file__).resolve().parent / "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig()
    if args.config:
        overrides = json.loads(Path(args.config).read_text())
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    _seed_rng(int(config.seed_base) + int(args.shard_id))
    (out_dir / "rollout_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False)
    )

    selected = _read_csv(out_dir / "selected_decisions.csv")
    selected = [
        row for index, row in enumerate(selected)
        if index % int(args.num_shards) == int(args.shard_id)
    ]
    if args.resume:
        summary_path = out_dir / f"decision_summary_shard{args.shard_id}.csv"
        if summary_path.exists():
            done = {row["decision_id"] for row in _read_csv(summary_path)}
            skipped = sum(1 for row in selected if decision_id(row) in done)
            selected = [row for row in selected if decision_id(row) not in done]
            print(f"[rollout] resume: skipped {skipped} already-completed decisions")
    if args.limit:
        selected = selected[: int(args.limit)]
    if args.only_ids:
        allowed = set(args.only_ids.split(","))
        selected = [row for row in selected if decision_id(row) in allowed]
        print(f"[rollout] only-ids filter: {len(selected)} decisions")

    summaries: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    candidate_value_rows: list[dict[str, Any]] = []
    if args.append:
        summary_path = out_dir / f"decision_summary_shard{args.shard_id}.csv"
        if summary_path.exists():
            summaries.extend(_read_csv(summary_path))
        stability_path = out_dir / f"stability_shard{args.shard_id}.csv"
        if stability_path.exists():
            stability_rows.extend(_read_csv(stability_path))
        candidate_path = out_dir / f"candidate_values_shard{args.shard_id}.csv"
        if candidate_path.exists():
            candidate_value_rows.extend(_read_csv(candidate_path))
    print(f"[rollout] shard={args.shard_id} decisions={len(selected)}")

    device = torch.device(args.device)
    adapter = load_policy_adapter(
        str(REPO_ROOT / config.continuation_policy), device=device
    )
    grp = GRPBatcher(config, device)
    store = RawKyokuStore(
        REPO_ROOT / config.raw_index,
        REPO_ROOT / config.raw_validation_dir,
    )
    store.load_needed({str(row["game_id"]) for row in selected})
    print(f"[rollout] raw kyoku locations loaded: {len(store.locations)}")

    progress: dict[str, Any] = {
        "started": time.perf_counter(),
        "policy_decisions": 0,
        "rollouts": 0,
        "waves": 0,
        "stability_assigned": 0,
    }
    for index, row in enumerate(selected):
        result = evaluate_decision(
            row,
            store,
            adapter,
            grp,
            config,
            progress=progress,
        )
        summaries.append(result["summary"])
        stability_rows.extend(result["stability"])
        candidate_value_rows.extend(result["candidate_values"])
        if (index + 1) % 5 == 0 or index + 1 == len(selected):
            _write_csv(out_dir / f"decision_summary_shard{args.shard_id}.csv", summaries)
            _write_csv(out_dir / f"stability_shard{args.shard_id}.csv", stability_rows)
            _write_csv(
                out_dir / f"candidate_values_shard{args.shard_id}.csv",
                candidate_value_rows,
            )
            elapsed = time.perf_counter() - progress["started"]
            print(
                f"[rollout] shard={args.shard_id} progress={index + 1}/{len(selected)} "
                f"elapsed={elapsed:.1f}s decisions/s={progress['policy_decisions'] / elapsed:.1f} "
                f"rollouts={progress['rollouts']}",
                flush=True,
            )
    store.close()
    _write_csv(out_dir / f"decision_summary_shard{args.shard_id}.csv", summaries)
    _write_csv(out_dir / f"stability_shard{args.shard_id}.csv", stability_rows)
    _write_csv(
        out_dir / f"candidate_values_shard{args.shard_id}.csv",
        candidate_value_rows,
    )
    elapsed = time.perf_counter() - progress["started"]
    summary = {
        "shard_id": args.shard_id,
        "decisions": len(selected),
        "policy_decisions": progress["policy_decisions"],
        "rollouts": progress["rollouts"],
        "waves": progress["waves"],
        "elapsed_s": round(elapsed, 3),
        "decisions_per_s": round(progress["policy_decisions"] / elapsed, 2),
        "rollouts_per_s": round(progress["rollouts"] / elapsed, 2),
    }
    (out_dir / f"rollout_summary_shard{args.shard_id}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"[rollout] shard={args.shard_id} done:", json.dumps(summary))


if __name__ == "__main__":
    main()
