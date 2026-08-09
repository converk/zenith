"""Streaming GRP dataset from the local tenhou MJAI log zips."""

from __future__ import annotations

import gzip
import hashlib
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from riichienv import MjaiReplay

from .model import NUM_PLAYERS, grp_features_from_scores


def _score_ranks(scores: Sequence[int]) -> list[int]:
    """Rank seats by score desc, breaking ties by lower seat index (0=best)."""
    order = sorted(
        range(len(scores)),
        key=lambda seat: (-int(scores[seat]), int(seat)),
    )
    ranks = [0] * len(scores)
    for position, seat in enumerate(order):
        ranks[seat] = position
    return ranks


def _sample_selected(name: str, denominator: int, remainders: Sequence[int]) -> bool:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % max(1, int(denominator)) in set(int(x) for x in remainders)


def iter_grp_rows(
    zip_paths: Iterable[str | Path],
    *,
    subset_denominator: int = 5,
    subset_remainders: Sequence[int] = (0,),
    max_games: int | None = None,
) -> Iterator[tuple[np.ndarray, int, float]]:
    """Yield ``(features, final_rank, point_delta)`` for every player seat.

    Sampling is deterministic per game filename, so reruns with the same
    denominator/remainders produce the same subset. ``point_delta`` is the
    raw kyoku point delta in thousands (the existing terminal reward before
    clipping), used for the GRP vs point-delta correlation check.
    """
    sampled = 0
    for path in zip_paths:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not _sample_selected(name, subset_denominator, subset_remainders):
                    continue
                if max_games is not None and sampled >= max_games:
                    return
                sampled += 1
                raw = archive.read(name)
                if raw.startswith(b"\x1f\x8b"):
                    content = gzip.decompress(raw).decode("utf-8")
                else:
                    content = raw.decode("utf-8")
                replay = MjaiReplay.from_jsonl_string(content)
                kyokus = list(replay.take_kyokus())
                if not kyokus:
                    continue
                # MJAI logs carry no ``end_game`` scores, so the last kyoku's
                # end scores are the hanchan's final scores. This is the label
                # the plan requires: each kyoku predicts the final hanchan rank.
                final_scores = kyokus[-1].take_grp_features()["round_end_scores"]
                final_ranks = _score_ranks(final_scores)
                for kyoku in kyokus:
                    features = kyoku.take_grp_features()
                    initial = features["round_initial_scores"]
                    end = features["round_end_scores"]
                    delta = features["round_delta_scores"]
                    chang = int(features["chang"])
                    ju = int(features["ju"])
                    ben = int(features["ben"])
                    liqibang = int(features["liqibang"])
                    for player in range(NUM_PLAYERS):
                        row = grp_features_from_scores(
                            initial,
                            end,
                            chang=chang,
                            ju=ju,
                            ben=ben,
                            liqibang=liqibang,
                            player=player,
                        )
                        yield row, int(final_ranks[player]), float(delta[player]) / 1000.0
