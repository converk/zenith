"""PPO rollout SoA(Structure-of-Arrays)紧凑缓冲:一次性物化 + 跨 epoch 复用。

旧路径在 ``learner.py`` 的每个 epoch、每个 minibatch 都对 ``list[Transition]``
重新取长度、重新分配 padded tensor、并逐样本复制。本模块把整个 rollout 一次性
物化成扁平 SoA 数组(变长字段用 offset 索引),并提供一个向量化的 ``collate``
把某个 minibatch 的 index 数组 gather 成 V16 padded host 张量。

约定与 ``learner.materialize_host_batch`` 保持完全一致(输出字段、dtype、shape),
因此本模块可作为一个安全的替代路径;旧路径保留为 semantic oracle/debug fallback。

边界说明:
- 所有变长字段都存成「flat 数组 + offset(长度 N+1)」;collate 用一次性 gather
  替代逐 transition 的 ``torch.tensor`` 拷贝。
- ``legal_mask`` 固定为 241 维,不涉及 padding。
- ``critic_factors`` 在长度为 0 时为空数组;collate 处理 max_critic==0 情形。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..model.schema import NUM_ACTIONS
from .trajectory import Transition, transition_sequence_length

# V16 各段的固定通道宽(与 model/action_query.py / snapshot.py / critic_features.py 一致)。
_HISTORY_W = 10
_NUMERIC_W = 8
_SNAPSHOT_CAT_W = 4
_SNAPSHOT_NUM_W = 7
_QUERY_W = 15
_CRITIC_W = 10


def _gather_padded(
    flat: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    max_len: int,
    default: int | float,
    *,
    width: int | None = None,
) -> np.ndarray:
    """把 flat 数组按 ``(starts, lengths)`` gather 成 ``[B, max_len(, width)]``。

    向量化版本,避免逐 transition 循环。``flat`` 是按行拼接的连续存储,
    ``starts``/``lengths`` 指向每行在 flat 中的起点与长度;超出长度的位置填
    ``default``。``width`` 为 None 时返回 2D,否则返回 3D。
    """
    batch = int(len(lengths))
    if max_len == 0:
        if width is None:
            return np.zeros((batch, 0), dtype=flat.dtype)
        return np.zeros((batch, 0, width), dtype=flat.dtype)
    positions = np.arange(max_len, dtype=np.int64)[None, :] + starts[:, None]
    valid = positions < (starts + lengths)[:, None]
    safe = np.where(valid, positions, 0)
    if width is None:
        out = flat[safe]
        out = np.where(valid, out, default)
    else:
        out = flat[safe]
        out = np.where(valid[:, :, None], out, default)
    return out


class RolloutBuffer:
    """一次物化整个 rollout 的 SoA 缓冲。

    只保留训练所需字段;reward/done/advantage 等标量也一并抽取,供 learner
    复用(不再遍历 ``Transition`` 对象)。构造时只做一次 O(N) 的数组拼接与
    标量抽取,之后的 ``collate`` 都是纯 numpy gather + 一次 torch 转换。
    """

    def __init__(self, transitions: Sequence[Transition]) -> None:
        count = len(transitions)
        self.size = count

        self.history_lengths = np.array(
            [int(item.history_length) for item in transitions], dtype=np.int64
        )
        self.snapshot_lengths = np.array(
            [int(item.snapshot_length) for item in transitions], dtype=np.int64
        )
        self.query_pair_counts = np.array(
            [int(item.query_pair_counts) for item in transitions], dtype=np.int64
        )
        self.critic_lengths = np.array(
            [int(item.critic_length) for item in transitions], dtype=np.int64
        )
        self.sequence_lengths = np.array(
            [transition_sequence_length(item) for item in transitions], dtype=np.int64
        )
        self.actions = np.array([int(item.action) for item in transitions], dtype=np.int64)
        self.old_logprobs = np.array(
            [float(item.logprob) for item in transitions], dtype=np.float32
        )
        self.values = np.array([float(item.value) for item in transitions], dtype=np.float32)
        self.rewards = np.array([float(item.reward) for item in transitions], dtype=np.float32)
        self.done = np.array([bool(item.done) for item in transitions], dtype=np.bool_)
        self.advantages = np.array(
            [float(item.advantage) for item in transitions], dtype=np.float32
        )
        self.legal_mask = np.stack([item.legal_mask for item in transitions], axis=0).astype(
            np.bool_
        )

        # ---- 可变长字段:flat + offset ----
        self.history_offsets, self.history_factors, self.history_numeric = self._concat_var(
            transitions,
            lambda item: item.history_factors.astype(np.uint8, copy=False),
        )
        (
            self.history_numeric_offsets,
            self.history_numeric_flat,
            _,
        ) = self._concat_var(
            transitions, lambda item: item.history_numeric.astype(np.float32, copy=False)
        )
        (
            self.snapshot_kinds_offsets,
            self.snapshot_kinds_flat,
            _,
        ) = self._concat_var(
            transitions, lambda item: item.snapshot_kinds.astype(np.uint8, copy=False)
        )
        (
            self.snapshot_cat_offsets,
            self.snapshot_cat_flat,
            _,
        ) = self._concat_var(
            transitions, lambda item: item.snapshot_cat.astype(np.uint8, copy=False)
        )
        (
            self.snapshot_num_offsets,
            self.snapshot_num_flat,
            _,
        ) = self._concat_var(
            transitions, lambda item: item.snapshot_num.astype(np.float32, copy=False)
        )
        (
            self.query_rows_offsets,
            self.query_rows_flat,
            _,
        ) = self._concat_var(
            transitions, lambda item: item.query_rows.astype(np.int32, copy=False)
        )
        (
            self.query_ids_offsets,
            self.query_ids_flat,
            _,
        ) = self._concat_var(
            transitions, lambda item: item.query_action_ids.astype(np.int32, copy=False)
        )

        # critic 是可选字段;长度为 0 时为空 array。
        critic_parts = []
        critic_offsets = []
        cursor = 0
        for item in transitions:
            critic_offsets.append(cursor)
            if int(item.critic_length) > 0:
                arr = np.asarray(item.critic_factors, dtype=np.uint8)
                critic_parts.append(arr)
                cursor += arr.shape[0]
        critic_offsets.append(cursor)
        self.critic_offsets = np.asarray(critic_offsets, dtype=np.int64)
        self.critic_factors_flat = (
            np.concatenate(critic_parts, axis=0) if critic_parts else np.zeros((0, _CRITIC_W), dtype=np.uint8)
        )

    @staticmethod
    def _concat_var(
        transitions: Sequence[Transition],
        extract,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按行拼接可变长数组,返回 (offsets[N+1], flat, per-row width)。

        ``extract`` 返回每个 transition 的 2D array(第一维为 token 数)。
        """
        parts = []
        offsets = []
        rows = 0
        for item in transitions:
            offsets.append(rows)
            arr = np.asarray(extract(item))
            parts.append(arr)
            rows += int(arr.shape[0])
        offsets.append(rows)
        flat = np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)
        return np.asarray(offsets, dtype=np.int64), flat, (flat.shape[1] if flat.ndim >= 2 else 0)

    def collate(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        """把一个 minibatch 的 index 数组 gather 成与 ``materialize_host_batch``
        一致的 V16 padded host 张量(CPU)。"""
        idx = np.asarray([int(index) for index in indices], dtype=np.int64)
        if len(idx) == 0:
            raise ValueError("cannot collate an empty minibatch")
        batch = int(len(idx))

        hist_lens = self.history_lengths[idx]
        snap_lens = self.snapshot_lengths[idx]
        pair_counts = self.query_pair_counts[idx]
        critic_lens = self.critic_lengths[idx]

        max_history = int(hist_lens.max(initial=0))
        max_snapshot = int(snap_lens.max(initial=0))
        max_pairs = int(pair_counts.max(initial=0))
        max_critic = int(critic_lens.max(initial=0))

        history_factors = _gather_padded(
            self.history_factors,
            self.history_offsets[idx],
            hist_lens,
            max_history,
            0,
            width=_HISTORY_W,
        )
        history_numeric = _gather_padded(
            self.history_numeric_flat,
            self.history_numeric_offsets[idx],
            hist_lens,
            max_history,
            0.0,
            width=_NUMERIC_W,
        )
        snapshot_kinds = _gather_padded(
            self.snapshot_kinds_flat,
            self.snapshot_kinds_offsets[idx],
            snap_lens,
            max_snapshot,
            0,
        )
        snapshot_cat = _gather_padded(
            self.snapshot_cat_flat,
            self.snapshot_cat_offsets[idx],
            snap_lens,
            max_snapshot,
            0,
            width=_SNAPSHOT_CAT_W,
        )
        snapshot_num = _gather_padded(
            self.snapshot_num_flat,
            self.snapshot_num_offsets[idx],
            snap_lens,
            max_snapshot,
            0.0,
            width=_SNAPSHOT_NUM_W,
        )
        query_rows = _gather_padded(
            self.query_rows_flat,
            self.query_rows_offsets[idx],
            2 * pair_counts,
            2 * max_pairs,
            0,
            width=_QUERY_W,
        )
        query_action_ids = _gather_padded(
            self.query_ids_flat,
            self.query_ids_offsets[idx],
            pair_counts,
            max_pairs,
            0,
        )
        if max_critic > 0:
            critic_factors = _gather_padded(
                self.critic_factors_flat,
                self.critic_offsets[idx],
                critic_lens,
                max_critic,
                0,
                width=_CRITIC_W,
            )
        else:
            critic_factors = np.zeros((batch, 0, _CRITIC_W), dtype=np.uint8)

        # 与 materialize_host_batch 一致:标量用 torch.empty/long,legal bool。
        history_lengths = torch.from_numpy(hist_lens)
        snapshot_lengths = torch.from_numpy(snap_lens)
        query_pair_counts = torch.from_numpy(pair_counts)
        critic_lengths = torch.from_numpy(critic_lens)
        actions = torch.from_numpy(self.actions[idx])
        old_logprobs = torch.from_numpy(self.old_logprobs[idx])
        advantages = torch.from_numpy(self.advantages[idx])
        legal = torch.from_numpy(self.legal_mask[idx])

        return {
            "history_factors": torch.from_numpy(np.ascontiguousarray(history_factors)),
            "history_numeric": torch.from_numpy(np.ascontiguousarray(history_numeric)),
            "history_lengths": history_lengths,
            "snapshot_kinds": torch.from_numpy(np.ascontiguousarray(snapshot_kinds)),
            "snapshot_cat": torch.from_numpy(np.ascontiguousarray(snapshot_cat)),
            "snapshot_num": torch.from_numpy(np.ascontiguousarray(snapshot_num)),
            "snapshot_lengths": snapshot_lengths,
            "query_rows": torch.from_numpy(np.ascontiguousarray(query_rows)),
            "query_action_ids": torch.from_numpy(np.ascontiguousarray(query_action_ids)),
            "query_pair_counts": query_pair_counts,
            "legal_mask": legal,
            "critic_factors": torch.from_numpy(np.ascontiguousarray(critic_factors)),
            "critic_lengths": critic_lengths,
            "actions": actions,
            "old_logprobs": old_logprobs,
            "advantages": advantages,
        }

    def bucketed_minibatches(self, minibatch_size: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, ...]:
        """与 ``learner.length_bucketed_minibatches`` 等价,但直接用已存的长度。"""
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        count = self.size
        if count == 0:
            raise ValueError("cannot bucket an empty rollout")
        lengths = self.sequence_lengths
        permutation = rng.permutation if rng is not None else np.random.permutation
        shuffled = permutation(count)
        sorted_indices = shuffled[np.argsort(lengths[shuffled], kind="stable")]
        batches = tuple(
            sorted_indices[start : start + minibatch_size]
            for start in range(0, count, minibatch_size)
        )
        batch_order = permutation(len(batches))
        return tuple(batches[index] for index in batch_order)
