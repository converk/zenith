"""V19 信念五头监督标签的 Python 边界。

标签由 RiichiEnv Rust 侧上帝视角批量生成(D26),Python 只做形状规范化与
存储打包。**标签只进训练,不进推理**——模型前向的信念 token 是网络自身输出,
与标签无关。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import riichienv

# 五头固定形状(3 家对手;Wait 含 1 位 N/A)。
HAND_LEN = 3 * 34
SHANTEN_LEN = 3
WAIT_LEN = 3 * 35
DANGER_LEN = 3 * 34
LOSS_LEN = 3 * 34


@dataclass(frozen=True)
class BeliefLabelBatch:
    """一批决策的五头标签(逐观测 = 一家视角,三家对手)。"""

    hand_counts: np.ndarray  # [B,102] uint8
    shanten: np.ndarray  # [B,3] uint8,0..8
    wait: np.ndarray  # [B,105] uint8,0/1
    danger: np.ndarray  # [B,102] uint8,0/1
    loss: np.ndarray  # [B,102] float32,原始点数;训练侧再归一化

    @property
    def batch_size(self) -> int:
        return int(self.hand_counts.shape[0])


def encode_belief_labels_batch(observations: list[object]) -> BeliefLabelBatch:
    """从批量 Observations(RiichiEnv/RiichiLab 同构)生成五头标签。"""
    if not observations:
        raise ValueError("cannot encode an empty belief-label batch")
    native = [getattr(obs, "native_observation", obs) for obs in observations]
    encoded = riichienv.prepare_belief_labels_batch(native)
    return BeliefLabelBatch(
        hand_counts=np.asarray(encoded.hand_counts, dtype=np.uint8).reshape(-1, HAND_LEN),
        shanten=np.asarray(encoded.shanten, dtype=np.uint8).reshape(-1, SHANTEN_LEN),
        wait=np.asarray(encoded.wait, dtype=np.uint8).reshape(-1, WAIT_LEN),
        danger=np.asarray(encoded.danger, dtype=np.uint8).reshape(-1, DANGER_LEN),
        loss=np.asarray(encoded.loss, dtype=np.float32).reshape(-1, LOSS_LEN),
    )
