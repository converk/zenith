"""V19 信念五头监督标签的 Python 边界（模糊化版）。

标签由 RiichiEnv Rust 侧上帝视角精确生成（D26），Python 在边界处把
「精确计数 / 精确待牌集」映射为**模糊语义标签**（用户 2026-09-07 决策）：

- Hand：每牌种 0..4 精确计数 → 花色×段位 16 组 × 计数桶 {0, 1, ≥2}；
  标签长度 [B, 48]，语义 = "对手手里大概有什么、有没有成对/刻子"。
- Wait：34 位精确待牌集 → 听牌 + 待牌宽度桶 {非听 / 1 面 / 2 面 /
  3-5 面 / ≥6 面}；标签长度 [B, 15]，语义 = "是否听牌 + 听牌质量"。

Shanten / Danger / Loss 保持精确语义（向听、逐牌可荣、逐牌反事实打点），
是信息量大、随机性低的头部。**标签只进训练，不进推理**——模型前向的信念
token 是网络自身输出，与标签无关。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import riichienv

# 三家对手（相对观察者：0=下家、1=对面、2=上家）。
BELIEF_PLAYERS = 3
# 花色×段位粗桶：万/筒/索 × {1-3,4-6,7-9} 共 9 组 + 字牌 7 组 = 16 组。
HAND_GROUPS = 16
# 每组计数桶：0 = 0 张, 1 = 1 张, 2 = ≥2 张。
HAND_BUCKETS = 3
HAND_LEN = BELIEF_PLAYERS * HAND_GROUPS  # 48
SHANTEN_LEN = BELIEF_PLAYERS  # 3
# 待牌宽度桶类别数：0=非听, 1=1面, 2=2面, 3=3-5面, 4=≥6面。
# 标签为每家一个类别索引，因此长度 = 玩家数 3（不是 3×5 的 one-hot）。
WAIT_CLASSES = 5
WAIT_LEN = BELIEF_PLAYERS  # 3
DANGER_LEN = BELIEF_PLAYERS * 34  # 102
LOSS_LEN = BELIEF_PLAYERS * 34  # 102

# 16 组的牌种下标区间（左闭右开），牌种序 = 万 0-8 / 筒 9-17 / 索 18-26 /
# 字牌 27-33（与 RiichiEnv 领域常量一致）。
HAND_GROUP_TILE_RANGES: tuple[tuple[int, int], ...] = (
    (0, 3), (3, 6), (6, 9),          # 万 1-3 / 4-6 / 7-9
    (9, 12), (12, 15), (15, 18),     # 筒 1-3 / 4-6 / 7-9
    (18, 21), (21, 24), (24, 27),    # 索 1-3 / 4-6 / 7-9
    (27, 28), (28, 29), (29, 30),    # 字牌 东/南/西
    (30, 31), (31, 32), (32, 33),    # 北/白/发
    (33, 34),                        # 中
)


@dataclass(frozen=True)
class BeliefLabelBatch:
    """一批决策的模糊五头标签（逐观测 = 一家视角，三家对手）。"""

    hand: np.ndarray  # [B,48] uint8：16 组 × 3 桶（扁平，玩家主序）
    shanten: np.ndarray  # [B,3] uint8, 0..8
    wait: np.ndarray  # [B,3] uint8：每家 1 个宽度桶类别索引
    danger: np.ndarray  # [B,102] uint8, 0/1
    loss: np.ndarray  # [B,102] float32, 原始点数;训练侧再归一化

    @property
    def batch_size(self) -> int:
        return int(self.hand.shape[0])


def _fuzzy_hand(exact_hand: np.ndarray) -> np.ndarray:
    """把 [B,3,34] 精确计数映射为 [B,3,16] 的 {0,1,≥2} 组计数桶。"""
    grouped = np.stack(
        [
            exact_hand[:, :, start:end].sum(axis=-1)
            for start, end in HAND_GROUP_TILE_RANGES
        ],
        axis=-1,
    )
    return np.clip(grouped, 0, 2).astype(np.uint8)


def _fuzzy_wait(exact_wait: np.ndarray) -> np.ndarray:
    """把 [B,3,35] 精确待牌集映射为 [B,3,5] 的听牌+宽度桶类别。

    第 35 位 N/A：1 = 非听；0 = 听牌。待牌宽度 = 34 位中置 1 的牌种数。
    """
    width = exact_wait[..., :34].sum(axis=-1)
    not_tenpai = exact_wait[..., 34] == 1
    cls = np.zeros(width.shape, dtype=np.uint8)
    cls[~not_tenpai & (width == 1)] = 1
    cls[~not_tenpai & (width == 2)] = 2
    cls[~not_tenpai & (width >= 3) & (width <= 5)] = 3
    cls[~not_tenpai & (width >= 6)] = 4
    return cls


def encode_belief_labels_batch(observations: list[object]) -> BeliefLabelBatch:
    """从批量 Observations(RiichiEnv/RiichiLab 同构)生成模糊五头标签。"""
    if not observations:
        raise ValueError("cannot encode an empty belief-label batch")
    native = [getattr(obs, "native_observation", obs) for obs in observations]
    encoded = riichienv.prepare_belief_labels_batch(native)
    exact_hand = np.asarray(encoded.hand_counts, dtype=np.uint8).reshape(-1, BELIEF_PLAYERS, 34)
    exact_wait = np.asarray(encoded.wait, dtype=np.uint8).reshape(-1, BELIEF_PLAYERS, 35)
    hand = _fuzzy_hand(exact_hand).reshape(-1, HAND_LEN)
    wait = _fuzzy_wait(exact_wait).reshape(-1, WAIT_LEN)
    return BeliefLabelBatch(
        hand=hand,
        shanten=np.asarray(encoded.shanten, dtype=np.uint8).reshape(-1, SHANTEN_LEN),
        wait=wait,
        danger=np.asarray(encoded.danger, dtype=np.uint8).reshape(-1, DANGER_LEN),
        loss=np.asarray(encoded.loss, dtype=np.float32).reshape(-1, LOSS_LEN),
    )
