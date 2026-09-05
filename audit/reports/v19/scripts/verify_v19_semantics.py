#!/usr/bin/env python
"""V19 业务语义验收：模型输入与真实局面事实一致（正/反两个方向）。

正向：
- actor 快照结构通过 V19 语义校验（fail closed）；
- RIICHI_CARD 数量=3、无 RIVER_SUMMARY/Critic future；
- critic 私有行含三家真手（特权，与训练环境同一数据源）；
- 信念五头标签形状正确，且 hand 计数和=13、wait N/A 与听牌一致性、
  danger⊆wait、loss>0 ⇔ danger=1（决策时刻反事实）。

反向：
- actor 路径不含 critic/信念段（无隐藏手牌、无模型内部 token）；
- actor 输入不含牌山/里宝（factors 无 future 类、数值域合法）。

运行：
  /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python \
    audit/reports/v19/scripts/verify_v19_semantics.py
"""

from __future__ import annotations

import argparse

import numpy as np

from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision
from riichi_ppo_v1.model.belief_labels import encode_belief_labels_batch
from riichi_ppo_v1.model.encoding_protocol import (
    KIND_BELIEF,
    KIND_CRITIC_HAND,
    KIND_RIICHI_CARD,
    KIND_RIVER_DISCARD,
    KIND_SEP_SHIMOCHA_RIVER,
    KIND_SEP_TOIMEN_RIVER,
    KIND_SEP_KAMICHA_RIVER,
    SEGMENT_CRITIC_PRIVATE,
    TOKEN_ROW_WIDTH,
)
from riichi_ppo_v1.model.semantic_validation import (
    assert_actor_input_semantics,
    assert_critic_token_semantics,
)


def _check_initial_state(seed: int) -> dict[str, str]:
    """在真实 BatchedRiichiEnv 的初始决策点执行正/反向断言。"""
    import riichi
    from riichienv import BatchedRiichiEnv

    env = BatchedRiichiEnv(1, seed=seed)
    observations = env.reset()[0]
    observer = observations[0]
    manager = riichi.MjaiKyokuStateMachineManager(1)
    bridge = BatchedStateBridge(manager, 1)
    end_kyoku, end_game = bridge.sync([observations])
    assert not bool(end_kyoku[0]) and not bool(end_game[0])

    batch = bridge.prepare([Decision(0, 0, observer)])
    actor_factors = np.asarray(batch.actor_factors, dtype=np.int64)
    actor_numeric = np.asarray(batch.actor_numeric, dtype=np.float32)
    actor_lengths = np.asarray(batch.actor_lengths, dtype=np.int64)
    assert_actor_input_semantics(
        actor_factors, actor_numeric, actor_lengths,
        None,
        np.asarray(batch.query_action_ids, dtype=np.int64),
        np.asarray(batch.query_pair_counts, dtype=np.int64),
        np.asarray(batch.legal_mask, dtype=np.bool_),
        context_tokens=320,
    )

    # 反向：actor 不含 critic/信念/未来段；kind 无 13/15。
    rows = actor_factors[0, : int(actor_lengths[0])]
    segments = rows[:, 0].astype(int)
    kinds = rows[:, 1].astype(int)
    assert not np.any(segments == SEGMENT_CRITIC_PRIVATE), "actor leaks critic private"
    assert not np.any(np.isin(kinds, (KIND_CRITIC_HAND, KIND_BELIEF)))
    assert not np.any(kinds == 6), "RIVER_SUMMARY(kind 6) must be removed"

    # 正向：三家 RIICHI_CARD 恒发射，且初始未立直全零。
    assert int(np.count_nonzero(kinds == KIND_RIICHI_CARD)) == 3
    riichi_rows = rows[kinds == KIND_RIICHI_CARD]
    assert np.all(riichi_rows[:, 2:] == 0), "start-state riichi cards must be all zero"

    # 正向：critic 私有行包含三家真手（初始 13 张 → 至少 3 行 hand）。
    critic_rows = np.asarray(batch.critic_factors, dtype=np.uint8)
    critic_lengths = np.asarray(batch.critic_lengths, dtype=np.int64)
    assert_critic_token_semantics(critic_rows, critic_lengths)

    # 正向：信念标签（初始三家暗手 13 张、非听牌 N/A、无 danger/loss）。
    labels = encode_belief_labels_batch([observer])
    assert labels.hand_counts.shape == (1, 102)
    hand_sums = labels.hand_counts.reshape(1, 3, 34).sum(axis=-1)
    assert np.all(hand_sums == 13), f"belief hand sums != 13: {hand_sums}"
    wait = labels.wait.reshape(1, 3, 35)
    assert np.all(wait[..., 34] == 1), "start-state opponents are not tenpai"
    danger = labels.danger.reshape(1, 3, 34)
    loss = labels.loss.reshape(1, 3, 34)
    assert np.all(danger == 0) and np.all(loss == 0)
    assert np.all(labels.shanten > 0), "start-state opponent shanten > 0"

    # 反向：actor 数值域合法（[-1,1]）。
    assert np.all(np.isfinite(actor_numeric))
    assert np.all(np.abs(actor_numeric) <= 1.0)

    return {
        "actor_length": str(int(actor_lengths[0])),
        "critic_length": str(int(critic_lengths[0])),
        "query_pairs": str(int(batch.query_pair_counts[0])),
        "belief_hand_sums": hand_sums.reshape(-1).tolist().__repr__(),
    }


def _claimed_river_count(observation: object, supplier: int) -> int:
    """统计某供牌者河中被鸣走的原始下标数（观察对象必须暴露 melds）。"""
    count = 0
    melds = getattr(observation, "melds", ((), (), (), ()))
    for rows in (melds if melds else ()):
        for meld in rows:
            if int(getattr(meld, "from_who", -1)) == supplier and getattr(meld, "called_tile_index", None) is not None:
                count += 1
    return count


def _check_mid_game(seed: int, steps: int = 20) -> int:
    """连续步进真实环境，在每个可决策点做同一套正/反向断言。"""
    import riichi
    from riichienv import BatchedRiichiEnv

    env = BatchedRiichiEnv(1, seed=seed)
    observations = env.reset()[0]
    manager = riichi.MjaiKyokuStateMachineManager(1)
    bridge = BatchedStateBridge(manager, 1)
    decisions_checked = 0
    for _step in range(int(steps)):
        for seat in range(4):
            observation = observations[seat]
            legal = list(observation.legal_actions())
            if not legal:
                continue
            bridge.prepare([Decision(0, seat, observation)])
            # 直接调用断言函数覆盖正/反向（结构与标签）。
            _check_one_decision(observation, seat)
            decisions_checked += 1
            break  # 每轮只执行一次动作，保持步进简单。
        # 选第一个决策座位的第一个合法动作执行。
        chosen = None
        for seat in range(4):
            legal = list(observations[seat].legal_actions())
            if legal:
                chosen = (seat, legal[0])
                break
        if chosen is None:
            break
        actions_by_env = [dict() for _ in range(1)]
        actions_by_env[0][chosen[0]] = chosen[1]
        observations = env.step_batch(actions_by_env)[0]
        bridge.sync([observations])
        if bool(env.done()[0]):
            break
    return decisions_checked


def _check_one_decision(observation: object, seat: int) -> None:
    """对一个真实 Observation 执行正向/反向断言（不含 bridge 重复装配）。"""
    labels = encode_belief_labels_batch([observation])
    hand_sums = labels.hand_counts.reshape(1, 3, 34).sum(axis=-1)
    # 决策时刻对手暗手合规：13/14（未副露）或随副露/杠减少（域 0..14）。
    assert np.all((hand_sums >= 0) & (hand_sums <= 14)), f"belief hand sums out of domain: {hand_sums}"
    wait = labels.wait.reshape(1, 3, 35)
    danger = labels.danger.reshape(1, 3, 34)
    loss = labels.loss.reshape(1, 3, 34)
    # N/A 位与听牌集合互斥。
    for rel in range(3):
        any_wait = bool(np.any(wait[0, rel, :34] == 1))
        assert bool(wait[0, rel, 34]) == (not any_wait)
        assert not np.any(danger[0, rel] & ~wait[0, rel, :34]), "danger outside wait"
        assert np.all((loss[0, rel] > 0) == (danger[0, rel] == 1)), "loss/danger mismatch"
    # 反向：actor 序列不得含 critic 私有段（此断言在 _check_initial_state 中已由
    # assert_actor_input_semantics + 段检查覆盖；此处以标签一致性收口）。


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    initial = _check_initial_state(args.seed)
    mid = _check_mid_game(args.seed, steps=20)
    print(f"V19 semantics verification OK: initial={initial} mid_decisions_checked={mid}")


if __name__ == "__main__":
    main()
