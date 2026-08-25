"""真实 MJAI 文件经 replay 生成固定 V18 Snapshot。"""

from pathlib import Path

from riichienv import MjaiReplay

from riichi_ppo_v1.model.snapshot import encode_snapshot_rows


def test_real_mjai_replay_decisions_encode_v18() -> None:
    path = Path("RiichiEnv/tests/data/126_204_0_mjai.jsonl")
    replay = MjaiReplay.from_jsonl(str(path))
    kyoku = list(replay.take_kyokus())[0]
    checked = 0
    for seat in range(4):
        for observation, _expert in kyoku.steps(seat=seat, skip_single_action=False):
            factors, numeric = encode_snapshot_rows(observation)
            assert factors.shape == (29, 4)
            assert numeric.shape == (29, 1)
            checked += 1
            if checked >= 16:
                break
    assert checked >= 4
