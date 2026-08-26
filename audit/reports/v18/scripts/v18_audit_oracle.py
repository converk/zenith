"""V18 当前局面输入独立审查 oracle（只读，不调用生产 encoder/validator 计算期望值）。

用法（Mahjong-AI 环境）：
    conda run -n Mahjong-AI python audit/reports/v18/scripts/v18_audit_oracle.py

输出：逐项 PASS/FAIL/PARTIAL 的核查结果到 stdout；不写任何文件。
本脚本的“独立 oracle”指：
- 行布局/字段偏移按 specs/010 契约 §3 手工书写（不从 CATEGORY_SCHEMAS 读取）；
- 实体计数、暗牌数、牌河/摘要、supplied 都由本脚本独立推导；
- 只用 riichienv.prepare_current_state_batch 生成“实际值”，绝不借用其内部逻辑算期望。
"""

from __future__ import annotations

import base64
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/disk1/hubowen/zenith")
FIXTURE = ROOT / "RiichiEnv/tests/data/126_204_0_mjai.jsonl"

# ---- 独立行布局（契约 §3，手工书写，不读生产 schema） ----
ROW_WIDTH = 32
KIND_BOS, KIND_TABLE, KIND_SELF_HAND, KIND_SELF_STATE, KIND_PLAYER = 1, 2, 3, 4, 5
KIND_RIVER_SUMMARY, KIND_RIVER_DISCARD, KIND_MELD, KIND_TILE_STATE = 6, 7, 8, 9
KIND_OPPONENT_ANALYSIS, KIND_ACT_O, KIND_ACT_D = 10, 11, 12
SEP_KINDS = set(range(101, 112))

TILE_KINDS = 34
RED_FIVE_IDS = {16, 52, 88}
KIND_OF = lambda tid: int(tid) // 4  # noqa: E731


def kind_of(tile: int) -> int:
    return int(tile) // 4


def tile_type_code(tile: int) -> int:
    return kind_of(tile) + 1


def is_red(tile: int) -> bool:
    return int(tile) in RED_FIVE_IDS


def load_kyoku_records() -> list[str]:
    """从真实 fixture 切出全部 kyoku 记录（start_kyoku..end_kyoku+end_game）。"""
    import riichienv  # noqa: F401  (确保扩展可用)
    lines = [line for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    records, current = [], None
    for line in lines:
        if '"start_kyoku"' in line:
            current = [line]
        elif current is not None:
            current.append(line)
            if '"end_kyoku"' in line:
                records.append("\n".join(current) + "\n" + json.dumps({"type": "end_game"}) + "\n")
                current = None
    if not records:
        raise RuntimeError("fixture contains no kyoku records")
    return records


def decode_actor_rows(rows: np.ndarray) -> dict[int, list[np.ndarray]]:
    """按 kind 分组解码（字段偏移独立书写）。"""
    by_kind: dict[int, list[np.ndarray]] = collections.defaultdict(list)
    for row in rows:
        by_kind[int(row[1])].append(row)
    return by_kind


def pending_draw_actor(obs: object) -> int | None:
    """独立判定“当前持摸牌者”：优先 drawn_tile，否则最近 tsumo/杠/鸣牌事件。

    契约 §3.4/§3.9：tsumo/pon/chi/daiminkan/ankan/kakan 后该玩家持有第 14 张
    （被鸣牌进副露 / 岭上牌），dahai 后各家 13 张。
    """
    if getattr(obs, "drawn_tile", None) is not None:
        return int(obs.player_id)
    for raw in reversed(getattr(obs, "new_events", lambda: [])()):
        value = json.loads(raw)
        kind = value.get("type")
        if kind in ("tsumo", "pon", "chi", "daiminkan", "ankan", "kakan"):
            return int(value["actor"])
        if kind == "dahai":
            return None
    return None


def independent_concealed(obs: object, player: int, pending: int | None) -> int:
    """契约 §3.4/§3.9 定义：13 + (有摸牌) - 3*三张副露 - 4*杠。"""
    melds = obs.melds[player]
    three = sum(1 for m in melds if int(m.meld_type) in (0, 1))
    kans = sum(1 for m in melds if int(m.meld_type) in (2, 3, 4))
    return 13 + (1 if pending == player else 0) - 3 * three - 4 * kans


def independent_entity_counts(obs: object) -> list[int]:
    """按“实体只计一次”统计可见牌：副露全部 + 未被鸣的河牌 + 宝牌指示牌。

    采用 called_tile_index 精确匹配；旧 JSON 无该字段时退化为按实体牌值去重。
    """
    claimed_exact: set[tuple[int, int]] = set()
    claimed_value: collections.Counter[tuple[int, int]] = collections.Counter()
    for meld_list in obs.melds:
        for m in meld_list:
            if int(m.from_who) >= 0 and m.called_tile is not None:
                index = getattr(m, "called_tile_index", None)
                if index is not None:
                    claimed_exact.add((int(m.from_who), int(index)))
                else:
                    claimed_value[(int(m.from_who), int(m.called_tile))] += 1
    counts = [0] * TILE_KINDS
    for player in range(4):
        for index, tile in enumerate(obs.discards[player]):
            if (player, index) in claimed_exact:
                continue
            key = (player, int(tile))
            if claimed_value.get(key, 0) > 0:
                claimed_value[key] -= 1
                continue
            counts[kind_of(tile)] += 1
    for meld_list in obs.melds:
        for m in meld_list:
            for tile in m.tiles:
                counts[kind_of(tile)] += 1
    for tile in obs.dora_indicators:
        counts[kind_of(tile)] += 1
    return counts


def independent_summary_expected(discards: list[int], flags: list[bool], riichi_declared: bool,
                                decl_index: int | None, recent: bool,
                                ) -> list[tuple[int, int, int, int]]:
    """返回该摘要的 6 槽 (tile_type, red, cut, riichi_stage)，0 表示 padding。"""
    n = len(discards)
    selected = discards[-6:] if recent else discards[:6]
    start = max(0, n - 6) if recent else 0
    slots = []
    for i in range(6):
        if i < len(selected):
            tile = selected[i]
            flag = flags[start + i]
            stage = 0
            if riichi_declared and decl_index is not None:
                index = start + i
                stage = 0 if index < decl_index else (1 if index == decl_index else 2)
            slots.append((tile_type_code(tile), int(is_red(tile)), int(flag), stage))
        else:
            slots.append((0, 0, 0, 0))
    return slots


def run_summary_and_sequence_checks(records: list[str]) -> dict[str, object]:
    """真实回放逐决策：序列结构、摘要槽、separator 数、context 上界、决策模式。"""
    import riichienv
    from riichienv import MjaiReplay

    result: dict[str, object] = {"decisions": 0}
    summary_bad: list[str] = []
    sep_counts: collections.Counter = collections.Counter()
    mode_counts: collections.Counter = collections.Counter()
    max_len = 0
    unknown_kind = set()
    for ridx, record in enumerate(records):
        replay = MjaiReplay.from_jsonl_string(record, rule="tenhou")
        kyoku = list(replay.take_kyokus())[0]
        for seat in range(4):
            for step, (obs, _act) in enumerate(kyoku.steps(seat=seat, skip_single_action=False)):
                enc = riichienv.prepare_current_state_batch([getattr(obs, "native_observation", obs)])
                rows = np.asarray(enc.rows).reshape(-1, ROW_WIDTH)
                kinds = rows[:, 1].astype(int)
                result["decisions"] += 1
                max_len = max(max_len, rows.shape[0])
                seps = [int(k) for k in kinds if int(k) in SEP_KINDS]
                sep_counts[len(seps)] += 1
                unknown_kind.update(int(k) for k in kinds if int(k) not in
                                    set(range(1, 15)) | SEP_KINDS)
                table = rows[rows[:, 1] == KIND_TABLE][0]
                mode_counts[int(table[8])] += 1
                # 摘要逐槽独立核对
                pos = int(np.flatnonzero(kinds == 103)[0]) + 1  # SEP_RIVERS
                for rel in (1, 2, 3):
                    assert int(kinds[pos]) in (104, 105, 106)
                    pos += 1
                    first = rows[pos].copy(); pos += 1
                    disc_rows = []
                    while pos < len(rows) and kinds[pos] == KIND_RIVER_DISCARD:
                        disc_rows.append(rows[pos].copy()); pos += 1
                    recent = rows[pos].copy(); pos += 1
                    player = (seat + rel) % 4
                    discards = [int(t) for t in obs.discards[player]]
                    flags = list(obs.tsumogiri_flags[player])
                    declared = bool(obs.riichi_declared[player])
                    decl = obs.riichi_declaration_indices[player]
                    decl_i = None if decl is None else int(decl)
                    exp_first = independent_summary_expected(discards, flags, declared, decl_i, recent=False)
                    exp_recent = independent_summary_expected(discards, flags, declared, decl_i, recent=True)
                    vl1 = int(first[2]); vl2 = int(recent[2])
                    if vl1 != min(6, len(discards)) or vl2 != min(6, len(discards)):
                        summary_bad.append(f"rec{ridx} seat{seat} step{step} rel{rel}: vl {vl1}/{vl2} != {len(discards)}")
                    for slot in range(6):
                        got = tuple(int(v) for v in first[3 + 4 * slot:7 + 4 * slot])
                        if got != exp_first[slot]:
                            summary_bad.append(f"rec{ridx} seat{seat} step{step} rel{rel} first slot{slot}: {got} != {exp_first[slot]}")
                        got = tuple(int(v) for v in recent[3 + 4 * slot:7 + 4 * slot])
                        if got != exp_recent[slot]:
                            summary_bad.append(f"rec{ridx} seat{seat} step{step} rel{rel} recent slot{slot}: {got} != {exp_recent[slot]}")
    result["summary_bad"] = summary_bad
    result["sep_counts"] = dict(sep_counts)
    result["mode_counts"] = dict(mode_counts)
    result["max_len"] = max_len
    result["unknown_kinds"] = sorted(unknown_kind)
    return result


def run_hidden_fact_checks(records: list[str]) -> dict[str, object]:
    """真实回放：concealed、TILE_STATE public/known 实体守恒、supplied 逐河。"""
    import riichienv
    from riichienv import MjaiReplay

    concealed_bad: list[str] = []
    public_bad: list[str] = []
    known_bad: list[str] = []
    supplied_bad: list[str] = []
    dec = 0
    for ridx, record in enumerate(records):
        replay = MjaiReplay.from_jsonl_string(record, rule="tenhou")
        kyoku = list(replay.take_kyokus())[0]
        for seat in range(4):
            for step, (obs, _act) in enumerate(kyoku.steps(seat=seat, skip_single_action=False)):
                dec += 1
                rows = np.asarray(riichienv.prepare_current_state_batch(
                    [getattr(obs, "native_observation", obs)]).rows).reshape(-1, ROW_WIDTH)
                pending = pending_draw_actor(obs)
                for player in range(4):
                    rel = (player - int(obs.player_id)) % 4
                    prows = [r for r in rows if r[1] == KIND_PLAYER and int(r[2]) == rel]
                    if prows:
                        got = int(prows[0][7])
                        exp = independent_concealed(obs, player, pending)
                        if got != exp:
                            concealed_bad.append(
                                f"rec{ridx} seat{seat} step{step} p{player}: {got} != {exp}")
                entity = independent_entity_counts(obs)
                own = [0] * TILE_KINDS
                for tile in obs.hands[int(obs.player_id)]:
                    own[kind_of(tile)] += 1
                for row in rows:
                    if int(row[1]) != KIND_TILE_STATE:
                        continue
                    k = int(row[2]) - 1
                    if int(row[6]) != entity[k]:
                        public_bad.append(
                            f"rec{ridx} seat{seat} step{step} kind{k}: public {row[6]} != entity {entity[k]}")
                    exp_known = min(4, entity[k] + own[k])
                    if int(row[7]) != exp_known:
                        known_bad.append(
                            f"rec{ridx} seat{seat} step{step} kind{k}: known {row[7]} != {exp_known}")
                # supplied：同牌种多次舍出 + 同牌种被鸣 → 全部被打标
                for rel in (1, 2, 3):
                    player = (seat + rel) % 4
                    kinds_disc = collections.Counter(kind_of(int(t)) for t in obs.discards[player])
                    claimed_rel = collections.Counter(
                        kind_of(int(m.called_tile))
                        for meld_list in obs.melds
                        for m in meld_list
                        if int(m.from_who) == player and m.called_tile is not None)
                    for k, _cc in claimed_rel.items():
                        if kinds_disc.get(k, 0) >= 2:
                            marks = [int(r[8]) for r in rows
                                     if r[1] == KIND_RIVER_DISCARD and int(r[3]) == rel and int(r[4]) == k + 1]
                            if len(marks) >= 2 and sum(marks) >= 2:
                                supplied_bad.append(
                                    f"rec{ridx} seat{seat} step{step} rel{rel} kind{k} marks={marks}")
    return {"decisions": dec, "concealed_bad": concealed_bad,
            "public_bad": public_bad, "known_bad": known_bad,
            "supplied_bad": supplied_bad}


def run_synthetic_supplied_check() -> dict[str, object]:
    """构造同牌种两次舍出、仅一次被鸣的 Observation，验证 supplied 只标实体。"""
    import riichienv

    def make_obs(discards, melds, flags):
        data = {
            "player_id": 0,
            "hands": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [], [], []],
            "melds": melds, "discards": discards, "dora_indicators": [],
            "scores": [25000] * 4,
            "riichi_declared": [False] * 4, "riichi_accepted": [False] * 4,
            "riichi_declaration_indices": [None] * 4,
            "missed_agari_doujun": False, "missed_agari_riichi": False,
            "tiles_left": 70, "honba": 0, "riichi_sticks": 0, "round_wind": 0,
            "oya": 0, "kyoku_index": 0, "waits": [], "is_tenpai": False,
            "tsumogiri_flags": flags, "riichi_sutehais": [None] * 4,
            "last_tedashis": [None] * 4, "last_discard": None, "drawn_tile": None,
            "_legal_actions": [], "events": [],
        }
        b64 = base64.b64encode(json.dumps(data).encode()).decode()
        return riichienv.Observation.deserialize_from_base64(b64)

    obs = make_obs(
        discards=[[], [108, 108, 104], [], []],
        melds=[[], [], [{"meld_type": "Pon", "tiles": [108, 108, 109],
                         "opened": True, "from_who": 1, "called_tile": 108,
                         "called_tile_index": 0}], []],
        flags=[[], [False, True, False], [], []],
    )
    rows = np.asarray(riichienv.prepare_current_state_batch(
        [getattr(obs, "native_observation", obs)]).rows).reshape(-1, ROW_WIDTH)
    marks = [(int(r[3]), int(r[8])) for r in rows if r[1] == KIND_RIVER_DISCARD and int(r[2]) == 1]
    return {"river_marks": marks}


def run_action_collision_check(records: list[str]) -> dict[str, object]:
    """真实回放：同一决策内两条不同 action ID 的 O/D token 特征完全相同。"""
    from riichienv import MjaiReplay

    from riichi_ppo_v1.model.action_groups import action_id
    from riichi_ppo_v1.model.current_state import encode_batch

    exact: list[str] = []
    decisions = 0
    for ridx, record in enumerate(records):
        replay = MjaiReplay.from_jsonl_string(record, rule="tenhou")
        kyoku = list(replay.take_kyokus())[0]
        for seat in range(4):
            for step, (obs, _act) in enumerate(kyoku.steps(seat=seat, skip_single_action=False)):
                by_id = {}
                for action in obs.legal_actions():
                    aid = action_id(action, obs)
                    if aid is not None:
                        by_id.setdefault(aid, action)
                if len(by_id) < 2:
                    continue
                decisions += 1
                enc = encode_batch([(obs, [(a, aid) for aid, a in sorted(by_id.items())])])
                rows = enc.actor_factors[0][: int(enc.actor_lengths[0])]
                act_rows = [r for r in rows if int(r[1]) in (11, 12)]
                ids = sorted(by_id.keys())
                off = [tuple(int(v) for v in act_rows[2 * i][2:17]) for i in range(len(ids))]
                deff = [tuple(int(v) for v in act_rows[2 * i + 1][2:17]) for i in range(len(ids))]
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        if off[i] == off[j] and deff[i] == deff[j]:
                            ai, aj = by_id[ids[i]], by_id[ids[j]]
                            exact.append(
                                f"rec{ridx} seat{seat} step{step} ids=({ids[i]},{ids[j]}) "
                                f"{json.loads(ai.to_mjai())['type']} pai={json.loads(ai.to_mjai()).get('pai')} "
                                f"cons_i={list(map(int, ai.consume_tiles))} cons_j={list(map(int, aj.consume_tiles))}")
    return {"decisions": decisions, "exact_collisions": exact}


def main() -> int:
    import riichi
    import riichienv

    print("== 环境 ==")
    print("riichi:", riichi.__file__)
    print("riichienv:", riichienv.__file__)
    print("ENCODING_PROTOCOL_VERSION:", getattr(riichi, "ENCODING_PROTOCOL_VERSION", None))
    print("ANALYSIS_VERSION:", getattr(riichi, "ANALYSIS_VERSION", None))
    print("REPLAY_SEMANTICS_VERSION:", getattr(riichienv, "REPLAY_SEMANTICS_VERSION", None))

    records = load_kyoku_records()
    print(f"== fixture kyoku records: {len(records)} ==")

    print("\n== 1) 序列/摘要/context ==")
    seq = run_summary_and_sequence_checks(records)
    print("decisions:", seq["decisions"], "max_len:", seq["max_len"])
    print("summary_bad:", len(seq["summary_bad"]), seq["summary_bad"][:5])
    print("separator counts:", seq["sep_counts"], "modes:", seq["mode_counts"])
    print("unknown kinds:", seq["unknown_kinds"])

    print("\n== 2) 隐藏信息/实体守恒 ==")
    hidden = run_hidden_fact_checks(records)
    print("decisions:", hidden["decisions"])
    print("concealed_bad:", len(hidden["concealed_bad"]), hidden["concealed_bad"][:3])
    print("public_bad:", len(hidden["public_bad"]), hidden["public_bad"][:3])
    print("known_bad:", len(hidden["known_bad"]), hidden["known_bad"][:3])
    print("supplied_bad(real):", len(hidden["supplied_bad"]), hidden["supplied_bad"][:3])

    print("\n== 3) 合成 supplied 反例 ==")
    print(run_synthetic_supplied_check())

    print("\n== 4) action O/D 特征碰撞 ==")
    col = run_action_collision_check(records)
    print("decisions checked:", col["decisions"])
    print("exact collisions:", len(col["exact_collisions"]))
    for line in col["exact_collisions"][:12]:
        print("  ", line)

    print("\n== 结论 ==")
    verdict = []
    if seq["max_len"] > 256:
        verdict.append("FAIL context")
    else:
        verdict.append("PASS context 上界")
    if seq["summary_bad"]:
        verdict.append("FAIL summary")
    else:
        verdict.append("PASS summary 逐槽")
    if hidden["concealed_bad"]:
        verdict.append("FAIL concealed_count(对手) 与契约不符")
    if hidden["public_bad"]:
        verdict.append("FAIL TILE_STATE public 与实体守恒不符(被鸣牌双计)")
    if hidden["known_bad"]:
        verdict.append("FAIL TILE_STATE known/unknown(被鸣牌双计)")
    if len(col["exact_collisions"]) > 0:
        verdict.append("FAIL action 可区分性(chi consume 碰撞)")
    print("；".join(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
