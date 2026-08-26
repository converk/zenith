"""V18 模型结构专项审查（只读）：RoPE 实际生效、mask 逐格 oracle、批内变长隔离、
内容 token 类别向量、结构边界 fail-closed。

用法：
    conda run -n Mahjong-AI python audit/reports/v18/scripts/v18_model_structure_audit.py

不修改任何文件。所有比较均为确定性（模型无 dropout）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/disk1/hubowen/zenith")

sys.path.insert(0, str(ROOT))

import riichi  # noqa: E402
import riichienv  # noqa: E402
from riichienv import MjaiReplay  # noqa: E402

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig  # noqa: E402
from riichi_ppo_v1.model import architecture as arch  # noqa: E402
from riichi_ppo_v1.model.current_state import encode_batch  # noqa: E402
from riichi_ppo_v1.model.encoding_protocol import (  # noqa: E402
    KIND_ACTION_DEFENSE_QUERY,
    KIND_ACTION_OFFENSE_QUERY,
    KIND_BOS,
    KIND_MELD,
    KIND_OPPONENT_ANALYSIS,
    KIND_PLAYER,
    KIND_RIVER_DISCARD,
    KIND_RIVER_SUMMARY,
    KIND_SEP_ACTIONS,
    KIND_SEP_KAMICHA_RIVER,
    KIND_SEP_MELDS,
    KIND_SEP_OPPONENT_ANALYSIS,
    KIND_SEP_PLAYERS,
    KIND_SEP_RIVERS,
    KIND_SEP_SELF_HAND,
    KIND_SEP_SHIMOCHA_RIVER,
    KIND_SEP_TILE_STATE,
    KIND_SEP_TOIMEN_RIVER,
    KIND_SELF_HAND,
    KIND_SELF_STATE_ANALYSIS,
    KIND_TABLE,
    KIND_TILE_STATE,
    SEGMENT_ACTIONS,
    SEGMENT_ANALYSIS,
    SEGMENT_SHARED,
)


# --------------------------------------------------------------------------
# 一、RoPE：forward hook 级验证
# --------------------------------------------------------------------------

def test_rope_applied_in_all_branches() -> dict[str, object]:
    """在 GQA.forward 前挂钩 _rope，验证三条分支的 Q/K 都实际旋转。"""
    from riichi_ppo_v1.model import architecture as mod

    model = KyokuTransformerActorCritic(ModelConfig.preset("v18")).eval()
    # 合成最短合法 actor 输入
    rows = _synthetic_actor_rows(action_ids=(1, 2))
    factors = torch.from_numpy(rows[0]).long()
    numeric = torch.zeros(1, factors.shape[1], 8)
    lengths = torch.from_numpy(rows[1]).long()
    ids = torch.tensor([[1, 2]])
    legal = torch.zeros(1, 241, dtype=torch.bool)
    legal[0, 1] = legal[0, 2] = True

    calls: list[dict[str, object]] = []
    original_rope = mod._rope

    def recording_rope(x: torch.Tensor, values: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        cos, sin = values
        calls.append({
            "x_shape": tuple(x.shape),
            "cos_shape": tuple(cos.shape),
            "sin_shape": tuple(sin.shape),
            "rotated_is_different": bool(not torch.allclose(x, original_rope(x, values))),
            "cos_positions_unique": bool(cos.shape[0] == 1 or cos[:, :, :, 0].unique().numel() > 1),
        })
        return original_rope(x, values)

    mod._rope = recording_rope  # type: ignore[assignment]
    try:
        with torch.no_grad():
            model.forward_actor(
                actor_factors=factors,
                actor_numeric=numeric,
                actor_lengths=lengths,
                query_action_ids=ids,
                query_pair_counts=torch.tensor([2]),
                legal_mask=legal,
            )
    finally:
        mod._rope = original_rope  # type: ignore[assignment]

    return {
        "hook_calls": len(calls),
        "all_rotated_different": all(c["rotated_is_different"] for c in calls),
        "cos_shapes": sorted(list(tuple(v) for v in {c["cos_shape"] for c in calls})),
        "unique_positions_seen": bool(any(c["cos_positions_unique"] for c in calls)),
        "sample_calls": calls[:4],
    }


def test_rope_changes_attention_output() -> dict[str, object]:
    """同一 x/mask：RoPE 开启 vs 关闭（恒等旋转）→ 输出必须不同；
    另验证全双向自注意力下整体平移不改变输出（RoPE 相对位置性质，属预期）。"""
    model = KyokuTransformerActorCritic(ModelConfig.preset("v18")).eval()
    gqa = model.public_backbone.blocks[0].attention
    torch.manual_seed(0)
    x = torch.randn(1, 6, 256)
    mask = torch.ones(1, 6, 6, dtype=torch.bool)
    rope_id = arch._rope_values(torch.tensor([[0, 1, 2, 3, 4, 5]]), 16, torch.float32, 10000.0)
    # 恒等旋转：cos=1, sin=0 → _rope 输出等于原向量
    rope_identity = (torch.ones_like(rope_id[0]), torch.zeros_like(rope_id[1]))
    rope_shift = arch._rope_values(torch.tensor([[3, 4, 5, 6, 7, 8]]), 16, torch.float32, 10000.0)
    with torch.no_grad():
        out_id = gqa(x, rope_identity, mask)
        out_rope = gqa(x, rope_id, mask)
        out_shift = gqa(x, rope_shift, mask)
    return {
        "rope_vs_identity_maxdiff": float((out_rope - out_id).abs().max().item()),
        "rope_vs_identity_differ": bool(not torch.allclose(out_rope, out_id, atol=1e-6)),
        "uniform_shift_maxdiff": float((out_rope - out_shift).abs().max().item()),
        "uniform_shift_is_translation_invariant": bool(torch.allclose(out_rope, out_shift, atol=1e-5)),
        "rope_values_nonidentity": bool(torch.unique(rope_id[0]).numel() > 1),
    }


# --------------------------------------------------------------------------
# 二、mask：独立 oracle 逐格比对（含真实编码样本与合成序列）
# --------------------------------------------------------------------------

def _synthetic_actor_rows(action_ids: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    """构造最短合法 actor 序列（复用测试 fixture 的布局，独立手写）。"""
    from riichi_ppo_v1.tests.v18_fixtures import actor_inputs  # noqa: PLC0415
    batch = actor_inputs(batch=1, action_ids=action_ids)
    factors = batch["actor_factors"].numpy()
    lengths = batch["actor_lengths"].numpy()
    return factors, lengths


def independent_actor_mask(segments: np.ndarray, kinds: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """按契约 §5 独立实现 mask：返回 [B,T,T] 布尔（只含有效查询×有效键）。

    SEP_ACTIONS（kind 110）独立角色：只可读自己；Action 行可读 SEP_ACTIONS。
    """
    batch, tokens = segments.shape
    mask = np.zeros((batch, tokens, tokens), dtype=bool)
    for b in range(batch):
        L = int(lengths[b])
        is_shared = segments[b] == SEGMENT_SHARED
        is_analysis = segments[b] == SEGMENT_ANALYSIS
        is_sep_actions = (segments[b] == SEGMENT_ACTIONS) & (kinds[b] == KIND_SEP_ACTIONS)
        is_action = np.isin(kinds[b], (KIND_ACTION_OFFENSE_QUERY, KIND_ACTION_DEFENSE_QUERY))
        # pair id
        pair = np.zeros(tokens, dtype=int)
        pair_id = -1
        prev_action = False
        for t in range(L):
            if is_action[t]:
                if not prev_action:
                    pair_id += 1
                pair[t] = pair_id
                prev_action = not prev_action  # O/D 成对：第二个 action 后翻转
            else:
                prev_action = False
        for q in range(L):
            if q >= L:
                continue
            for k in range(L):
                if k >= L:
                    continue
                if is_shared[q] and is_shared[k]:
                    mask[b, q, k] = True
                elif is_analysis[q] and (is_shared[k] or is_analysis[k]):
                    mask[b, q, k] = True
                elif is_sep_actions[q] and is_sep_actions[k]:
                    mask[b, q, k] = True
                elif is_action[q] and (
                    is_shared[k] or is_analysis[k] or is_sep_actions[k]
                    or (is_action[k] and pair[q] == pair[k])
                ):
                    mask[b, q, k] = True
    return mask


def test_mask_cell_by_cell() -> dict[str, object]:
    """生产 `_actor_structured_layout` 与独立 oracle 在有效查询×有效键上逐格一致。"""
    # 1) 合成序列
    factors, lengths = _synthetic_actor_rows((1, 7, 12))
    segments = torch.from_numpy(factors[..., 0]).long()
    kinds = torch.from_numpy(factors[..., 1]).long()
    prod_mask, valid = arch._actor_structured_layout(segments, kinds, torch.from_numpy(lengths).long(), factors.shape[1])
    prod_np = prod_mask[0].numpy()
    oracle_np = independent_actor_mask(segments.numpy(), kinds.numpy(), lengths)[0]
    # 生产 mask 含 padding escape（padding 查询→首个 key），去掉 padding 查询比较
    valid_np = valid[0].numpy()
    syn_mismatch = int(np.logical_xor(prod_np[valid_np][:, valid_np], oracle_np[valid_np][:, valid_np]).sum())

    # 2) 真实样本（首个 kyoku，含真实长度与多 action）
    records = load_kyoku_records()
    replay = MjaiReplay.from_jsonl_string(records[0], rule="tenhou")
    kyoku = list(replay.take_kyokus())[0]
    for seat in range(4):
        obs, _act = next(iter(kyoku.steps(seat=seat, skip_single_action=False)))
        break
    by_id = {}
    for action in obs.legal_actions():
        from riichi_ppo_v1.model.action_groups import action_id  # noqa: PLC0415
        aid = action_id(action, obs)
        if aid is not None:
            by_id.setdefault(aid, action)
    enc = encode_batch([(obs, [(a, aid) for aid, a in sorted(by_id.items())])])
    fac = enc.actor_factors[0:1]
    len_arr = enc.actor_lengths[0:1]
    seg = torch.from_numpy(fac[..., 0]).long()
    kind = torch.from_numpy(fac[..., 1]).long()
    prod_mask2, valid2 = arch._actor_structured_layout(seg, kind, torch.from_numpy(len_arr).long(), fac.shape[1])
    prod2 = prod_mask2[0].numpy()
    orc2 = independent_actor_mask(seg.numpy(), kind.numpy(), len_arr)[0]
    valid2_np = valid2[0].numpy()
    real_mismatch = int(np.logical_xor(prod2[valid2_np][:, valid2_np], orc2[valid2_np][:, valid2_np]).sum())
    return {
        "synthetic_cells": int(prod_np.shape[0] * prod_np.shape[1]),
        "synthetic_mismatch": syn_mismatch,
        "real_seq_len": int(len_arr[0]),
        "real_pairs": int(enc.query_pair_counts[0]),
        "real_mismatch": real_mismatch,
        "padding_escape_present": bool(prod_np[~valid_np].any(axis=1).all()),
    }


def test_padding_output_strictly_zero() -> dict[str, object]:
    """forward 后 padding 位置输出必须严格为零（block 内 valid 清零）。"""
    model = KyokuTransformerActorCritic(ModelConfig.preset("v18")).eval()
    factors, lengths = _synthetic_actor_rows((1, 2))
    # 人为加一行 padding，使 T=73 > 有效长度 72
    T = factors.shape[1]
    factors_padded = np.zeros((1, T + 1, 32), dtype=factors.dtype)
    factors_padded[0, :T] = factors[0]
    numeric_padded = np.zeros((1, T + 1, 8), dtype=np.float32)
    lengths_padded = np.array([T], dtype=np.int64)
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inp, out: torch.Tensor) -> None:
        captured.append(out.detach().clone())

    model.actor_backbone.blocks[-1].register_forward_hook(hook)
    with torch.no_grad():
        model.forward_actor(
            actor_factors=torch.from_numpy(factors_padded).long(),
            actor_numeric=torch.from_numpy(numeric_padded),
            actor_lengths=torch.from_numpy(lengths_padded).long(),
            query_action_ids=torch.tensor([[1, 2]]),
            query_pair_counts=torch.tensor([2]),
            legal_mask=torch.zeros(1, 241, dtype=torch.bool).scatter_(1, torch.tensor([[1, 2]]), True),
        )
    out = captured[-1]  # [1,T+1,256]
    L = int(lengths_padded[0])
    return {
        "padding_max_abs": float(out[0, L:].abs().max().item()),
        "valid_nonzero": bool(out[0, :L].abs().sum().item() > 0),
    }


# --------------------------------------------------------------------------
# 三、批内变长/单样本一致性
# --------------------------------------------------------------------------

def load_kyoku_records() -> list[str]:
    fixture = ROOT / "RiichiEnv/tests/data/126_204_0_mjai.jsonl"
    lines = [line for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    records, current = [], None
    for line in lines:
        if '"start_kyoku"' in line:
            current = [line]
        elif current is not None:
            current.append(line)
            if '"end_kyoku"' in line:
                records.append("\n".join(current) + "\n" + json.dumps({"type": "end_game"}) + "\n")
                current = None
    return records


def _first_decision():
    records = load_kyoku_records()
    replay = MjaiReplay.from_jsonl_string(records[0], rule="tenhou")
    kyoku = list(replay.take_kyokus())[0]
    for seat in range(4):
        obs, _act = next(iter(kyoku.steps(seat=seat, skip_single_action=False)))
        return obs
    raise RuntimeError("no decision")


def test_batch_length_independence() -> dict[str, object]:
    """同一采样：单样本 forward 与 [A,A] / [A,B] 批内 forward 的逐 action logits 一致。"""
    from riichi_ppo_v1.model.action_groups import action_id  # noqa: PLC0415

    model = KyokuTransformerActorCritic(ModelConfig.preset("v18")).eval()
    obs_a = _first_decision()
    obs_b = _first_decision()  # 同局面；再用第二个 kyoku 换一个不同长度
    records = load_kyoku_records()
    replay_b = MjaiReplay.from_jsonl_string(records[1], rule="tenhou")
    kyoku_b = list(replay_b.take_kyokus())[0]
    obs_b = next(iter(kyoku_b.steps(seat=0, skip_single_action=False)))[0]

    def encode_one(obs):
        by_id = {}
        for action in obs.legal_actions():
            aid = action_id(action, obs)
            if aid is not None:
                by_id.setdefault(aid, action)
        return encode_batch([(obs, [(a, aid) for aid, a in sorted(by_id.items())])])

    def forward(enc: np.ndarray, idx: int):
        with torch.no_grad():
            return model.forward_actor(
                actor_factors=torch.from_numpy(enc.actor_factors[idx:idx + 1]).long(),
                actor_numeric=torch.from_numpy(enc.actor_numeric[idx:idx + 1]),
                actor_lengths=torch.tensor([int(enc.actor_lengths[idx])]),
                query_action_ids=torch.tensor([enc.action_ids[idx].tolist()]),
                query_pair_counts=torch.tensor([int(enc.query_pair_counts[idx])]),
                legal_mask=torch.from_numpy(enc.legal_mask[idx:idx + 1]),
            )["raw_policy_logits"]

    enc_a = encode_one(obs_a)
    enc_b = encode_one(obs_b)
    ids_a = enc_a.action_ids[0].tolist()
    logits_single = forward(enc_a, 0)[0, ids_a].numpy()

    # [A,A]
    aa = torch.from_numpy(enc_a.actor_factors).long()
    with torch.no_grad():
        logits_aa = model.forward_actor(
            actor_factors=aa,
            actor_numeric=torch.from_numpy(enc_a.actor_numeric),
            actor_lengths=torch.from_numpy(enc_a.actor_lengths).long(),
            query_action_ids=torch.tensor(enc_a.action_ids.tolist()),
            query_pair_counts=torch.from_numpy(enc_a.query_pair_counts).long(),
            legal_mask=torch.from_numpy(enc_a.legal_mask),
        )["raw_policy_logits"].numpy()

    # [A,B] 混合（不同长度/不同 action 数）
    la, lb = int(enc_a.actor_lengths[0]), int(enc_b.actor_lengths[0])
    cap = max(enc_a.actor_factors.shape[1], enc_b.actor_factors.shape[1])
    mixed = np.zeros((2, cap, 32), dtype=enc_a.actor_factors.dtype)
    mixed[0, :la] = enc_a.actor_factors[0, :la]
    mixed[1, :lb] = enc_b.actor_factors[0, :lb]
    mixed_num = np.zeros((2, cap, 8), dtype=np.float32)
    mixed_num[0, :la] = enc_a.actor_numeric[0, :la]
    mixed_num[1, :lb] = enc_b.actor_numeric[0, :lb]
    qa, qb = int(enc_a.query_pair_counts[0]), int(enc_b.query_pair_counts[0])
    qcap = max(qa, qb)
    mixed_ids = np.zeros((2, qcap), dtype=np.int64)
    mixed_ids[0, :qa] = enc_a.action_ids[0, :qa]
    mixed_ids[1, :qb] = enc_b.action_ids[0, :qb]
    mixed_legal = np.zeros((2, 241), dtype=bool)
    mixed_legal[0] = enc_a.legal_mask[0]
    mixed_legal[1] = enc_b.legal_mask[0]
    with torch.no_grad():
        logits_mixed = model.forward_actor(
            actor_factors=torch.from_numpy(mixed).long(),
            actor_numeric=torch.from_numpy(mixed_num),
            actor_lengths=torch.tensor([la, lb]),
            query_action_ids=torch.tensor(mixed_ids),
            query_pair_counts=torch.tensor([qa, qb]),
            legal_mask=torch.from_numpy(mixed_legal),
        )["raw_policy_logits"].numpy()

    return {
        "len_a": int(enc_a.actor_lengths[0]),
        "len_b": int(enc_b.actor_lengths[0]),
        "single_vs_AA_maxdiff": float(np.abs(logits_single - logits_aa[0][ids_a]).max()),
        "single_vs_mixed_maxdiff": float(np.abs(logits_single - logits_mixed[0][ids_a]).max()),
        "A_equal_AB_row0": bool(np.allclose(logits_aa[0][ids_a], logits_mixed[0][ids_a], atol=1e-6)),
    }


# --------------------------------------------------------------------------
# 四、内容 token 类别向量（segment/kind）是否被覆盖
# --------------------------------------------------------------------------

def test_content_token_base_embedding_dropped() -> dict[str, object]:
    """验证内容 token 的 segment+kind 基础向量是否保留（相加）。

    B5 修复后：对同一内容 token 仅改 segment 字段必须改变最终 embedding（非 0=相加，
    0.0=被覆盖）。
    """
    from riichi_ppo_v1.tests.v18_fixtures import shared_prefix_rows  # noqa: PLC0415

    model = KyokuTransformerActorCritic(ModelConfig.preset("v18")).eval()
    emb = model.token_embedding
    rows = np.stack(shared_prefix_rows())
    factors = torch.from_numpy(rows).long().unsqueeze(0)
    numeric = torch.zeros(1, factors.shape[1], 8)
    with torch.no_grad():
        output = emb(factors, numeric)  # [1,T,256]
    content_mask = (factors[0, :, 1] != 1)  # 非 BOS
    # 独立计算 base = segment + kind
    base = emb.segment(factors[..., 0].long()) + emb.kind(factors[..., 1].long())
    # 若“相加”成立：output ≈ base + (output - base)。检查 output 与 base 的关系：
    # 对内容 token，若 base 被丢弃，则 output 与 base 无相加关系：output - field_proj == 0 更难直接得，
    # 改用等价判定：把 factor 字段全置 0 但保留 segment/kind → 若 base 起作用，输出应等于 base（对 separator）
    # 对内容 token（有字段），字段为 0 时输出=field_proj(0)=0? 不行。
    # 用更强判定：对每个内容 token，比较 output 与 base 的“接近度”以及 output 与基向量零空间。
    # 直接证据：把任意内容 token 的 segment 值改为另一合法 segment（kind 不变，表不变），
    # base 应平移；若被覆盖则输出完全不变。
    factors2 = factors.clone()
    # 找一个 PLAYER 内容行，改它的 segment（表按 kind 独立，不受 segment 影响）
    p_row = torch.nonzero(factors2[0, :, 1] == KIND_PLAYER)[0, 0].item()
    factors2[0, p_row, 0] = 2  # SEGMENT_ANALYSIS（非法组合，但只测嵌入层行为）
    with torch.no_grad():
        output2 = emb(factors2, numeric)
    return {
        "segment_change_output_maxdiff_per_token": float((output[0, p_row] - output2[0, p_row]).abs().max().item()),
        "base_norm_at_that_token": float(base[0, p_row].norm().item()),
        "output_norm_at_that_token": float(output[0, p_row].norm().item()),
    }


# --------------------------------------------------------------------------
# 五、结构边界 fail-closed
# --------------------------------------------------------------------------

def test_critic_empty_fail_closed() -> dict[str, object]:
    """critic_factors 为空/长度 0 时应报错而不是 IndexError 崩溃。"""
    from riichi_ppo_v1.tests.v18_fixtures import actor_inputs  # noqa: PLC0415

    model = KyokuTransformerActorCritic(ModelConfig.preset("v18")).eval()
    batch = actor_inputs(batch=1, action_ids=(1, 7))
    try:
        with torch.no_grad():
            model(
                actor_factors=batch["actor_factors"],
                actor_numeric=batch["actor_numeric"],
                actor_lengths=batch["actor_lengths"],
                query_action_ids=batch["action_ids"],
                query_pair_counts=batch["query_pair_counts"],
                legal_mask=batch["legal_mask"],
                critic_factors=torch.zeros(1, 0, 32, dtype=torch.long),
                critic_lengths=torch.tensor([0]),
            )
        return {"raises": False, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"raises": True, "error": f"{type(exc).__name__}: {exc}"}


def test_duplicate_action_id_scatter() -> dict[str, object]:
    """重复 action id 是否被模型层拒绝（防御性）。"""
    model = KyokuTransformerActorCritic(ModelConfig.preset("v18")).eval()
    batch = _synthetic_actor_rows((1, 2))
    try:
        with torch.no_grad():
            model.forward_actor(
                actor_factors=torch.from_numpy(batch[0]).long(),
                actor_numeric=torch.zeros(1, batch[0].shape[1], 8),
                actor_lengths=torch.tensor(batch[1]),
                query_action_ids=torch.tensor([[1, 1]]),
                query_pair_counts=torch.tensor([2]),
                legal_mask=torch.zeros(1, 241, dtype=torch.bool).scatter_(1, torch.tensor([[1]]), True),
            )
        return {"rejected": False}
    except Exception as exc:  # noqa: BLE001
        return {"rejected": True, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    print("环境：", riichi.__file__, riichienv.__file__)
    print("ENCODING_PROTOCOL_VERSION:", getattr(riichi, "ENCODING_PROTOCOL_VERSION", None))
    print("\n[1] RoPE forward-hook 级验证")
    print(json.dumps(test_rope_applied_in_all_branches(), ensure_ascii=False, indent=2))
    print("\n[2] RoPE 对注意力输出的因果影响（同一 x/mask 换位置）")
    print(json.dumps(test_rope_changes_attention_output(), ensure_ascii=False, indent=2))
    print("\n[3] mask 逐格独立 oracle")
    print(json.dumps(test_mask_cell_by_cell(), ensure_ascii=False, indent=2))
    print("\n[4] padding 输出严格为零")
    print(json.dumps(test_padding_output_strictly_zero(), ensure_ascii=False, indent=2))
    print("\n[5] 批内变长/单样本一致性")
    print(json.dumps(test_batch_length_independence(), ensure_ascii=False, indent=2))
    print("\n[6] 内容 token 的 segment 变化是否影响嵌入（>0=相加保留基础向量，0=被覆盖）")
    print(json.dumps(test_content_token_base_embedding_dropped(), ensure_ascii=False, indent=2))
    print("\n[7] critic 空输入 fail-closed")
    print(json.dumps(test_critic_empty_fail_closed(), ensure_ascii=False, indent=2))
    print("\n[8] 重复 action id 模型层防御")
    print(json.dumps(test_duplicate_action_id_scatter(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
