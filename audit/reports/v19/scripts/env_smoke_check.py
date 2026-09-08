"""训练环境端到端体检：conda/GPU → RiichiEnv 活环境 → MJAI 交互转换 → Rust 编码 → GPU 前向 → SFT 数据集。

只读检查,不写任何产物;任何一步失败立即报错退出。
"""
from __future__ import annotations

import gzip
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

FAILURES: list[str] = []


def step(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)


# ── 1. conda / GPU ────────────────────────────────────────────────────────────
print("=== 1. conda / GPU ===", flush=True)
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
cuda_ok = torch.cuda.is_available()
device_count = torch.cuda.device_count() if cuda_ok else 0
step("CUDA 可用", cuda_ok)
step("双卡可见", device_count >= 2, f"device_count={device_count}")
for index in range(min(device_count, 4)):
    props = torch.cuda.get_device_properties(index)
    print(f"  cuda:{index} = {props.name}, {props.total_memory / 2**30:.1f} GiB, cc {props.major}{props.minor}")
step("bf16 支持", cuda_ok and torch.cuda.is_bf16_supported())

# ── 2. RiichiEnv 扩展导入 ─────────────────────────────────────────────────────
print("=== 2. RiichiEnv 扩展 ===", flush=True)
try:
    import riichi
    import riichienv

    step("riichi(状态机) 导入", True, f"api={[n for n in dir(riichi) if not n.startswith('_')][:4]}")
    step("riichienv(PyO3) 导入", True, f"api={[n for n in dir(riichienv) if not n.startswith('_')][:6]}")
except Exception as exc:  # noqa: BLE001
    step("扩展导入", False, repr(exc))
    print("\n".join(FAILURES))
    sys.exit(1)

# ── 3. 活环境(游戏环境, PPO rollout 同路径) ───────────────────────────────────
print("=== 3. 活环境 BatchedRiichiEnv ===", flush=True)
try:
    from riichienv import BatchedRiichiEnv

    envs = BatchedRiichiEnv(
        2, seed=7, step_threads=2, game_mode="4p-red-half", skip_global_mjai_log=True,
    )
    observations = list(envs.reset())
    assert len(observations) == 2
    decisions = 0
    import random

    rng = random.Random(7)
    for _iteration in range(60):
        actions_by_env: list[dict[int, object]] = [{}, {}]
        for env_index, seat_observations in enumerate(observations):
            for seat_id, obs in seat_observations.items():
                legal = obs.legal_actions()
                if legal:
                    actions_by_env[env_index][seat_id] = legal[rng.randrange(len(legal))]
                    decisions += 1
        if envs.done():
            envs.reset_indices([i for i, flag in enumerate(envs.done()) if flag])
        observations = list(envs.step_batch(actions_by_env))
        if decisions > 200:
            break
    scores = envs.scores()
    step(
        "活环境 60 步随机驱动",
        decisions > 0 and len(scores) == 2,
        f"decisions={decisions}, scores={scores}",
    )
except Exception as exc:  # noqa: BLE001
    step("活环境驱动", False, repr(exc))

# ── 4. MJAI 交互与转换(回放 → 状态机 → 合法动作/映射 → Rust 编码) ─────────────
print("=== 4. MJAI 交互与转换 ===", flush=True)
try:
    from riichi_ppo_v1.model.belief_labels import encode_belief_labels_batch
    from riichi_ppo_v1.model.bridge import action_jsons
    from riichi_ppo_v1.model.current_state import encode_batch
    from riichi_ppo_v1.sft.contract import assert_runtime_contract

    assert_runtime_contract()

    tar_path = Path("datasets/tenhou_sft_2024_2025/train/train-00000.tar")
    with tarfile.open(tar_path) as tar:
        member = next(m for m in tar.getmembers() if m.isfile())
        raw = gzip.decompress(tar.extractfile(member).read()).decode("utf-8")

    from riichienv import MjaiReplay

    replay = MjaiReplay.from_jsonl_string(raw, rule="tenhou")
    kyokus = list(replay.take_kyokus())
    kyoku = kyokus[0]
    manager = riichi.MjaiKyokuStateMachineManager(4)
    streams = [iter(kyoku.steps(seat=seat, skip_single_action=False)) for seat in range(4)]
    active = set(range(4))
    checked = 0
    while active and checked < 24:
        batch = []
        for seat in sorted(active):
            try:
                batch.append((seat, *next(streams[seat])))
            except StopIteration:
                active.remove(seat)
        if not batch:
            continue
        env_indices = [seat for seat, _o, _a in batch]
        events_by_env = []
        action_rows = []
        for seat, observation, _expert in batch:
            events = [[], [], [], []]
            events[seat] = list(observation.new_events())
            events_by_env.append(events)
            action_rows.append(action_jsons(observation))
        manager.apply_events_batch(env_indices, events_by_env)
        batch_indices = [seat * 4 + seat for seat, _o, _a in batch]
        prepared_legal = np.asarray(
            manager.prepare_decisions(batch_indices, action_rows), dtype=np.bool_,
        )
        index_rows = manager.action_ids_with_source_indices(batch_indices)
        for row, (seat, observation, expert) in enumerate(batch):
            ids = np.flatnonzero(prepared_legal[row])
            assert ids.size > 0, "空合法掩码"
            mappings = index_rows[row]
            # 映射元组语义与 data.py 一致:(action_id, source_index)。
            assert [int(m[0]) for m in mappings] == ids.tolist(), "action-id 映射与合法掩码不一致"
            assert all(0 <= int(m[1]) < len(action_rows[row]) for m in mappings), "source_index 越界"
        checked += len(batch)
    step(
        "状态机交互+合法动作映射",
        checked >= 24,
        f"批内校验 {checked} 个决策(apply_events/prepare_decisions/action_ids 映射一致)",
    )

    # 重放前 8 个决策并走 Rust 编码 + 信念标签(全新 manager:事件游标不可复用)
    decisions = []
    active = set(range(4))
    manager = riichi.MjaiKyokuStateMachineManager(4)
    streams = [iter(kyoku.steps(seat=seat, skip_single_action=False)) for seat in range(4)]
    while active and len(decisions) < 8:
        batch = []
        for seat in sorted(active):
            try:
                batch.append((seat, *next(streams[seat])))
            except StopIteration:
                active.remove(seat)
        if not batch:
            continue
        env_indices = [seat for seat, _o, _a in batch]
        events_by_env, action_rows = [], []
        for seat, observation, _expert in batch:
            events = [[], [], [], []]
            events[seat] = list(observation.new_events())
            events_by_env.append(events)
            action_rows.append(action_jsons(observation))
        manager.apply_events_batch(env_indices, events_by_env)
        batch_indices = [seat * 4 + seat for seat, _o, _a in batch]
        manager.prepare_decisions(batch_indices, action_rows)
        index_rows = manager.action_ids_with_source_indices(batch_indices)
        for row, (seat, observation, expert) in enumerate(batch):
            legal_actions = list(observation.legal_actions())
            mappings = index_rows[row]
            actions_by_id = [
                (legal_actions[int(src)], int(aid)) for aid, src in mappings
            ]
            decisions.append((observation, actions_by_id))
    encoded = encode_batch(decisions)
    beliefs = encode_belief_labels_batch([obs for obs, _ in decisions])
    step(
        "Rust 当前局面编码(交互转换产物)",
        encoded.actor_factors.shape[2] == 32 and encoded.legal_mask.shape[1] == 241,
        f"actor_factors={tuple(encoded.actor_factors.shape)}, "
        f"legal_mask={tuple(encoded.legal_mask.shape)}, "
        f"belief_hand={beliefs.hand.shape}, loss={beliefs.loss.shape}",
    )
except Exception as exc:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    step("MJAI 交互与转换", False, repr(exc))
    encoded = None

# ── 5. GPU 模型前向(bf16 autocast + 反向) ─────────────────────────────────────
print("=== 5. GPU 模型前向 ===", flush=True)
try:
    assert encoded is not None
    from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
    from riichi_ppo_v1.sft.data import EncodedSample
    from riichi_ppo_v1.sft.trainer import _forward_actor, collate_samples

    # EncodedStateBatch 按行拆成 EncodedSample(与 data.py pending 拆包同构)。
    samples = []
    for row in range(int(encoded.actor_factors.shape[0])):
        count = int(encoded.query_pair_counts[row])
        legal = encoded.legal_mask[row]
        samples.append(EncodedSample(
            actor_factors=encoded.actor_factors[row, : int(encoded.actor_lengths[row])].copy(),
            actor_numeric=encoded.actor_numeric[row, : int(encoded.actor_lengths[row])].copy(),
            query_rows=encoded.query_rows[row, : 2 * count].copy(),
            action_ids=encoded.action_ids[row, :count].copy(),
            legal_mask=legal.copy(),
            action=int(np.flatnonzero(legal)[0]),
            year=0, game_id="smoke", kyoku_index=0, seat=row % 4,
            belief_hand=beliefs.hand[row].copy(),
            belief_shanten=beliefs.shanten[row].copy(),
            belief_wait=beliefs.wait[row].copy(),
            belief_danger=beliefs.danger[row].copy(),
            belief_loss=beliefs.loss[row].copy(),
        ))
    device = torch.device("cuda:0")
    model = KyokuTransformerActorCritic(ModelConfig.preset("v19")).to(device)
    batch = collate_samples(samples, device, validate_semantics=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = _forward_actor(model, batch, {"validate_structure": False})
        logits = output["policy_logits"].float()
        loss = -torch.log_softmax(logits, -1).max(-1).values.mean()
    loss.backward()
    grad_ok = model.belief_network.token_matrix.weight.grad is not None
    # 协议契约:非法动作 logits 恒为 -inf,只检查合法位置有限。
    legal_finite = bool(torch.isfinite(logits[batch["legal_mask"]]).all())
    illegal_inf = bool(torch.isinf(logits[~batch["legal_mask"]]).all())
    step(
        "CUDA bf16 前向+反向",
        legal_finite and illegal_inf and grad_ok,
        f"logits={tuple(logits.shape)}, belief_tokens={tuple(output['belief_tokens'].shape)}, "
        f"token 范数="
        f"{float(model.belief_network.token_matrix.weight.norm()):.4f}",
    )
    del model
    torch.cuda.empty_cache()
except Exception as exc:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    step("GPU 模型前向", False, repr(exc))

# ── 6. SFT 数据集可读性 ───────────────────────────────────────────────────────
print("=== 6. SFT 数据集 ===", flush=True)
try:
    dataset = Path("datasets/tenhou_sft_2024_2025_encoded_60pct_v19_fuzzy")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    shards = sorted((dataset / "train").glob("*.npz"))
    with np.load(shards[0]) as data:
        keys = sorted(data.files)
        rows = int(data[keys[0]].shape[0])
    step(
        "数据集 manifest+分片",
        len(shards) > 0 and rows > 0,
        f"protocol={manifest.get('encoding_protocol_version')}, "
        f"train 分片 {len(shards)} 个, 首片 {rows} 行",
    )
except Exception as exc:  # noqa: BLE001
    step("SFT 数据集", False, repr(exc))

print("=== 体检结果 ===", flush=True)
if FAILURES:
    print("FAILED:", FAILURES)
    sys.exit(1)
print("ALL CHECKS PASSED")
