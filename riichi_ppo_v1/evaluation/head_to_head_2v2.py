"""确定性 2v2 跨代评测(V19 侧 host):V19 SFT 双人对阵 V18 SFT 双人。

V18 与 V19 的 Rust 扩展各自定义同名 pyclass,跨 .so 传递 Observation/Action
会被 PyO3 精确类型检查拒绝,单进程共栈不可行。本评测因此采用**双进程锁步**:

- 本进程(host,V19 栈)持有对局环境、V19 状态机与 V19 模型,只驱动 model_a
  的两席;对端(partner,V18 独立工作副本进程)以相同种子运行同构环境、V18
  状态机与 V18 模型,驱动 model_b 的两席。
- 每个决策波:两侧各自对本队席位做模型推理,把
  ``[env_index, seat_id, mjai_action]`` 列表发给对端,再用对端发来的 MJAI
  动作字符串经 ``observation.select_action_from_mjai`` 还原成本代动作对象
  后统一 step。两侧事件流由同一份种子与动作完全确定,天然同步。
- 指标面与 1v3 一致(一位率/平均名次/每座相对其余三家的平均点差/配对
  bootstrap 95% CI/动作分组率/逐小局业务指标);队伍口径为两席点数和之差
  (team_point_diff)。model_b 的业务指标由 partner 统计并在终局汇合。

机制:每分片一个 host+partner 进程对,10 进程(5 对)× 每进程 600 半庄,
双卡各 5 进程;座位轮换为全局半庄序号奇偶 → 席位组 {0,2}/{1,3}。

运行时一致性:锁步要求两侧环境/状态机对同一物理局面产生逐字节相同的事件流
(含 dahai 的 tsumogiri 标记)。本模块因此要求 host 侧加载 ``--v19-runtime-dir``
指定的 V19 扩展新鲜构建(与 V18 工作副本构建同源同特征,行为已验证一致),
而不是站点上可能陈旧的安装版本;加载以 sys.modules 预注册完成,不影响其他
进程与已安装扩展。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# 与训练入口保持一致的项目设备约定;必须先于 torch 导入。
if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

import numpy as np
import torch

from ..model.action_groups import action_group as _action_group
from ..model.bridge import NUM_PLAYERS, BatchedStateBridge
from ..training.metrics import SemanticMetrics
from ..training.rewards import PublicStateTracker
from ..training.worker import active_decisions
from .policy_adapter import load_policy_adapter

def ensure_v19_runtime(runtime_dir: str | Path) -> None:
    """把 V19 扩展新鲜构建预注册进 sys.modules(须先于 riichi/riichienv 导入)。

    ``runtime_dir`` 须含 ``libriichi.so`` 与 ``lib_riichienv.so``(由当前 V19
    源码构建);riichienv 的 Python 包装包沿用本仓库 ``RiichiEnv/src/riichienv``。
    """
    import importlib.machinery
    import importlib.util

    runtime_dir = Path(runtime_dir).resolve()
    riichi_so = runtime_dir / "libriichi.so"
    native_so = runtime_dir / "lib_riichienv.so"
    for so_path in (riichi_so, native_so):
        if not so_path.is_file():
            raise RuntimeError(
                f"V19 评测运行时扩展缺失: {so_path};请用当前源码构建 "
                "`cargo build --release -p riichienv-state-machine "
                "-p riichienv-python --features pyo3/extension-module` "
                "并提供其产物目录"
            )
    if "riichi" in sys.modules or "riichienv" in sys.modules:
        raise RuntimeError("V19 评测运行时必须在导入 riichi/riichienv 前引导")

    def load_extension(so_path: Path, name: str):
        loader = importlib.machinery.ExtensionFileLoader(name, str(so_path))
        spec = importlib.util.spec_from_loader(name, loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        return module

    load_extension(riichi_so, "riichi")
    native = load_extension(native_so, "_riichienv")
    sys.modules.pop("_riichienv", None)
    env_pkg_dir = Path(__file__).resolve().parent.parent.parent / "RiichiEnv" / "src" / "riichienv"
    sys.modules["riichienv._riichienv"] = native
    spec = importlib.util.spec_from_file_location(
        "riichienv", env_pkg_dir / "__init__.py",
        submodule_search_locations=[str(env_pkg_dir)],
    )
    env_pkg = importlib.util.module_from_spec(spec)
    sys.modules["riichienv"] = env_pkg
    spec.loader.exec_module(env_pkg)


TEAM_A_PAIR_EVEN = (0, 2)
TEAM_A_PAIR_ODD = (1, 3)


def team_a_seats_for(hanchan_index: int) -> tuple[int, int]:
    """按全局半庄序号奇偶轮换 model_a 的对家席位组合。"""
    return TEAM_A_PAIR_EVEN if hanchan_index % 2 == 0 else TEAM_A_PAIR_ODD


def send_msg(sock: socket.socket, payload: dict[str, Any]) -> None:
    """长度前缀 + JSON 的单帧发送。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sock.sendall(len(data).to_bytes(4, "big") + data)


def recv_msg(sock: socket.socket) -> dict[str, Any]:
    """长度前缀 + JSON 的单帧接收;对端断开视为致命同步失败。"""
    header = _recv_exact(sock, 4)
    (length,) = np.frombuffer(header, dtype=">u4")
    if length == 0:
        return {}
    return json.loads(_recv_exact(sock, int(length)).decode("utf-8"))


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("2v2 lockstep partner disconnected unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@torch.inference_mode()
def _greedy_action_ids(
    adapter: Any,
    bridge: BatchedStateBridge,
    decisions: list[Any],
    *,
    metrics: SemanticMetrics | None = None,
    public: PublicStateTracker | None = None,
) -> tuple[list[int], dict[str, float]]:
    """V19 侧贪心推理:policy-only 前向 + 信念指标面,返回动作 id。"""
    batch = adapter.prepare(bridge, decisions, None)
    outputs = adapter.outputs(batch)
    logits = outputs["policy_logits"]
    action_ids = logits.argmax(-1).tolist()
    if metrics is not None and public is not None:
        for decision, action_id, legal_row in zip(
            decisions, action_ids, batch.legal, strict=True,
        ):
            metrics.record_decision(
                int(action_id),
                legal_row,
                threat=public.has_riichi_threat(decision.env_index, decision.seat_id),
                prior_riichi_count=int(public.riichi[decision.env_index].sum()),
                seat=decision.seat_id,
            )
    # 信念指标面:标签由评测环境 Rust 侧生成,与 1v3 口径一致。
    belief_metrics: dict[str, float] = {}
    if "belief_hand_logits" in outputs:
        from ..model.belief_labels import encode_belief_labels_batch
        from ..training.belief import belief_metrics_batch

        labels = encode_belief_labels_batch(
            [decision.observation for decision in decisions]
        )
        device_tensors = {
            "belief_hand": torch.as_tensor(labels.hand, device=logits.device),
            "belief_shanten": torch.as_tensor(labels.shanten, device=logits.device),
            "belief_wait": torch.as_tensor(labels.wait, device=logits.device),
            "belief_danger": torch.as_tensor(labels.danger, device=logits.device),
            "belief_loss": torch.as_tensor(labels.loss, device=logits.device),
        }
        belief_metrics = belief_metrics_batch(outputs, device_tensors)
        belief_metrics["decision_count"] = float(len(decisions))
    return action_ids, belief_metrics


def evaluate_2v2_host(
    model_a_path: str | Path,
    *,
    partner_fd: int,
    device: str = "cuda",
    model_a_device: str | None = None,
    hanchan_count: int = 600,
    parallel_hanchans: int = 600,
    seed_base: int = 0,
    game_mode: str = "4p-red-half",
    max_steps: int = 4000,
) -> dict[str, Any]:
    """host 侧主循环;partner 的 model_b 业务指标在终局帧汇合。"""
    if "riichi" not in sys.modules or "riichienv" not in sys.modules:
        raise RuntimeError(
            "2v2 host 必须经由 _2v2_host_entry 入口启动以引导 V19 评测运行时"
        )
    try:
        import riichi
        from riichienv import BatchedRiichiEnv, HandEvaluator
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "install the local riichi and RiichiEnv extensions before evaluation"
        ) from exc

    partner = socket.socket(fileno=partner_fd)

    def final_tenpai(env_index: int, actions_by_env: list[dict[int, Any]]) -> list[bool | None]:
        """流局结算前每座的听牌状态(与 1v3 口径一致)。"""
        flags: list[bool | None] = [None] * NUM_PLAYERS
        observations_by_env = observations[env_index]
        for seat in range(NUM_PLAYERS):
            obs = observations_by_env[seat]
            hands = getattr(obs, "hands", None)
            melds = getattr(obs, "melds", None)
            if hands is None or melds is None:
                continue
            hand = list(hands[seat])
            meld_list = list(melds[seat])
            tile_count = len(hand) + 3 * len(meld_list)
            if tile_count == 13:
                flags[seat] = HandEvaluator(hand, meld_list).is_tenpai()
            elif tile_count == 14:
                action = actions_by_env[env_index].get(seat)
                tile = getattr(action, "tile", None)
                if tile is not None and int(tile) in hand:
                    remaining = list(hand)
                    remaining.remove(int(tile))
                    flags[seat] = HandEvaluator(remaining, meld_list).is_tenpai()
        return flags

    device_a = torch.device(model_a_device or device)
    if device_a.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    model_a_path = str(Path(model_a_path).resolve())
    adapter_a = load_policy_adapter(model_a_path, device=device_a)
    metric_a = SemanticMetrics()

    batch_size = max(1, min(int(parallel_hanchans), int(hanchan_count)))
    started = time.perf_counter()
    rank_history: dict[str, list[int]] = {"a": [], "b": []}
    point_diff_samples: dict[str, list[float]] = {"a": [], "b": []}
    team_point_diffs: list[float] = []
    completed = 0
    seat_counts: dict[str, Counter] = {"a": Counter(), "b": Counter()}
    action_counts: dict[str, Counter] = {"a": Counter()}
    belief_sums: dict[str, float] = {}
    belief_counts = 0
    next_milestone = 100

    for batch_start in range(0, int(hanchan_count), batch_size):
        batch_size_now = min(batch_size, int(hanchan_count) - batch_start)
        envs = BatchedRiichiEnv(
            batch_size_now,
            seed=int(seed_base) + batch_start,
            step_threads=batch_size_now,
            game_mode=game_mode,
        )
        bridge_a = BatchedStateBridge(
            riichi.MjaiKyokuStateMachineManager(batch_size_now), batch_size_now,
        )
        observations = list(envs.reset())
        bridge_a.sync(observations)
        public = PublicStateTracker(batch_size_now)
        public.update(bridge_a.last_events)
        start_scores = [[int(value) for value in row] for row in envs.scores()]
        team_pairs = [
            team_a_seats_for(batch_start + env_index)
            for env_index in range(batch_size_now)
        ]
        for pair in team_pairs:
            seat_counts["a"].update(pair)
            seat_counts["b"].update(
                seat for seat in range(NUM_PLAYERS) if seat not in pair
            )
        active_envs = set(range(batch_size_now))
        kyoku_counts: list[int] = [0] * batch_size_now

        for _step in range(int(max_steps)):
            actions_by_env: list[dict[int, Any]] = [{} for _ in range(batch_size_now)]
            decisions = active_decisions(observations, active_envs)
            own_decisions = [
                decision for decision in decisions
                if decision.seat_id in team_pairs[decision.env_index]
            ]
            # 空决策波也必须发帧,维持与 partner 的逐波锁步。
            frame: list[list[Any]] = []
            if own_decisions:
                action_ids, belief = _greedy_action_ids(
                    adapter_a, bridge_a, own_decisions,
                    metrics=metric_a, public=public,
                )
                # belief_metrics_batch 返回批均值;按批内决策数加权累加,
                # 汇总处再除以总决策数,得到全程决策级均值。
                batch_weight = float(len(own_decisions))
                for name, value in belief.items():
                    belief_sums[name] = (
                        belief_sums.get(name, 0.0) + float(value) * batch_weight
                    )
                belief_counts += len(own_decisions)
                action_counts["a"].update(int(value) for value in action_ids)
                own_actions = bridge_a.decode(own_decisions, action_ids)
                for decision, action in zip(own_decisions, own_actions, strict=True):
                    actions_by_env[decision.env_index][decision.seat_id] = action
                # 本队动作经状态机解码为 MJAI 串后发给对端。
                mjai_actions = bridge_a.state_machine.decode_actions(
                    [decision.batch_index for decision in own_decisions],
                    [int(value) for value in action_ids],
                )
                frame = [
                    [decision.env_index, decision.seat_id, mjai]
                    for decision, mjai in zip(own_decisions, mjai_actions, strict=True)
                ]
            send_msg(partner, {"t": "act", "acts": frame})
            remote = recv_msg(partner)
            for env_index, seat_id, mjai in remote["acts"]:
                action = observations[env_index][seat_id].select_action_from_mjai(mjai)
                if action is None:
                    observation = observations[env_index][seat_id]
                    drawn = getattr(observation, "drawn_tile", None)
                    legal = [
                        json.loads(raw) if raw.startswith("{") else raw
                        for raw in (
                            item.to_mjai() for item in observation.legal_actions()
                        )
                    ]
                    raise RuntimeError(
                        "partner MJAI action rejected: "
                        f"env={env_index} seat={seat_id} mjai={mjai} "
                        f"drawn_tile={drawn} legal={json.dumps(legal[:20])}"
                    )
                actions_by_env[env_index][seat_id] = action

            final_tenpai_by_env: dict[int, list[bool | None]] = {}
            for env_index in active_envs:
                tiles_left = min(
                    int(getattr(observations[env_index][seat], "tiles_left", 1))
                    for seat in range(NUM_PLAYERS)
                )
                if tiles_left <= 0:
                    final_tenpai_by_env[env_index] = final_tenpai(
                        env_index, actions_by_env,
                    )

            observations = list(envs.step_batch(actions_by_env))
            end_kyoku, _end_game = bridge_a.sync(observations)
            public.update(bridge_a.last_events)
            done = envs.done()
            scores_by_env = envs.scores()
            for env_index in list(active_envs):
                if not bool(end_kyoku[env_index]):
                    continue
                kyoku_counts[env_index] += 1
                scores = [int(value) for value in scores_by_env[env_index]]
                pair = team_pairs[env_index]
                ryukyoku_reason = None
                for rows in bridge_a.last_events[env_index]:
                    for raw in rows:
                        try:
                            event = json.loads(raw)
                        except (TypeError, ValueError):
                            continue
                        if event.get("type") == "ryukyoku":
                            ryukyoku_reason = event.get("reason")
                tenpai_flags = final_tenpai_by_env.get(env_index)
                exhaustive_draw = ryukyoku_reason == "exhaustive_draw"
                score_deltas = [
                    scores[player_seat] - start_scores[env_index][player_seat]
                    for player_seat in range(NUM_PLAYERS)
                ]
                seats_a = list(pair)
                seats_b = [seat for seat in range(NUM_PLAYERS) if seat not in pair]
                metric_a.record_kyoku(
                    seats_a, score_deltas, bridge_a.last_events[env_index],
                    draw_tenpai=tenpai_flags if exhaustive_draw else None,
                    exhaustive_draw=exhaustive_draw,
                )
                start_scores[env_index] = scores
                if not bool(done[env_index]):
                    continue
                ranking = sorted(range(NUM_PLAYERS), key=lambda s: (-scores[s], s))
                score_a = sum(scores[seat] for seat in seats_a)
                score_b = sum(scores[seat] for seat in seats_b)
                team_point_diffs.append(float(score_a - score_b))
                # model_a 的 match 面在本进程结算;model_b 的由 partner 结算。
                for seat in seats_a:
                    rank = ranking.index(seat) + 1
                    rank_history["a"].append(rank)
                    others = [
                        scores[other] for other in range(NUM_PLAYERS) if other != seat
                    ]
                    point_diff = float(scores[seat] - float(np.mean(others)))
                    point_diff_samples["a"].append(point_diff)
                    # 与 1v3 一致:半庄结算进 SemanticMetrics(最终点数/被飞/
                    # 名次面),供 summary 的 match 指标与合并加权使用。
                    metric_a.record_match_result(
                        seat, scores, point_delta=point_diff,
                        kyoku_count=kyoku_counts[env_index],
                    )
                for seat in seats_b:
                    rank = ranking.index(seat) + 1
                    rank_history["b"].append(rank)
                    others = [
                        scores[other] for other in range(NUM_PLAYERS) if other != seat
                    ]
                    point_diff_samples["b"].append(
                        float(scores[seat] - float(np.mean(others)))
                    )
                active_envs.remove(env_index)
                completed += 1
            if not active_envs:
                break
        else:
            raise RuntimeError(
                f"2v2 batch {batch_start // batch_size} exceeded {max_steps} steps"
            )
        print(
            f"head_to_head_2v2 completed={completed}/{hanchan_count} "
            f"model_a_first_places={sum(rank == 1 for rank in rank_history['a'])} "
            f"elapsed_s={time.perf_counter() - started:.2f}",
            flush=True,
        )
        while completed >= next_milestone:
            prefix = np.asarray(rank_history["a"][: next_milestone * 2], dtype=np.int64)
            team_prefix = np.asarray(team_point_diffs[:next_milestone], dtype=np.float64)
            print(
                f"2v2_per100 milestone={next_milestone} "
                f"model_a_first_rate={float((prefix == 1).mean()):.3f} "
                f"model_a_last_rate={float((prefix == 4).mean()):.3f} "
                f"model_a_mean_rank={float(prefix.mean()):.3f} "
                f"team_point_diff={float(team_prefix.mean()):+.1f}",
                flush=True,
            )
            next_milestone += 100

    elapsed = time.perf_counter() - started
    # 终局汇合:partner 的 model_b 业务指标与完成数校验。
    final = recv_msg(partner)
    if int(final.get("completed", -1)) != completed:
        raise RuntimeError(
            "2v2 lockstep desync: host completed "
            f"{completed}, partner completed {final.get('completed')}"
        )
    model_b_summary = final["model_b"]

    team_deltas = np.asarray(team_point_diffs, dtype=np.float64)
    bootstrap_rng = np.random.default_rng(int(seed_base))
    team_bootstrap = np.asarray([
        float(np.mean(bootstrap_rng.choice(team_deltas, size=len(team_deltas), replace=True)))
        for _ in range(2000)
    ], dtype=np.float64)
    team_ci95 = [
        float(np.percentile(team_bootstrap, 2.5)),
        float(np.percentile(team_bootstrap, 97.5)),
    ]

    def action_rates(policy: str) -> dict[str, float]:
        grouped = Counter()
        source = (
            action_counts["a"] if policy == "a" else Counter(model_b_summary["action_counts"])
        )
        for action_id, count in source.items():
            grouped[_action_group(int(action_id))] += count
        total = max(sum(grouped.values()), 1)
        return {name: grouped[name] / total for name in (
            "pass", "discard", "reach", "chi", "pon", "kan", "hora", "ryukyoku",
        )}

    def kyoku_metrics(prefix: str, summary: dict[str, float]) -> dict[str, float]:
        return {
            "riichi_rate": summary[f"{prefix}/action/riichi_rate"],
            "riichi_opportunity_accept_rate": summary[
                f"{prefix}/action/riichi_opportunity_accept_rate"
            ],
            "win_rate": summary[f"{prefix}/kyoku/win_rate"],
            "deal_in_rate": summary[f"{prefix}/kyoku/deal_in_rate"],
            "tsumo_loss_rate": summary[f"{prefix}/kyoku/tsumo_loss_rate"],
            "win_points_mean": summary[f"{prefix}/kyoku/win_points_mean"],
            "deal_in_points_mean": summary[f"{prefix}/kyoku/deal_in_points_mean"],
            "draw_tenpai_rate": summary[f"{prefix}/kyoku/draw_tenpai_rate"],
            "kyoku_point_delta_mean": summary[f"{prefix}/kyoku/point_delta_mean"],
            "kyoku_count": summary[f"{prefix}/kyoku/count"],
            "draw_count": summary[f"{prefix}/kyoku/draw_rate"]
            * summary[f"{prefix}/kyoku/count"],
            "exhaustive_draw_count": summary[f"{prefix}/kyoku/exhaustive_draw_count"],
            "exhaustive_draw_rate": summary[f"{prefix}/kyoku/exhaustive_draw_rate"],
            "draw_tenpai_count": summary[f"{prefix}/kyoku/draw_tenpai_count"],
        }

    def belief_metrics() -> dict[str, float]:
        count = max(belief_counts, 1)
        result = {
            name: float(value) / float(count)
            for name, value in belief_sums.items()
            if name != "decision_count"
        }
        result["decision_count"] = float(belief_counts)
        return result

    summary_a = metric_a.summary("model_a")

    def rank_block(policy: str) -> dict[str, Any]:
        ranks = rank_history[policy]
        count = max(len(ranks), 1)
        counts = {rank: sum(rank == value for rank in ranks) for rank in range(1, 5)}
        seat_deltas = np.asarray(point_diff_samples[policy], dtype=np.float64)
        rng = np.random.default_rng(int(seed_base))
        bootstrap = np.asarray([
            float(np.mean(rng.choice(seat_deltas, size=seat_deltas.size, replace=True)))
            for _ in range(2000)
        ], dtype=np.float64)
        return {
            "sample_count": len(ranks),
            "first_place_rate": counts[1] / count,
            "first_place_count": counts[1],
            "second_place_rate": counts[2] / count,
            "second_place_count": counts[2],
            "third_place_rate": counts[3] / count,
            "third_place_count": counts[3],
            "fourth_place_rate": counts[4] / count,
            "fourth_place_count": counts[4],
            "top2_rate": (counts[1] + counts[2]) / count,
            "top2_count": counts[1] + counts[2],
            "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
            "point_diff_vs_mean_others_mean": float(seat_deltas.mean()),
            "point_diff_vs_mean_others_bootstrap_ci95": [
                float(np.percentile(bootstrap, 2.5)),
                float(np.percentile(bootstrap, 97.5)),
            ],
            "point_diff_samples": [float(value) for value in seat_deltas],
        }

    block_a = rank_block("a")
    block_b = rank_block("b")

    return {
        "protocol_version": 1,
        "game_mode": game_mode,
        "format": "2v2",
        "hanchan_count": int(hanchan_count),
        "parallel_hanchans": batch_size,
        "samples_per_hanchan": NUM_PLAYERS // 2,
        "seed_base": int(seed_base),
        "team_a_seat_rotation": "global_hanchan_index % 2: even={0,2}, odd={1,3}",
        "model_a_seat_counts": {
            str(seat): int(count) for seat, count in sorted(seat_counts["a"].items())
        },
        "model_b_seat_counts": {
            str(seat): int(count) for seat, count in sorted(seat_counts["b"].items())
        },
        "model_a": {
            "checkpoint": model_a_path,
            **block_a,
            "final_score_mean": summary_a["model_a/match/final_score_mean"],
            "flying_rate": summary_a["model_a/match/flying_rate"],
            "team_point_diff_mean": float(team_deltas.mean()),
            "team_point_diff_bootstrap_ci95": team_ci95,
            "team_point_diff_samples": [float(value) for value in team_deltas],
            "action_type_rates": action_rates("a"),
            "kyoku_metrics": kyoku_metrics("model_a", summary_a),
            "semantic_metrics": summary_a,
            "belief_metrics": belief_metrics(),
            "metadata": adapter_a.metadata(),
        },
        "model_b": {
            "checkpoint": model_b_summary["checkpoint"],
            **block_b,
            "final_score_mean": model_b_summary["final_score_mean"],
            "flying_rate": model_b_summary["flying_rate"],
            "action_type_rates": action_rates("b"),
            "kyoku_metrics": model_b_summary["kyoku_metrics"],
            "semantic_metrics": model_b_summary["semantic_metrics"],
            "metadata": model_b_summary["metadata"],
        },
        "elapsed_s": elapsed,
        "hanchan_per_s": int(hanchan_count) / max(elapsed, 1e-9),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True, help="V19 SFT checkpoint(本代代码加载)")
    parser.add_argument(
        "--partner-fd", type=int, required=True,
        help="父进程注入的锁步 socket fd(partner 侧对端)",
    )
    parser.add_argument("--hanchans", type=int, default=600)
    parser.add_argument("--parallel-hanchans", type=int, default=600)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--game-mode", default="4p-red-half")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-a-device")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_2v2_host(
        args.model_a,
        partner_fd=args.partner_fd,
        device=args.device,
        model_a_device=args.model_a_device,
        hanchan_count=args.hanchans,
        parallel_hanchans=args.parallel_hanchans,
        seed_base=args.seed_base,
        game_mode=args.game_mode,
        max_steps=args.max_steps,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
