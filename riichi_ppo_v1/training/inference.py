"""GPU PPO 推理 actor:V16 输入的跨 worker 批处理与 Top-3 Q 输出。"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

try:
    import ray
except ImportError:
    ray = None

from ..model import KyokuTransformerActorCritic, ModelConfig
from ..model.schema import NUM_ACTIONS
from .learner import select_top3_candidates
from .profiling import StageProfiler


def parse_history_namespace(namespace: str) -> int:
    """Parse ``history:uNNN`` into the frozen checkpoint update number."""
    label = str(namespace).removeprefix("history:").removeprefix("u")
    if not label.isdigit():
        raise RuntimeError(f"malformed history namespace {namespace!r}")
    return int(label)


def dispatch_reason(
    worker_ids: list[int], target_workers: int, deadline: float, now: float,
    *, row_count: int = 0, target_rows: int | None = None,
) -> str | None:
    """Return why a pending inference batch may flush, without clock I/O."""
    if target_rows is not None and row_count >= target_rows:
        return "rows"
    if len(set(worker_ids)) >= max(1, target_workers):
        return "target"
    if now >= deadline:
        return "timeout"
    return None


def collate_request_rows(
    requests: list[dict[str, Any]], group: list[tuple[int, int]],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray,
]:
    """把选中请求行按 V16 分段 host-padding,保留请求/行映射。"""
    if not group:
        raise ValueError("cannot collate an empty inference group")

    def column(name: str, dtype: np.dtype) -> np.ndarray:
        return np.asarray(
            [requests[request_index][name][row] for request_index, row in group],
            dtype=dtype,
        )

    history_lengths = column("history_lengths", np.int64)
    snapshot_lengths = column("snapshot_lengths", np.int64)
    pair_counts = column("query_pair_counts", np.int64)
    critic_lengths = column("critic_lengths", np.int64)
    max_history = int(history_lengths.max())
    max_snapshot = int(snapshot_lengths.max())
    max_pairs = int(pair_counts.max())
    max_critic = int(critic_lengths.max(initial=0))
    batch = len(group)
    history_factors = np.zeros((batch, max_history, 10), dtype=np.uint8)
    history_numeric = np.zeros((batch, max_history, 8), dtype=np.float32)
    snapshot_kinds = np.zeros((batch, max_snapshot), dtype=np.uint8)
    snapshot_cat = np.zeros((batch, max_snapshot, 4), dtype=np.uint8)
    snapshot_num = np.zeros((batch, max_snapshot, 7), dtype=np.float32)
    query_rows = np.zeros((batch, 2 * max_pairs, 15), dtype=np.int32)
    query_action_ids = np.zeros((batch, max_pairs), dtype=np.int32)
    critic_factors = np.zeros((batch, max_critic, 10), dtype=np.uint8)
    legal = np.empty((batch, NUM_ACTIONS), dtype=np.bool_)
    for index, (request_index, row) in enumerate(group):
        request = requests[request_index]
        history_length = int(history_lengths[index])
        history_factors[index, :history_length] = request["history_factors"][row, :history_length]
        history_numeric[index, :history_length] = request["history_numeric"][row, :history_length]
        snapshot_length = int(snapshot_lengths[index])
        snapshot_kinds[index, :snapshot_length] = request["snapshot_kinds"][row, :snapshot_length]
        snapshot_cat[index, :snapshot_length] = request["snapshot_cat"][row, :snapshot_length]
        snapshot_num[index, :snapshot_length] = request["snapshot_num"][row, :snapshot_length]
        pair_count = int(pair_counts[index])
        query_rows[index, : 2 * pair_count] = request["query_rows"][row, : 2 * pair_count]
        query_action_ids[index, :pair_count] = request["query_action_ids"][row, :pair_count]
        critic_length = int(critic_lengths[index])
        if critic_length:
            critic_factors[index, :critic_length] = request["critic_factors"][row, :critic_length]
        legal[index] = request["legal_mask"][row]
    return (
        history_factors,
        history_numeric,
        history_lengths,
        snapshot_kinds,
        snapshot_cat,
        snapshot_num,
        snapshot_lengths,
        query_rows,
        query_action_ids,
        pair_counts,
        critic_factors,
        critic_lengths,
        legal,
    )


def assign_batch_outputs(
    responses: list[dict[str, Any]],
    group: list[tuple[int, int]],
    action_ids: list[int],
    logprobs: list[float],
    values: list[float],
    q_taken: list[float],
    expected_q: list[float],
    top3_ids: np.ndarray,
) -> None:
    """Route batched model outputs back to their original RPC and row."""
    if not (
        len(group) == len(action_ids) == len(logprobs) == len(values)
        == len(q_taken) == len(expected_q)
    ):
        raise ValueError("inference output dimensions differ")
    for index, (request_index, row) in enumerate(group):
        responses[request_index]["action_ids"][row] = int(action_ids[index])
        responses[request_index]["logprobs"][row] = float(logprobs[index])
        responses[request_index]["values"][row] = float(values[index])
        responses[request_index]["q_taken"][row] = float(q_taken[index])
        responses[request_index]["expected_q"][row] = float(expected_q[index])
        responses[request_index]["top3_ids"][row] = top3_ids[index].astype(np.int32)


if ray is not None:
    @ray.remote(num_gpus=1, max_concurrency=128)
    class RolloutInferenceActor:
        """一个 GPU 推理 owner:按 namespace 持有 rollout/SFT/历史模型。

        PPO 优化器由 driver 侧 learner 持有;每个 update 后 driver 经
        ``update_weights`` 把最新权重推送到本 actor,worker 的 rollout 请求由
        本 actor 跨 worker 批处理。
        """

        def __init__(self, config: dict[str, Any]) -> None:
            if not torch.cuda.is_available():
                raise RuntimeError("central rollout inference requires CUDA")
            self.config = config
            self.device = torch.device("cuda", 0)
            torch.cuda.set_device(0)
            self.model_config = self._model_config()
            self.model = KyokuTransformerActorCritic(self.model_config).to(self.device)
            self.model.eval()
            self._weights: dict[str, torch.Tensor] | None = None
            self.sft_model: torch.nn.Module | None = None
            self.history_models: dict[str, torch.nn.Module] = {}
            self.profiler = StageProfiler(enabled=bool(config.get("profile_enabled", True)))
            self.profile_cuda_sync = bool(config.get("profile_cuda_sync", False))
            self.cuda_event_interval = max(0, int(config.get("profile_cuda_event_interval", 100)))
            self._forward_calls = 0
            self._pending: list[tuple[dict[str, Any], asyncio.Future[dict[str, Any]], float]] = []
            self._pending_event = asyncio.Event()
            self._drain_task: asyncio.Task[None] | None = None
            self._profile_checkpoint = self.profiler.checkpoint()
            self._counter_checkpoint: dict[str, float] = {}
            self._counters: dict[str, float] = {}
            self._rollout_forward_max_tokens: list[int] = []
            self._rollout_target_workers = int(
                config.get("inference_actor_num_workers", config.get("num_workers", 1))
            )

        def _model_config(self) -> ModelConfig:
            values = vars(ModelConfig.preset("v16"))
            values["context_tokens"] = int(self.config.get("context_tokens", 4096))
            return ModelConfig(**values)

        def _sync_cuda(self) -> None:
            if self.profile_cuda_sync:
                torch.cuda.synchronize(self.device)

        @contextmanager
        def _gpu_stage(self, name: str):
            """Measure submitted GPU work; exact fencing is diagnostic-only."""
            self._sync_cuda()
            with self.profiler.stage(name):
                try:
                    yield
                finally:
                    self._sync_cuda()

        def _add_counter(self, name: str, value: float = 1.0) -> None:
            self._counters[name] = self._counters.get(name, 0.0) + float(value)

        def _target_workers(self) -> int:
            configured = int(self.config.get("inference_batch_target_workers", 0))
            return max(1, configured or self._rollout_target_workers)

        def update_weights(self, weights: dict[str, torch.Tensor]) -> dict[str, float]:
            """接收 driver learner 的最新权重并加载到 rollout 模型。"""
            self._weights = {
                name: value.to(self.device) for name, value in weights.items()
            }
            self.model.load_state_dict(self._weights, strict=True)
            self.model.eval()
            return {}

        def begin_rollout(self, update: int) -> None:
            """开始一个 on-policy rollout,重置本迭代的耗时/计数基线。"""
            self._profile_checkpoint = self.profiler.checkpoint()
            self._counter_checkpoint = dict(self._counters)
            self._rollout_forward_max_tokens = []
            self._rollout_target_workers = int(
                self.config.get(
                    "inference_actor_num_workers", self.config.get("num_workers", 1)
                )
            )

        def _model_for_namespace(self, namespace: str) -> torch.nn.Module:
            if namespace == "rollout":
                return self.model
            if namespace == "sft":
                return self._sft_model()
            if namespace.startswith("history:"):
                return self._history_model(namespace)
            raise RuntimeError(f"unknown inference namespace {namespace!r}")

        def _sft_model(self) -> torch.nn.Module:
            if self.sft_model is None:
                path = Path(self.config["init_model"])
                payload = torch.load(path, map_location="cpu", weights_only=False)
                model = KyokuTransformerActorCritic(self.model_config).to(self.device)
                model.load_state_dict(payload["model"], strict=True)
                model.eval()
                model.requires_grad_(False)
                self.sft_model = model
            return self.sft_model

        def _history_model(self, namespace: str) -> torch.nn.Module:
            """Lazily load one frozen historical PPO checkpoint, then cache it."""
            if namespace in self.history_models:
                return self.history_models[namespace]
            update = parse_history_namespace(namespace)
            path = Path(self.config["checkpoint_dir"]) / f"checkpoint_{update:05d}.pt"
            if not path.is_file():
                raise RuntimeError(
                    f"historical checkpoint missing: {path} (requested {namespace})"
                )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise RuntimeError(f"invalid historical checkpoint: {path}")
            state = payload.get("model")
            if not isinstance(state, dict):
                raise RuntimeError(f"historical checkpoint is missing model weights: {path}")
            model = KyokuTransformerActorCritic(self.model_config).to(self.device)
            model.load_state_dict(state, strict=True)
            model.eval()
            model.requires_grad_(False)
            self.history_models[namespace] = model
            return model

        def rng_state(self) -> dict[str, Any]:
            return {
                "torch_rng": torch.get_rng_state().cpu(),
                "cuda_rng": torch.cuda.get_rng_state(self.device).cpu(),
            }

        def load_rng_state(self, state: dict[str, Any]) -> None:
            torch.set_rng_state(state["torch_rng"].cpu())
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)

        def shutdown(self) -> None:
            self.history_models.clear()
            self.sft_model = None

        def profile_summary(self) -> dict[str, float]:
            """Return one iteration's actor totals without RPC fan-out double counting."""
            stats = self.profiler.delta(self._profile_checkpoint, prefix="timing")
            counters = {
                name: value - self._counter_checkpoint.get(name, 0.0)
                for name, value in self._counters.items()
            }
            dispatches = counters.get("inference/dispatches", 0.0)
            forwards = counters.get("inference/full_forwards", 0.0)
            stats.update(counters)
            stats["inference/dispatch_rows_mean"] = counters.get("inference/dispatch_rows", 0.0) / max(dispatches, 1.0)
            stats["inference/dispatch_workers_mean"] = counters.get("inference/dispatch_workers", 0.0) / max(dispatches, 1.0)
            stats["inference/full_forward_rows_mean"] = counters.get("inference/full_forward_rows", 0.0) / max(forwards, 1.0)
            effective_tokens = counters.get("inference/effective_input_tokens", 0.0)
            padded_tokens = counters.get("inference/padded_input_tokens", 0.0)
            padding_tokens = counters.get("inference/padding_input_tokens", 0.0)
            stats["inference/padding_to_effective_token_ratio"] = padding_tokens / max(effective_tokens, 1.0)
            stats["inference/padding_fraction_of_padded_tokens"] = padding_tokens / max(padded_tokens, 1.0)
            if self._rollout_forward_max_tokens:
                input_lengths = np.asarray(self._rollout_forward_max_tokens, dtype=np.int64)
                for prefix, values in (
                    ("inference/full_forward_max_input_tokens", input_lengths),
                    ("inference/full_forward_max_tokens", input_lengths),
                ):
                    stats[f"{prefix}/min"] = float(values.min())
                    stats[f"{prefix}/mean"] = float(values.mean())
                    stats[f"{prefix}/p50"] = float(np.percentile(values, 50))
                    stats[f"{prefix}/p90"] = float(np.percentile(values, 90))
                    stats[f"{prefix}/p99"] = float(np.percentile(values, 99))
                    stats[f"{prefix}/max"] = float(values.max())
            return stats

        async def infer(
            self,
            *,
            worker_id: int,
            namespace: str,
            batch_indices: np.ndarray,
            history_factors: np.ndarray,
            history_numeric: np.ndarray,
            history_lengths: np.ndarray,
            snapshot_kinds: np.ndarray,
            snapshot_cat: np.ndarray,
            snapshot_num: np.ndarray,
            snapshot_lengths: np.ndarray,
            query_rows: np.ndarray,
            query_action_ids: np.ndarray,
            query_pair_counts: np.ndarray,
            legal_mask: np.ndarray,
            critic_factors: np.ndarray,
            critic_lengths: np.ndarray,
            greedy: bool,
        ) -> dict[str, Any]:
            if len(batch_indices) != len(history_lengths) or history_factors.shape[0] != len(batch_indices):
                raise ValueError("decision metadata batch dimensions differ")
            if (
                snapshot_kinds.shape[0] != len(batch_indices)
                or snapshot_cat.shape[0] != len(batch_indices)
                or snapshot_num.shape[0] != len(batch_indices)
            ):
                raise ValueError("snapshot metadata batch dimensions differ")
            if (
                query_rows.shape[0] != len(batch_indices)
                or query_action_ids.shape[0] != len(batch_indices)
                or len(query_pair_counts) != len(batch_indices)
                or len(snapshot_lengths) != len(batch_indices)
            ):
                raise ValueError("query metadata batch dimensions differ")
            if (
                legal_mask.shape[0] != len(batch_indices)
                or critic_factors.shape[0] != len(batch_indices)
                or len(critic_lengths) != len(batch_indices)
            ):
                raise ValueError("critic metadata batch dimensions differ")
            loop = asyncio.get_running_loop()
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            self._pending.append(({
                "worker_id": int(worker_id),
                "namespace": str(namespace),
                "batch_indices": batch_indices,
                "history_factors": history_factors,
                "history_numeric": history_numeric,
                "history_lengths": history_lengths,
                "snapshot_kinds": snapshot_kinds,
                "snapshot_cat": snapshot_cat,
                "snapshot_num": snapshot_num,
                "snapshot_lengths": snapshot_lengths,
                "query_rows": query_rows,
                "query_action_ids": query_action_ids,
                "query_pair_counts": query_pair_counts,
                "legal_mask": legal_mask,
                "critic_factors": critic_factors,
                "critic_lengths": critic_lengths,
                "greedy": bool(greedy),
            }, future, time.perf_counter()))
            self._pending_event.set()
            if self._drain_task is None:
                self._drain_task = asyncio.create_task(self._drain_requests())
            return await future

        async def _drain_requests(self) -> None:
            try:
                while self._pending:
                    target_workers = self._target_workers()
                    target_rows = max(
                        1,
                        int(self.config.get("inference_batch_target_rows", 0))
                        or int(self.config.get("inference_max_batch_size", 512)),
                    )
                    deadline = time.perf_counter() + max(
                        0.0, float(self.config.get("inference_batch_wait_ms", 5.0))
                    ) / 1_000.0
                    flush_reason: str | None = None
                    while flush_reason is None:
                        flush_reason = dispatch_reason(
                            [request["worker_id"] for request, _future, _queued_at in self._pending],
                            target_workers, deadline, time.perf_counter(),
                            row_count=sum(len(request["batch_indices"]) for request, _future, _queued_at in self._pending),
                            target_rows=target_rows,
                        )
                        if flush_reason is not None:
                            break
                        remaining = deadline - time.perf_counter()
                        self._pending_event.clear()
                        try:
                            await asyncio.wait_for(self._pending_event.wait(), timeout=remaining)
                        except asyncio.TimeoutError:
                            pass
                    pending, self._pending = self._pending, []
                    now = time.perf_counter()
                    workers = {request["worker_id"] for request, _future, _queued_at in pending}
                    self._add_counter("inference/rpc_requests", len(pending))
                    self._add_counter("inference/dispatches")
                    self._add_counter("inference/dispatch_workers", len(workers))
                    self._add_counter(
                        "inference/dispatch_rows",
                        sum(len(request["batch_indices"]) for request, _future, _queued_at in pending),
                    )
                    self._add_counter(f"inference/dispatch_{flush_reason or 'timeout'}")
                    for _request, _future, queued_at in pending:
                        self.profiler.add("inference/queue_wait", now - queued_at)
                    try:
                        results = self._infer_many(
                            [request for request, _future, _queued_at in pending]
                        )
                        for (_request, future, _queued_at), result in zip(pending, results):
                            if not future.done():
                                future.set_result(result)
                    except Exception as exc:
                        for _request, future, _queued_at in pending:
                            if not future.done():
                                future.set_exception(exc)
            finally:
                self._drain_task = None

        def _infer_many(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
            started = time.perf_counter()
            responses = [
                {
                    "action_ids": [0] * len(request["batch_indices"]),
                    "logprobs": [0.0] * len(request["batch_indices"]),
                    "values": [0.0] * len(request["batch_indices"]),
                    "q_taken": [0.0] * len(request["batch_indices"]),
                    "expected_q": [0.0] * len(request["batch_indices"]),
                    "top3_ids": [
                        np.zeros(3, dtype=np.int32) for _row in request["batch_indices"]
                    ],
                }
                for request in requests
            ]
            rows_by_mode: dict[tuple[str, bool], list[tuple[int, int]]] = {}
            for request_index, request in enumerate(requests):
                key = (str(request["namespace"]), bool(request["greedy"]))
                rows_by_mode.setdefault(key, []).extend(
                    (request_index, row) for row in range(len(request["batch_indices"]))
                )
            max_batch = max(1, int(self.config.get("inference_max_batch_size", 512)))
            for (namespace, greedy), rows in rows_by_mode.items():
                model = self._model_for_namespace(namespace)
                for offset in range(0, len(rows), max_batch):
                    group = rows[offset : offset + max_batch]
                    self._run_full_forward(requests, responses, group, greedy, model)
            self.profiler.add("inference/rpc_total", time.perf_counter() - started)
            return responses

        def _run_full_forward(
            self,
            requests: list[dict[str, Any]],
            responses: list[dict[str, Any]],
            group: list[tuple[int, int]],
            greedy: bool,
            model: torch.nn.Module,
        ) -> None:
            if not group:
                return
            with self.profiler.stage("inference/host_collate"):
                (
                    history_factors,
                    history_numeric,
                    history_lengths,
                    snapshot_kinds,
                    snapshot_cat,
                    snapshot_num,
                    snapshot_lengths,
                    query_rows,
                    query_action_ids,
                    pair_counts,
                    critic_factors,
                    critic_lengths,
                    legal,
                ) = collate_request_rows(requests, group)
            sequence_tokens = (
                history_lengths + snapshot_lengths + 2 * pair_counts
            )
            max_tokens = int(sequence_tokens.max())
            effective_input_tokens = int(sequence_tokens.sum())
            padded_input_tokens = len(group) * max_tokens
            self._add_counter("inference/effective_input_tokens", effective_input_tokens)
            self._add_counter("inference/padded_input_tokens", padded_input_tokens)
            self._add_counter(
                "inference/padding_input_tokens", padded_input_tokens - effective_input_tokens
            )
            self._rollout_forward_max_tokens.append(max_tokens)
            with self._gpu_stage("inference/h2d"):
                device_tensors = {
                    name: torch.as_tensor(value, device=self.device)
                    for name, value in (
                        ("history_factors", history_factors.astype(np.int64)),
                        ("history_numeric", history_numeric),
                        ("history_lengths", history_lengths),
                        ("snapshot_kinds", snapshot_kinds.astype(np.int64)),
                        ("snapshot_cat", snapshot_cat.astype(np.int64)),
                        ("snapshot_num", snapshot_num),
                        ("snapshot_lengths", snapshot_lengths),
                        ("query_rows", query_rows.astype(np.int64)),
                        ("query_action_ids", query_action_ids.astype(np.int64)),
                        ("pair_counts", pair_counts),
                        ("critic_factors", critic_factors.astype(np.int64)),
                        ("critic_lengths", critic_lengths),
                        ("legal", legal),
                    )
                }
            self._forward_calls += 1
            sample_cuda = (
                self.profiler.enabled and self.cuda_event_interval > 0
                and self._forward_calls % self.cuda_event_interval == 0
            )
            start_event = torch.cuda.Event(enable_timing=True) if sample_cuda else None
            end_event = torch.cuda.Event(enable_timing=True) if sample_cuda else None
            if start_event is not None:
                start_event.record()
            with self._gpu_stage("inference/full_forward"):
                # forward 必须包 inference_mode,否则导出的张量带 requires_grad,
                # 在无梯度上下文下 .numpy() 会直接报错。
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16,
                    enabled=torch.cuda.is_bf16_supported(),
                ):
                    output = model.forward_v16(
                        device_tensors["history_factors"],
                        device_tensors["history_numeric"],
                        device_tensors["history_lengths"],
                        device_tensors["snapshot_kinds"],
                        device_tensors["snapshot_cat"],
                        device_tensors["snapshot_num"],
                        device_tensors["snapshot_lengths"],
                        device_tensors["query_rows"],
                        device_tensors["query_action_ids"],
                        device_tensors["pair_counts"],
                        device_tensors["legal"],
                        critic_factors=device_tensors["critic_factors"],
                        critic_lengths=device_tensors["critic_lengths"],
                    )
                    logits = output["policy_logits"].float()
                    probabilities = F.softmax(logits, dim=-1)
                    chosen = (
                        logits.argmax(dim=-1)
                        if greedy
                        else torch.multinomial(probabilities, 1).squeeze(1)
                    )
                    logprob = (
                        F.log_softmax(logits, dim=-1)
                        .gather(1, chosen[:, None])
                        .squeeze(1)
                    )
                    boost_ids, _training_ids = select_top3_candidates(
                        logits, device_tensors["legal"], chosen,
                    )
                    boost_valid = boost_ids.ge(0)
                    top3_q = model.q_scores_v16(
                        output["critic_hidden"],
                        output["action_hiddens"],
                        device_tensors["query_action_ids"],
                        device_tensors["pair_counts"],
                        boost_ids.clamp(min=0).long(),
                    )
                    top3_q = top3_q.masked_fill(~boost_valid, 0.0)
                    top3_prob = probabilities.gather(1, boost_ids.clamp(min=0).long())
                    top3_prob = top3_prob.masked_fill(~boost_valid, 0.0)
                    expected_q = (top3_prob * top3_q).sum(dim=-1)
                    behavior_q = model.q_scores_v16(
                        output["critic_hidden"],
                        output["action_hiddens"],
                        device_tensors["query_action_ids"],
                        device_tensors["pair_counts"],
                        chosen[:, None].long(),
                    ).squeeze(-1)
                    values = output["value"].float()
                chosen_cpu = chosen.cpu().numpy().astype(np.int64).tolist()
                logprob_cpu = logprob.cpu().numpy().astype(np.float32).tolist()
                values_cpu = values.cpu().numpy().astype(np.float32).tolist()
                q_taken_cpu = behavior_q.cpu().numpy().astype(np.float32).tolist()
                expected_q_cpu = expected_q.cpu().numpy().astype(np.float32).tolist()
                boost_ids_cpu = boost_ids.cpu().numpy().astype(np.int32)
            if end_event is not None and start_event is not None:
                end_event.record()
                end_event.synchronize()
                self.profiler.add(
                    "inference/full_forward_cuda_event",
                    start_event.elapsed_time(end_event) / 1_000.0,
                )
            self._add_counter("inference/full_forwards")
            self._add_counter("inference/full_forward_rows", len(group))
            with self._gpu_stage("inference/sample_and_d2h"):
                assign_batch_outputs(
                    responses, group, chosen_cpu, logprob_cpu, values_cpu,
                    q_taken_cpu, expected_q_cpu, boost_ids_cpu,
                )
else:
    RolloutInferenceActor = None
