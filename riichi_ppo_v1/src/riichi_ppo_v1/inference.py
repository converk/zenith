"""Single-GPU PPO actor with cross-worker rollout inference batches."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import time
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

try:
    import ray
except ImportError:
    ray = None

from .learner import PPOLearner
from .profiling import StageProfiler
from .trajectory import Transition


def dispatch_reason(worker_ids: list[int], target_workers: int, deadline: float, now: float) -> str | None:
    """Return why a pending inference batch may flush, without clock I/O."""
    if len(set(worker_ids)) >= max(1, target_workers):
        return "target"
    if now >= deadline:
        return "timeout"
    return None


def collate_request_rows(
    requests: list[dict[str, Any]], group: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host-pad selected request rows while preserving their request/row mapping."""
    if not group:
        raise ValueError("cannot collate an empty inference group")
    lengths = np.asarray(
        [requests[request_index]["block_lengths"][row] for request_index, row in group], dtype=np.int64,
    )
    max_length = int(lengths.max())
    batch = len(group)
    kinds = np.zeros((batch, max_length), dtype=np.uint8)
    turn = np.zeros((batch, max_length, 4, 4), dtype=np.uint8)
    meld = np.zeros((batch, max_length, 8), dtype=np.uint8)
    board = np.empty((batch, 12, 160), dtype=np.uint8)
    legal = np.empty((batch, 241), dtype=np.bool_)
    for index, (request_index, row) in enumerate(group):
        request = requests[request_index]
        length = int(lengths[index])
        kinds[index, :length] = request["block_kinds"][row, :length]
        turn[index, :length] = request["turn_fields"][row, :length]
        meld[index, :length] = request["meld_fields"][row, :length]
        board[index] = request["board_state"][row]
        legal[index] = request["legal_mask"][row]
    return kinds, turn, meld, board, legal, lengths


def assign_batch_outputs(
    responses: list[dict[str, Any]], group: list[tuple[int, int]], actions: list[int], logprobs: list[float], values: list[float],
) -> None:
    """Route batched model outputs back to their original RPC and row."""
    if not (len(group) == len(actions) == len(logprobs) == len(values)):
        raise ValueError("inference output dimensions differ")
    for index, (request_index, row) in enumerate(group):
        responses[request_index]["action_ids"][row] = int(actions[index])
        responses[request_index]["logprobs"][row] = float(logprobs[index])
        responses[request_index]["values"][row] = float(values[index])


if ray is not None:
    @ray.remote(num_gpus=1, max_concurrency=128)
    class RolloutInferenceActor:
        """The only process that owns the PPO model, optimizer and CUDA context.

        Workers submit their active decisions synchronously.  The actor waits
        briefly for one request from each worker, then runs a single padded
        full-sequence forward across all those requests.  This avoids the
        tiny exact-length KV-cache buckets that previously serialized rollout
        throughput into thousands of GPU launches.
        """

        def __init__(self, config: dict[str, Any]) -> None:
            if not torch.cuda.is_available():
                raise RuntimeError("central rollout inference requires CUDA")
            self.config = config
            self.device = torch.device("cuda")
            self.dtype = torch.bfloat16 if str(config.get("inference_dtype", "bf16")) == "bf16" else torch.float16
            hyperparameters = {key: value for key, value in config.items() if key not in {"model_size", "device"}}
            self.learner = PPOLearner(str(config["model_size"]), "cuda", **hyperparameters)
            if config.get("resume"):
                self.learner.load(config["resume"])
            self.model = self.learner.model.eval()
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
            # Kept only for the active rollout.  This is intentionally a
            # compact list of scalar lengths (one entry per GPU forward), not
            # per-row tracing data.
            self._rollout_forward_max_event_blocks: list[int] = []

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
            return max(1, configured or int(self.config.get("num_workers", 1)))

        def begin_rollout(self) -> None:
            """Start an on-policy rollout using the current model weights."""
            self._profile_checkpoint = self.profiler.checkpoint()
            self._counter_checkpoint = dict(self._counters)
            self._rollout_forward_max_event_blocks = []

        def iteration(self) -> int:
            return int(self.learner.iteration)

        def update(self, transitions: list[Transition]) -> dict[str, float]:
            """Optimise the same model that served the completed rollout."""
            metrics = self.learner.update(transitions)
            self.model.eval()
            return metrics

        def save(self, path: str, train_config: dict[str, Any]) -> None:
            self.learner.save(path, train_config)

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
            if self._rollout_forward_max_event_blocks:
                # The transformer receives 12 board tokens and one decision
                # token after the padded event prefix for every row.
                input_lengths = np.asarray(self._rollout_forward_max_event_blocks, dtype=np.int64) + 13
                event_lengths = np.asarray(self._rollout_forward_max_event_blocks, dtype=np.int64)
                for prefix, values in (
                    ("inference/full_forward_max_input_tokens", input_lengths),
                    ("inference/full_forward_max_event_blocks", event_lengths),
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
            batch_indices: list[int],
            block_kinds: np.ndarray,
            turn_fields: np.ndarray,
            meld_fields: np.ndarray,
            board_state: np.ndarray,
            legal_mask: np.ndarray,
            block_lengths: np.ndarray,
            greedy: bool,
        ) -> dict[str, Any]:
            if len(batch_indices) != len(block_lengths) or block_kinds.shape[0] != len(batch_indices):
                raise ValueError("decision metadata batch dimensions differ")
            loop = asyncio.get_running_loop()
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            self._pending.append(({
                "worker_id": int(worker_id), "namespace": str(namespace),
                "batch_indices": batch_indices, "block_kinds": block_kinds,
                "turn_fields": turn_fields, "meld_fields": meld_fields,
                "board_state": board_state, "legal_mask": legal_mask,
                "block_lengths": block_lengths, "greedy": bool(greedy),
            }, future, time.perf_counter()))
            self._pending_event.set()
            if self._drain_task is None:
                self._drain_task = asyncio.create_task(self._drain_requests())
            return await future

        async def _drain_requests(self) -> None:
            try:
                while self._pending:
                    target_workers = self._target_workers()
                    deadline = time.perf_counter() + max(0.0, float(self.config.get("inference_batch_wait_ms", 5.0))) / 1_000.0
                    flush_reason: str | None = None
                    while flush_reason is None:
                        flush_reason = dispatch_reason(
                            [request["worker_id"] for request, _future, _queued_at in self._pending],
                            target_workers, deadline, time.perf_counter(),
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
                    self._add_counter("inference/dispatch_rows", sum(len(request["batch_indices"]) for request, _future, _queued_at in pending))
                    self._add_counter(f"inference/dispatch_{flush_reason or 'timeout'}")
                    for _request, _future, queued_at in pending:
                        self.profiler.add("inference/queue_wait", now - queued_at)
                    try:
                        results = self._infer_many([request for request, _future, _queued_at in pending])
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
                {"action_ids": [0] * len(request["batch_indices"]), "logprobs": [0.0] * len(request["batch_indices"]), "values": [0.0] * len(request["batch_indices"])}
                for request in requests
            ]
            rows_by_greedy: dict[bool, list[tuple[int, int]]] = {False: [], True: []}
            for request_index, request in enumerate(requests):
                rows_by_greedy[bool(request["greedy"])].extend(
                    (request_index, row) for row in range(len(request["batch_indices"]))
                )
            max_batch = max(1, int(self.config.get("inference_max_batch_size", 512)))
            for greedy, rows in rows_by_greedy.items():
                for offset in range(0, len(rows), max_batch):
                    group = rows[offset:offset + max_batch]
                    self._run_full_forward(requests, responses, group, greedy)
            self.profiler.add("inference/rpc_total", time.perf_counter() - started)
            return responses

        def _run_full_forward(
            self,
            requests: list[dict[str, Any]],
            responses: list[dict[str, Any]],
            group: list[tuple[int, int]],
            greedy: bool,
        ) -> None:
            if not group:
                return
            with self.profiler.stage("inference/host_collate"):
                kinds, turn, meld, board, legal, lengths = collate_request_rows(requests, group)
            max_event_blocks = int(lengths.max())
            # ``model.forward`` materializes [max_event_blocks + 12 board +
            # 1 decision] positions for every row.  The latter thirteen are
            # always real tokens, so only the event prefix contributes padding.
            effective_input_tokens = int(lengths.sum()) + len(group) * 13
            padded_input_tokens = len(group) * (max_event_blocks + 13)
            self._add_counter("inference/effective_input_tokens", effective_input_tokens)
            self._add_counter("inference/padded_input_tokens", padded_input_tokens)
            self._add_counter("inference/padding_input_tokens", padded_input_tokens - effective_input_tokens)
            self._rollout_forward_max_event_blocks.append(max_event_blocks)
            with self._gpu_stage("inference/h2d"):
                group_kinds = torch.as_tensor(kinds, device=self.device, dtype=torch.long)
                group_turn = torch.as_tensor(turn, device=self.device, dtype=torch.long)
                group_meld = torch.as_tensor(meld, device=self.device, dtype=torch.long)
                group_board = torch.as_tensor(board, device=self.device, dtype=torch.long)
                group_legal = torch.as_tensor(legal, device=self.device, dtype=torch.bool)
                group_lengths = torch.as_tensor(lengths, device=self.device, dtype=torch.long)
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
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self.dtype):
                    output = self.model(group_kinds, group_turn, group_meld, group_board, group_legal, group_lengths)
            if end_event is not None and start_event is not None:
                end_event.record()
                end_event.synchronize()
                self.profiler.add("inference/full_forward_cuda_event", start_event.elapsed_time(end_event) / 1_000.0)
            self._add_counter("inference/full_forwards")
            self._add_counter("inference/full_forward_rows", len(group))
            with self._gpu_stage("inference/sample_and_d2h"):
                distribution = Categorical(logits=output["policy_logits"])
                chosen = output["policy_logits"].argmax(-1) if greedy else distribution.sample()
                chosen_cpu = chosen.tolist()
                logprob_cpu = distribution.log_prob(chosen).tolist()
                value_cpu = output["value"].tolist()
            assign_batch_outputs(responses, group, chosen_cpu, logprob_cpu, value_cpu)
else:
    RolloutInferenceActor = None
