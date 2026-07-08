from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from ppo.checkpoint import load_checkpoint, save_checkpoint
from ppo.config import PPOConfig, parse_args
from ppo.distributed import (
    cleanup_distributed,
    distributed_mean,
    distributed_sum,
    make_run_name,
    setup_distributed,
    unwrap_model,
)
from ppo.model import TileCountTransformerActorCritic, make_toy_ppo_config
from ppo.rollout import collect_rollout, compute_gae, evaluate, make_env, reset_env, update_policy
from riichi_parallel_env import AGENTS


def train(args: PPOConfig) -> None:
    distributed_state = setup_distributed(args)
    device = torch.device(
        f"cuda:{distributed_state.local_rank}"
        if distributed_state.enabled
        else ("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    )
    run_name = make_run_name(args, device, distributed_state)
    writer = SummaryWriter(f"runs/{run_name}") if distributed_state.is_main else None
    if writer is not None:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
        )

    random.seed(args.seed + distributed_state.rank)
    np.random.seed(args.seed + distributed_state.rank)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if distributed_state.is_main:
        print(
            f"Starting run {run_name} on {device} | "
            f"model_size={args.model_size} num_envs_per_rank={args.num_envs} "
            f"world_size={distributed_state.world_size} "
            f"num_steps={args.num_steps} target_iterations={args.target_iterations}",
            flush=True,
        )

    env_seed = args.seed + distributed_state.rank * 100_000
    env = make_env(args.num_envs, env_seed)
    base_agent = TileCountTransformerActorCritic(make_toy_ppo_config(args.model_size)).to(device)
    agent: nn.Module = base_agent
    if distributed_state.enabled:
        agent = DDP(
            base_agent,
            device_ids=[distributed_state.local_rank],
            output_device=distributed_state.local_rank,
            broadcast_buffers=False,
        )
    torch.manual_seed(args.seed + distributed_state.rank)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    next_obs, next_masks = reset_env(env, device)
    episode_returns = np.zeros((args.num_envs, len(AGENTS)), dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    global_step = 0
    start_iteration = 1
    completed_episodes = 0
    if args.resume_from:
        global_step, start_iteration, completed_episodes = load_checkpoint(
            agent,
            optimizer,
            args.resume_from,
            device,
        )
        if distributed_state.is_main:
            print(
                f"Resumed from {args.resume_from} | "
                f"global_step={global_step} start_iteration={start_iteration} "
                f"completed_episodes={completed_episodes}",
                flush=True,
            )
    start_time = time.time()

    last_iteration = start_iteration - 1
    try:
        for iteration in range(start_iteration, args.target_iterations + 1):
            iteration_start_time = time.time()
            last_iteration = iteration
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.target_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            (
                rollout,
                next_obs,
                next_masks,
                episode_returns,
                episode_lengths,
                global_step,
                episodic_returns,
                episodic_lengths,
            ) = collect_rollout(
                env,
                agent,
                args,
                device,
                distributed_state,
                next_obs,
                next_masks,
                episode_returns,
                episode_lengths,
                global_step,
            )
            rollout_steps = rollout["rewards"].shape[0]
            advantages, returns = compute_gae(agent, rollout, next_obs, next_masks, args)
            metrics = update_policy(
                agent,
                optimizer,
                rollout,
                advantages,
                returns,
                args,
                distributed_state,
                device,
            )

            local_episode_count = len(episodic_returns)
            global_episode_count = int(
                distributed_sum(local_episode_count, device, distributed_state)
            )
            completed_episodes += global_episode_count
            global_return_sum = distributed_sum(float(np.sum(episodic_returns)), device, distributed_state)
            global_length_sum = distributed_sum(float(np.sum(episodic_lengths)), device, distributed_state)
            global_return_mean = (
                global_return_sum / global_episode_count
                if global_episode_count > 0
                else float("nan")
            )
            global_length_mean = (
                global_length_sum / global_episode_count
                if global_episode_count > 0
                else float("nan")
            )
            global_metrics = {
                key: distributed_mean(value, device, distributed_state)
                for key, value in metrics.items()
            }

            if writer is not None:
                if global_episode_count > 0:
                    writer.add_scalar("charts/episodic_return", global_return_mean, global_step)
                    writer.add_scalar("charts/episodic_length", global_length_mean, global_step)
                writer.add_scalar("charts/completed_episodes", completed_episodes, global_step)
                writer.add_scalar("charts/rollout_steps", rollout_steps, global_step)
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/value_loss", global_metrics["value_loss"], global_step)
                writer.add_scalar("losses/policy_loss", global_metrics["policy_loss"], global_step)
                writer.add_scalar("losses/entropy", global_metrics["entropy"], global_step)
                writer.add_scalar("losses/old_approx_kl", global_metrics["old_approx_kl"], global_step)
                writer.add_scalar("losses/approx_kl", global_metrics["approx_kl"], global_step)
                writer.add_scalar("losses/clipfrac", global_metrics["clipfrac"], global_step)
                writer.add_scalar(
                    "losses/explained_variance",
                    global_metrics["explained_var"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

            sps = int(global_step / (time.time() - start_time))
            iteration_seconds = time.time() - iteration_start_time
            if distributed_state.is_main:
                print(
                    f"iter={iteration}/{args.target_iterations} "
                    f"global_step={global_step} rollout_steps={rollout_steps} "
                    f"episodes={completed_episodes} episodic_return={global_return_mean:.3f} "
                    f"episodic_length={global_length_mean:.1f} "
                    f"value_loss={global_metrics['value_loss']:.4f} "
                    f"policy_loss={global_metrics['policy_loss']:.4f} "
                    f"entropy={global_metrics['entropy']:.4f} "
                    f"approx_kl={global_metrics['approx_kl']:.5f} "
                    f"sps={sps} time={iteration_seconds:.1f}s",
                    flush=True,
                )

            if args.eval_interval > 0 and iteration % args.eval_interval == 0:
                if distributed_state.enabled:
                    dist.barrier()
                if distributed_state.is_main:
                    eval_return = evaluate(unwrap_model(agent), args, device)
                    if writer is not None:
                        writer.add_scalar("eval/episodic_return", eval_return, global_step)
                    print(
                        f"eval iteration={iteration} global_step={global_step} "
                        f"episodic_return={eval_return:.3f}",
                        flush=True,
                    )
                if distributed_state.enabled:
                    dist.barrier()

            if args.save_interval > 0 and iteration % args.save_interval == 0:
                if distributed_state.enabled:
                    dist.barrier()
                if distributed_state.is_main:
                    saved_path = save_checkpoint(
                        agent,
                        optimizer,
                        args,
                        global_step,
                        iteration,
                        completed_episodes,
                        run_name,
                    )
                    print(f"saved checkpoint: {saved_path}", flush=True)
                if distributed_state.enabled:
                    dist.barrier()
    finally:
        env.close()
        if writer is not None:
            writer.close()
        cleanup_distributed(distributed_state)

    if not distributed_state.is_main:
        return

    saved_path = save_checkpoint(
        agent,
        optimizer,
        args,
        global_step,
        last_iteration,
        completed_episodes,
        run_name,
    )
    print(f"finished run {run_name} | final checkpoint: {saved_path}", flush=True)


if __name__ == "__main__":
    train(parse_args())
