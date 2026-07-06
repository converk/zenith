from __future__ import annotations

import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from ppo_config import PPOConfig, parse_args
from ppo_model import TileCountTransformerActorCritic, make_toy_ppo_config
from riichi_parallel_env import AGENTS, RiichiVectorEnv


def make_env(num_envs: int, seed: int) -> RiichiVectorEnv:
    return RiichiVectorEnv(num_envs=num_envs, seed=seed)


def checkpoint_path(checkpoint_dir: str, run_name: str, global_step: int) -> Path:
    return Path(checkpoint_dir) / run_name / f"checkpoint_{global_step}.pt"


def save_checkpoint(
    agent: TileCountTransformerActorCritic,
    optimizer: optim.Optimizer,
    args: PPOConfig,
    global_step: int,
    iteration: int,
    completed_episodes: int,
    run_name: str,
) -> Path:
    path = checkpoint_path(args.checkpoint_dir, run_name, global_step)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": asdict(args),
            "global_step": global_step,
            "iteration": iteration,
            "completed_episodes": completed_episodes,
        },
        path,
    )
    return path


def load_checkpoint(
    agent: TileCountTransformerActorCritic,
    optimizer: optim.Optimizer,
    resume_from: str,
    device: torch.device,
) -> tuple[int, int, int]:
    # torch.load uses pickle internally; only load checkpoints you trust.
    checkpoint = torch.load(resume_from, map_location=device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    global_step = int(checkpoint.get("global_step", 0))
    start_iteration = int(checkpoint.get("iteration", 0)) + 1
    completed_episodes = int(checkpoint.get("completed_episodes", 0))
    return global_step, start_iteration, completed_episodes


def batchify_obs(observations: np.ndarray, device: torch.device) -> Tensor:
    """Converts [num_envs, 4, 34] observations to an actor batch."""
    return torch.tensor(
        observations.reshape((-1, 34)),
        dtype=torch.float32,
        device=device,
    )


def batchify_action_mask(action_masks: np.ndarray, device: torch.device) -> Tensor:
    return torch.tensor(
        action_masks.reshape((-1, 34)),
        dtype=torch.bool,
        device=device,
    )


def batchify_rewards(rewards: np.ndarray, device: torch.device) -> Tensor:
    return torch.tensor(rewards.reshape(-1), dtype=torch.float32, device=device)


def batchify_dones(dones: np.ndarray, device: torch.device) -> Tensor:
    actor_dones = np.repeat(dones[:, None], len(AGENTS), axis=1)
    return torch.tensor(actor_dones.reshape(-1), dtype=torch.float32, device=device)


def reset_env(
    env: RiichiVectorEnv,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    observations, action_masks = env.reset(seed=None)
    return batchify_obs(observations, device), batchify_action_mask(action_masks, device)


@torch.no_grad()
def evaluate(
    agent: TileCountTransformerActorCritic,
    args: PPOConfig,
    device: torch.device,
) -> float:
    eval_env = make_env(args.num_envs, args.seed + 10_000)
    env_obs, env_masks = eval_env.reset(seed=None)
    episode_returns = np.zeros((args.num_envs, len(AGENTS)), dtype=np.float32)
    completed_returns: list[float] = []
    max_eval_steps = max(args.eval_episodes * 64, 1_000)

    was_training = agent.training
    agent.eval()
    try:
        for _step in range(max_eval_steps):
            obs = batchify_obs(env_obs, device)
            masks = batchify_action_mask(env_masks, device)
            outputs = agent(obs, masks)
            actions = torch.argmax(outputs["policy_logits"], dim=-1)
            action_batch = actions.reshape(args.num_envs, len(AGENTS)).cpu().numpy()
            env_obs, env_masks, env_rewards, env_dones = eval_env.step(action_batch)
            episode_returns += env_rewards

            for env_index in np.nonzero(env_dones)[0]:
                completed_returns.append(float(np.mean(episode_returns[env_index])))
                episode_returns[env_index] = 0.0
                if len(completed_returns) >= args.eval_episodes:
                    return float(np.mean(completed_returns))
    finally:
        eval_env.close()
        if was_training:
            agent.train()

    if not completed_returns:
        return float("nan")
    return float(np.mean(completed_returns))


@torch.no_grad()
def get_action_and_value(
    agent: TileCountTransformerActorCritic,
    observations: Tensor,
    legal_masks: Tensor,
    actions: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    outputs = agent(observations, legal_masks)
    distribution = Categorical(logits=outputs["policy_logits"])
    if actions is None:
        actions = distribution.sample()
    return (
        actions,
        distribution.log_prob(actions),
        distribution.entropy(),
        outputs["value"],
    )


@torch.no_grad()
def collect_rollout(
    env: RiichiVectorEnv,
    agent: TileCountTransformerActorCritic,
    args: PPOConfig,
    device: torch.device,
    next_obs: Tensor,
    next_masks: Tensor,
    episode_returns: np.ndarray,
    episode_lengths: np.ndarray,
    global_step: int,
) -> tuple[dict[str, Tensor], Tensor, Tensor, np.ndarray, np.ndarray, int, list[float], list[int]]:
    num_actors = args.num_envs * len(AGENTS)
    obs: list[Tensor] = []
    legal_masks: list[Tensor] = []
    actions: list[Tensor] = []
    logprobs: list[Tensor] = []
    rewards: list[Tensor] = []
    dones: list[Tensor] = []
    values: list[Tensor] = []
    episodic_returns: list[float] = []
    episodic_lengths: list[int] = []
    completed_envs_after_min_steps = np.zeros(args.num_envs, dtype=np.bool_)

    while True:
        global_step += num_actors
        obs.append(next_obs)
        legal_masks.append(next_masks)

        action, logprob, _entropy, value = get_action_and_value(
            agent,
            next_obs,
            next_masks,
        )
        values.append(value)
        actions.append(action)
        logprobs.append(logprob)

        action_batch = action.reshape(args.num_envs, len(AGENTS)).detach().cpu().numpy()
        env_obs, env_masks, env_rewards, env_dones = env.step(action_batch)
        episode_returns += env_rewards
        episode_lengths += 1

        for env_index in np.nonzero(env_dones)[0]:
            episodic_returns.append(float(np.mean(episode_returns[env_index])))
            episodic_lengths.append(int(episode_lengths[env_index]))
            episode_returns[env_index] = 0.0
            episode_lengths[env_index] = 0

        rewards.append(batchify_rewards(env_rewards, device))
        dones.append(batchify_dones(env_dones, device))
        next_obs = batchify_obs(env_obs, device)
        next_masks = batchify_action_mask(env_masks, device)

        if len(obs) >= args.num_steps:
            completed_envs_after_min_steps |= env_dones
        if completed_envs_after_min_steps.all():
            break

    rollout = {
        "obs": torch.stack(obs),
        "legal_masks": torch.stack(legal_masks),
        "actions": torch.stack(actions),
        "logprobs": torch.stack(logprobs),
        "rewards": torch.stack(rewards),
        "dones": torch.stack(dones),
        "values": torch.stack(values),
    }
    return (
        rollout,
        next_obs,
        next_masks,
        episode_returns,
        episode_lengths,
        global_step,
        episodic_returns,
        episodic_lengths,
    )


@torch.no_grad()
def compute_gae(
    agent: TileCountTransformerActorCritic,
    rollout: dict[str, Tensor],
    next_obs: Tensor,
    next_masks: Tensor,
    args: PPOConfig,
) -> tuple[Tensor, Tensor]:
    next_value = agent(next_obs, next_masks)["value"].reshape(1, -1)
    advantages = torch.zeros_like(rollout["rewards"])
    lastgaelam = 0
    rollout_steps = rollout["rewards"].shape[0]

    for t in reversed(range(rollout_steps)):
        if t == rollout_steps - 1:
            nextvalues = next_value.reshape(-1)
        else:
            nextvalues = rollout["values"][t + 1]
        nextnonterminal = 1.0 - rollout["dones"][t]
        delta = rollout["rewards"][t] + args.gamma * nextvalues * nextnonterminal - rollout["values"][t]
        advantages[t] = lastgaelam = (
            delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        )

    returns = advantages + rollout["values"]
    return advantages, returns


def update_policy(
    agent: TileCountTransformerActorCritic,
    optimizer: optim.Optimizer,
    rollout: dict[str, Tensor],
    advantages: Tensor,
    returns: Tensor,
    args: PPOConfig,
) -> dict[str, float]:
    b_obs = rollout["obs"].reshape((-1, 34))
    b_legal_masks = rollout["legal_masks"].reshape((-1, 34))
    b_logprobs = rollout["logprobs"].reshape(-1)
    b_actions = rollout["actions"].reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = rollout["values"].reshape(-1)
    batch_size = b_obs.shape[0]
    minibatch_size = max(batch_size // args.num_minibatches, 1)
    b_inds = np.arange(batch_size)
    clipfracs: list[float] = []

    for _epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            outputs = agent(b_obs[mb_inds], b_legal_masks[mb_inds])
            distribution = Categorical(logits=outputs["policy_logits"])
            newlogprob = distribution.log_prob(b_actions[mb_inds])
            entropy = distribution.entropy().mean()
            newvalue = outputs["value"]
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio,
                1 - args.clip_coef,
                1 + args.clip_coef,
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            loss = pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    explained_var = np.nan
    y_pred = b_values.detach().cpu().numpy()
    y_true = b_returns.detach().cpu().numpy()
    var_y = np.var(y_true)
    if var_y > 0:
        explained_var = 1 - np.var(y_true - y_pred) / var_y

    return {
        "value_loss": float(v_loss.detach().cpu()),
        "policy_loss": float(pg_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "old_approx_kl": float(old_approx_kl.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
        "clipfrac": float(np.mean(clipfracs)),
        "explained_var": float(explained_var),
    }


def train(args: PPOConfig) -> None:
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(
        f"Starting run {run_name} on {device} | "
        f"model_size={args.model_size} num_envs={args.num_envs} "
        f"num_steps={args.num_steps} target_iterations={args.target_iterations}",
        flush=True,
    )

    env = make_env(args.num_envs, args.seed)
    agent = TileCountTransformerActorCritic(make_toy_ppo_config(args.model_size)).to(device)
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
        print(
            f"Resumed from {args.resume_from} | "
            f"global_step={global_step} start_iteration={start_iteration} "
            f"completed_episodes={completed_episodes}",
            flush=True,
        )
    start_time = time.time()

    last_iteration = start_iteration - 1
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
            next_obs,
            next_masks,
            episode_returns,
            episode_lengths,
            global_step,
        )
        rollout_steps = rollout["rewards"].shape[0]
        advantages, returns = compute_gae(agent, rollout, next_obs, next_masks, args)
        metrics = update_policy(agent, optimizer, rollout, advantages, returns, args)
        completed_episodes += len(episodic_returns)

        if episodic_returns:
            writer.add_scalar("charts/episodic_return", np.mean(episodic_returns), global_step)
        if episodic_lengths:
            writer.add_scalar("charts/episodic_length", np.mean(episodic_lengths), global_step)
        writer.add_scalar("charts/completed_episodes", completed_episodes, global_step)
        writer.add_scalar("charts/rollout_steps", rollout_steps, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", metrics["value_loss"], global_step)
        writer.add_scalar("losses/policy_loss", metrics["policy_loss"], global_step)
        writer.add_scalar("losses/entropy", metrics["entropy"], global_step)
        writer.add_scalar("losses/old_approx_kl", metrics["old_approx_kl"], global_step)
        writer.add_scalar("losses/approx_kl", metrics["approx_kl"], global_step)
        writer.add_scalar("losses/clipfrac", metrics["clipfrac"], global_step)
        writer.add_scalar("losses/explained_variance", metrics["explained_var"], global_step)
        writer.add_scalar(
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )
        sps = int(global_step / (time.time() - start_time))
        iteration_seconds = time.time() - iteration_start_time
        episodic_return = float(np.mean(episodic_returns)) if episodic_returns else float("nan")
        episodic_length = float(np.mean(episodic_lengths)) if episodic_lengths else float("nan")
        print(
            f"iter={iteration}/{args.target_iterations} "
            f"global_step={global_step} rollout_steps={rollout_steps} "
            f"episodes={completed_episodes} episodic_return={episodic_return:.3f} "
            f"episodic_length={episodic_length:.1f} "
            f"value_loss={metrics['value_loss']:.4f} "
            f"policy_loss={metrics['policy_loss']:.4f} "
            f"entropy={metrics['entropy']:.4f} "
            f"approx_kl={metrics['approx_kl']:.5f} "
            f"sps={sps} time={iteration_seconds:.1f}s",
            flush=True,
        )
        if args.eval_interval > 0 and iteration % args.eval_interval == 0:
            eval_return = evaluate(agent, args, device)
            writer.add_scalar("eval/episodic_return", eval_return, global_step)
            print(
                f"eval iteration={iteration} global_step={global_step} "
                f"episodic_return={eval_return:.3f}",
                flush=True,
            )
        if args.save_interval > 0 and iteration % args.save_interval == 0:
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
    env.close()
    writer.close()


if __name__ == "__main__":
    train(parse_args())
