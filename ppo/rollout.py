from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn, optim
from torch.distributions import Categorical

from ppo.config import PPOConfig
from ppo.distributed import DistributedState
from riichi_parallel_env import AGENTS, RiichiVectorEnv


def make_env(num_envs: int, seed: int) -> RiichiVectorEnv:
    return RiichiVectorEnv(num_envs=num_envs, seed=seed)


def batchify_obs(observations: np.ndarray, device: torch.device) -> Tensor:
    """Converts [num_envs, 4, 34] observations to an actor batch."""
    return torch.as_tensor(
        observations.reshape((-1, 34)),
        dtype=torch.float32,
        device=device,
    )


def batchify_action_mask(action_masks: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(
        action_masks.reshape((-1, 34)),
        dtype=torch.bool,
        device=device,
    )


def batchify_rewards(rewards: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(rewards.reshape(-1), dtype=torch.float32, device=device)


def batchify_dones(dones: np.ndarray, device: torch.device) -> Tensor:
    actor_dones = np.repeat(dones[:, None], len(AGENTS), axis=1)
    return torch.as_tensor(actor_dones.reshape(-1), dtype=torch.float32, device=device)


def reset_env(
    env: RiichiVectorEnv,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    observations, action_masks = env.reset(seed=None)
    return batchify_obs(observations, device), batchify_action_mask(action_masks, device)


@torch.no_grad()
def evaluate(
    agent: nn.Module,
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
    agent: nn.Module,
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
    agent: nn.Module,
    args: PPOConfig,
    device: torch.device,
    distributed_state: DistributedState,
    next_obs: Tensor,
    next_masks: Tensor,
    episode_returns: np.ndarray,
    episode_lengths: np.ndarray,
    global_step: int,
) -> tuple[dict[str, Tensor], Tensor, Tensor, np.ndarray, np.ndarray, int, list[float], list[int]]:
    num_actors = args.num_envs * len(AGENTS) * distributed_state.world_size
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
    agent: nn.Module,
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
    agent: nn.Module,
    optimizer: optim.Optimizer,
    rollout: dict[str, Tensor],
    advantages: Tensor,
    returns: Tensor,
    args: PPOConfig,
    distributed_state: DistributedState,
    device: torch.device,
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
    clipfracs: list[Tensor] = []

    for _epoch in range(args.update_epochs):
        b_inds = torch.randperm(batch_size, device=device)
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
                clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean())

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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None:
            stop_kl = approx_kl.detach()
            if distributed_state.enabled:
                stop_kl = stop_kl.clone()
                dist.all_reduce(stop_kl, op=dist.ReduceOp.MAX)
            if stop_kl > args.target_kl:
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
        "clipfrac": float(torch.stack(clipfracs).mean().detach().cpu()),
        "explained_var": float(explained_var),
    }
