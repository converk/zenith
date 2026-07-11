from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import riichi
import torch
import torch.distributed as dist
from torch import Tensor, nn, optim
from torch.distributions import Categorical

from ppo.config import PPOConfig
from ppo.distributed import DistributedState


NUM_PLAYERS = 4
NUM_ACTIONS = 241

# Expected Rust env protocol:
#   reset() -> observation
#   step(actions: list[str | None]) -> observation, reward[num_envs, 4], done[num_envs]
#
# observation can be either:
#   {"events": [event_or_events_per_env], "requests": [request_or_none] * (num_envs * 4)}
# or:
#   (events, requests)
#
# Events are MJAI JSON strings. Each env entry may contain one event string, None, or
# a list of event strings. Requests are RiichiEnv request_action JSON strings in
# env-major player order; absent requests are None.


def make_env(num_envs: int, seed: int) -> Any:
    return riichi.VecEnv(num_envs, seed)


def make_state_machine(num_envs: int) -> Any:
    return riichi.MjaiKyokuStateMachineManager(num_envs)


def reset_env(
    env: Any,
    state_machine: Any,
    device: torch.device,
    num_envs: int,
) -> tuple[Tensor, Tensor]:
    state_machine.reset()
    observation = env.reset()
    apply_observation(state_machine, observation, num_envs)
    return model_batch(state_machine, device)


def model_batch(state_machine: Any, device: torch.device) -> tuple[Tensor, Tensor]:
    input_ids, _attention_mask, _sequence_lengths = state_machine.model_inputs()
    action_mask = np.asarray(state_machine.action_mask(), dtype=np.bool_)
    action_mask = ensure_valid_action_mask(action_mask)
    return (
        torch.as_tensor(np.asarray(input_ids, dtype=np.int64), dtype=torch.long, device=device),
        torch.as_tensor(action_mask, dtype=torch.bool, device=device),
    )


def ensure_valid_action_mask(action_mask: np.ndarray) -> np.ndarray:
    if action_mask.ndim != 2 or action_mask.shape[1] != NUM_ACTIONS:
        raise ValueError("action_mask must have shape [num_envs * 4, 241]")
    fixed = action_mask.copy()
    empty_rows = ~fixed.any(axis=1)
    fixed[empty_rows, 0] = True
    return fixed


def apply_observation(state_machine: Any, observation: object, num_envs: int) -> None:
    events_by_env, requests = normalize_observation(observation, num_envs)
    for env_index, events in enumerate(events_by_env):
        for event_json in events:
            if event_json is not None:
                state_machine.apply_event(env_index, event_json)
    if requests is not None:
        state_machine.apply_requests(requests)


def normalize_observation(
    observation: object,
    num_envs: int,
) -> tuple[list[list[str | None]], list[str | None] | None]:
    if isinstance(observation, dict):
        events = observation.get("events", observation.get("event"))
        requests = observation.get("requests", observation.get("request_actions"))
    elif isinstance(observation, tuple) and len(observation) == 2:
        events, requests = observation
    else:
        events, requests = observation, None

    events_by_env = normalize_events(events, num_envs)
    normalized_requests = normalize_requests(requests, num_envs)
    return events_by_env, normalized_requests


def normalize_events(events: object, num_envs: int) -> list[list[str | None]]:
    if events is None:
        return [[] for _ in range(num_envs)]
    if isinstance(events, str):
        if num_envs != 1:
            raise ValueError("a single event string is only valid when num_envs == 1")
        return [[events]]
    if not isinstance(events, Sequence):
        raise TypeError("events must be None, a JSON string, or a sequence")
    if len(events) != num_envs:
        raise ValueError(f"events must contain {num_envs} entries")

    normalized: list[list[str | None]] = []
    for item in events:
        if item is None or isinstance(item, str):
            normalized.append([item] if item is not None else [])
        elif isinstance(item, Sequence):
            env_events = []
            for event in item:
                if event is not None and not isinstance(event, str):
                    raise TypeError("event entries must be JSON strings or None")
                env_events.append(event)
            normalized.append(env_events)
        else:
            raise TypeError("each env event entry must be a JSON string, None, or a sequence")
    return normalized


def normalize_requests(requests: object, num_envs: int) -> list[str | None] | None:
    if requests is None:
        return None
    expected = num_envs * NUM_PLAYERS
    if not isinstance(requests, Sequence) or isinstance(requests, str):
        raise TypeError("requests must be a sequence in env-major player order")
    if len(requests) == num_envs:
        flattened: list[str | None] = []
        for env_requests in requests:
            if not isinstance(env_requests, Sequence) or isinstance(env_requests, str):
                raise TypeError("per-env requests must be a sequence of four entries")
            if len(env_requests) != NUM_PLAYERS:
                raise ValueError("each per-env request list must contain four entries")
            flattened.extend(_request_or_none(request) for request in env_requests)
        return flattened
    if len(requests) != expected:
        raise ValueError(f"requests must contain {expected} flat entries or {num_envs} groups")
    return [_request_or_none(request) for request in requests]


def _request_or_none(request: object) -> str | None:
    if request is None or isinstance(request, str):
        return request
    raise TypeError("request entries must be JSON strings or None")


def rewards_array(raw_rewards: object, num_envs: int) -> np.ndarray:
    return np.asarray(raw_rewards, dtype=np.float32).reshape(num_envs, NUM_PLAYERS)


def dones_array(raw_dones: object, num_envs: int) -> np.ndarray:
    return np.asarray(raw_dones, dtype=np.bool_).reshape(num_envs)


def actions_to_env(actions: Tensor, state_machine: Any) -> list[str | None]:
    action_ids = [int(action) for action in actions.detach().cpu().reshape(-1).tolist()]
    return list(state_machine.model_to_env(action_ids))


def batchify_rewards(rewards: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(rewards.reshape(-1), dtype=torch.float32, device=device)


def batchify_dones(dones: np.ndarray, device: torch.device) -> Tensor:
    actor_dones = np.repeat(dones[:, None], NUM_PLAYERS, axis=1)
    return torch.as_tensor(actor_dones.reshape(-1), dtype=torch.float32, device=device)


def reset_done_state_machines(state_machine: Any, dones: np.ndarray) -> None:
    for env_index in np.nonzero(dones)[0]:
        state_machine.reset_env(int(env_index))


def stack_observations(observations: list[Tensor]) -> Tensor:
    rollout_steps = len(observations)
    batch_size = observations[0].shape[0]
    max_length = max(tensor.shape[1] for tensor in observations)
    token_dim = observations[0].shape[2]
    stacked = observations[0].new_zeros((rollout_steps, batch_size, max_length, token_dim))
    for step, tensor in enumerate(observations):
        stacked[step, :, : tensor.shape[1], :] = tensor
    return stacked


@torch.no_grad()
def evaluate(
    agent: nn.Module,
    args: PPOConfig,
    device: torch.device,
) -> float:
    eval_env = make_env(args.num_envs, args.seed + 10_000)
    state_machine = make_state_machine(args.num_envs)
    next_obs, next_masks = reset_env(eval_env, state_machine, device, args.num_envs)
    episode_returns = np.zeros((args.num_envs, NUM_PLAYERS), dtype=np.float32)
    completed_returns: list[float] = []
    max_eval_steps = max(args.eval_episodes * 64, 1_000)

    was_training = agent.training
    agent.eval()
    try:
        for _step in range(max_eval_steps):
            outputs = agent(next_obs, next_masks)
            actions = torch.argmax(outputs["policy_logits"], dim=-1)
            observation, raw_rewards, raw_dones = eval_env.step(
                actions_to_env(actions, state_machine)
            )
            rewards = rewards_array(raw_rewards, args.num_envs)
            dones = dones_array(raw_dones, args.num_envs)
            episode_returns += rewards

            reset_done_state_machines(state_machine, dones)
            apply_observation(state_machine, observation, args.num_envs)
            next_obs, next_masks = model_batch(state_machine, device)

            for env_index in np.nonzero(dones)[0]:
                completed_returns.append(float(np.mean(episode_returns[env_index])))
                episode_returns[env_index] = 0.0
                if len(completed_returns) >= args.eval_episodes:
                    return float(np.mean(completed_returns))
    finally:
        close = getattr(eval_env, "close", None)
        if close is not None:
            close()
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
    env: Any,
    state_machine: Any,
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
    num_actors = args.num_envs * NUM_PLAYERS * distributed_state.world_size
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

        observation, raw_rewards, raw_dones = env.step(actions_to_env(action, state_machine))
        env_rewards = rewards_array(raw_rewards, args.num_envs)
        env_dones = dones_array(raw_dones, args.num_envs)
        episode_returns += env_rewards
        episode_lengths += 1

        for env_index in np.nonzero(env_dones)[0]:
            episodic_returns.append(float(np.mean(episode_returns[env_index])))
            episodic_lengths.append(int(episode_lengths[env_index]))
            episode_returns[env_index] = 0.0
            episode_lengths[env_index] = 0

        rewards.append(batchify_rewards(env_rewards, device))
        dones.append(batchify_dones(env_dones, device))
        reset_done_state_machines(state_machine, env_dones)
        apply_observation(state_machine, observation, args.num_envs)
        next_obs, next_masks = model_batch(state_machine, device)

        if len(obs) >= args.num_steps:
            completed_envs_after_min_steps |= env_dones
        if completed_envs_after_min_steps.all():
            break

    rollout = {
        "obs": stack_observations(obs),
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
    b_obs = rollout["obs"].reshape((-1, *rollout["obs"].shape[2:]))
    b_legal_masks = rollout["legal_masks"].reshape((-1, rollout["legal_masks"].shape[-1]))
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
