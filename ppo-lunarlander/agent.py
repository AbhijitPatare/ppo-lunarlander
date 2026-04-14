from typing import Tuple, List
import numpy as np
import torch
import gymnasium as gym
from model import ActorCritic


def collect_rollout(
    env:           gym.Env,
    net:           ActorCritic,
    rollout_steps: int,
    device:        torch.device,
) -> Tuple[torch.Tensor, ...]:
    """Run policy for rollout_steps steps. Return all experience as tensors."""

    states_buf, actions_buf   = [], []
    log_probs_buf, values_buf = [], []
    rewards_buf, dones_buf    = [], []
    episode_reward  = 0.0
    episode_rewards = []

    state, _ = env.reset()

    for step in range(rollout_steps):

        state_t = torch.tensor(state, dtype=torch.float32).to(device)

        with torch.no_grad():
            action, log_prob, value, _ = net.get_action(state_t)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states_buf.append(state)
        actions_buf.append(action.item())
        log_probs_buf.append(log_prob.item())
        values_buf.append(value.item())
        rewards_buf.append(float(reward))
        dones_buf.append(float(done))

        episode_reward += reward

        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            state, _ = env.reset()
        else:
            state = next_state

    # Bootstrap final state value
    with torch.no_grad():
        final_t = torch.tensor(state, dtype=torch.float32).to(device)
        _, next_value = net.forward(final_t)
        next_value = next_value.item()

    # Convert to tensors
    states_t    = torch.tensor(np.array(states_buf),  dtype=torch.float32).to(device)
    actions_t   = torch.tensor(actions_buf,   dtype=torch.long).to(device)
    log_probs_t = torch.tensor(log_probs_buf, dtype=torch.float32).to(device)
    values_t    = torch.tensor(values_buf,    dtype=torch.float32).to(device)
    rewards_t   = torch.tensor(rewards_buf,   dtype=torch.float32).to(device)
    dones_t     = torch.tensor(dones_buf,     dtype=torch.float32).to(device)

    return (
        states_t, actions_t, log_probs_t,
        values_t, rewards_t, dones_t,
        next_value, episode_rewards,
    )


def compute_gae(
    rewards:    torch.Tensor,
    values:     torch.Tensor,
    dones:      torch.Tensor,
    next_value: float,
    gamma:      float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns."""

    advantages = torch.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        next_val = (
            next_value if t == len(rewards) - 1
            else values[t + 1].item()
        )
        mask  = 1.0 - dones[t].item()
        delta = rewards[t].item() + gamma * next_val * mask - values[t].item()
        gae   = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae

    returns     = advantages + values
    advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


