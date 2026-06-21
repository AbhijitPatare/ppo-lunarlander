from typing import Tuple, Dict
import torch
import torch.nn as nn
from model import ActorCritic


def get_minibatches(rollout_steps: int, batch_size: int) -> list:
    """Randomly split rollout indices into minibatches."""
    indices = torch.randperm(rollout_steps)
    return list(torch.split(indices, batch_size))


def compute_ppo_loss(
    net:           ActorCritic,
    states:        torch.Tensor,
    actions:       torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages:    torch.Tensor,
    returns:       torch.Tensor,
    clip_eps:      float,
    vf_coef:       float,
    ent_coef:      float,
) -> Tuple[torch.Tensor, float, float, float]:
    """Compute combined PPO loss for one minibatch."""

    # Re-run forward pass with gradients
    dist, values_pred = net.forward(states)
    new_log_probs = dist.log_prob(actions)
    entropy       = dist.entropy().mean()

    # ── Policy loss (clipped surrogate) ──
    ratio   = (new_log_probs - old_log_probs).exp()
    surr1   = ratio * advantages
    surr2   = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # ── Value loss (MSE) ──
    value_loss = nn.functional.mse_loss(values_pred, returns)

    # ── Entropy loss (maximize entropy) ──
    entropy_loss = -entropy

    # ── Combined loss ──
    total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

    return total_loss, policy_loss.item(), value_loss.item(), entropy.item()


def ppo_update(
    net:           ActorCritic,
    optimizer:     torch.optim.Optimizer,
    states:        torch.Tensor,
    actions:       torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages:    torch.Tensor,
    returns:       torch.Tensor,
    clip_eps:      float,
    vf_coef:       float,
    ent_coef:      float,
    ppo_epochs:    int,
    batch_size:    int,
    max_grad_norm: float,
) -> Dict[str, float]:
    """Run K epochs of PPO updates. Returns dict of average losses."""

    total_pl, total_vl, total_ent, n = 0.0, 0.0, 0.0, 0

    for _ in range(ppo_epochs):
        for idx in get_minibatches(len(states), batch_size):

            loss, pl, vl, ent = compute_ppo_loss(
                net, states[idx], actions[idx],
                old_log_probs[idx], advantages[idx], returns[idx],
                clip_eps, vf_coef, ent_coef,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            total_pl  += pl
            total_vl  += vl
            total_ent += ent
            n         += 1

    return {
        "losses/policy"  : total_pl  / n,
        "losses/value"   : total_vl  / n,
        "losses/entropy" : total_ent / n,
        "losses/total"   : (total_pl + total_vl) / n,
    }
    
    
