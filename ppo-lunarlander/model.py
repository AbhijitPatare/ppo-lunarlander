"""
model.py — ActorCritic network for PPO on LunarLander-v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO.

    Args:
        state_dim  (int): observation size. LunarLander = 8
        action_dim (int): number of discrete actions. LunarLander = 4
        hidden_dim (int): hidden layer width. Default 256.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared backbone — both heads learn from the same features
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head — outputs logits for each action
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head — outputs a single value V(s)
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Apply orthogonal initialization
        self._init_weights()

    def _init_weights(self):
        """Orthogonal init — faster convergence, more stable PPO training."""
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

        # Actor: tiny init → near-uniform policy at start → max exploration
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

        # Critic: standard init
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, state: torch.Tensor):
        """Run a forward pass. Called automatically by model(state).

        Args:
            state: shape (batch, state_dim) or (state_dim,)

        Returns:
            dist  : Categorical — action distribution
            value : tensor (batch,) — V(s) estimate
        """
        features = self.backbone(state)
        logits   = self.actor_head(features)
        dist     = Categorical(logits=logits)
        value    = self.critic_head(features).squeeze(-1)
        return dist, value

    def get_action(self, state: torch.Tensor):
        """Sample an action and return all data needed for rollout buffer.

        Args:
            state: shape (state_dim,) — single state, no batch dim

        Returns:
            action   : sampled action index
            log_prob : log probability of that action (stored as π_old)
            value    : V(s) from critic
            entropy  : policy entropy (for monitoring exploration)
        """
        dist, value = self.forward(state)
        action      = dist.sample()
        log_prob    = dist.log_prob(action)
        entropy     = dist.entropy()
        return action, log_prob, value, entropy

    def count_parameters(self):
        """Utility — returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing ActorCritic network...")

    # Create the network
    model = ActorCritic(state_dim=8, action_dim=4, hidden_dim=256)
    print(f"Total parameters: {model.count_parameters():,}")
    # Expected: 69,381

    # ── Test 1: single state forward pass ─────────────────────
    single_state = torch.randn(8)              # one random state
    dist, value = model.forward(single_state)

    print(f"\nTest 1 — single state:")
    print(f"  dist type   : {type(dist).__name__}")    # Categorical
    print(f"  probs shape : {dist.probs.shape}")         # torch.Size([4])
    print(f"  probs sum   : {dist.probs.sum().item():.4f}") # must be 1.0000
    print(f"  value shape : {value.shape}")              # torch.Size([])
    print(f"  value       : {value.item():.4f}")          # some float near 0

    # ── Test 2: batch forward pass ────────────────────────────
    batch_states = torch.randn(64, 8)             # 64 states at once
    dist_b, value_b = model.forward(batch_states)

    print(f"\nTest 2 — batch of 64:")
    print(f"  probs shape : {dist_b.probs.shape}")    # torch.Size([64, 4])
    print(f"  value shape : {value_b.shape}")          # torch.Size([64])

    # ── Test 3: get_action ────────────────────────────────────
    action, log_prob, value, entropy = model.get_action(single_state)

    print(f"\nTest 3 — get_action:")
    print(f"  action    : {action.item()}")          # 0, 1, 2, or 3
    print(f"  log_prob  : {log_prob.item():.4f}")    # negative number
    print(f"  entropy   : {entropy.item():.4f}")     # ~1.38 (near log(4)=max)

    # ── Test 4: initial policy is near-uniform ────────────────
    print(f"\nTest 4 — initial policy near-uniform (due to gain=0.01):")
    print(f"  probs: {dist.probs.detach().numpy()}")
    # Expected: something like [0.249, 0.251, 0.248, 0.252]
    # All near 0.25 — confirms gain=0.01 init worked correctly

    print("\nAll tests passed!")

