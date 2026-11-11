# src/ppo/actor_critic.py
"""Actor‑Critic networks for PPO.
Both networks are simple fully‑connected MLPs.
The actor outputs per‑zone logits that are later reshaped to
(max_zones, max_neighbors) inside the agent.
"""
import torch
import torch.nn as nn


class Actor(nn.Module):
    """Maps an observation vector to logits for each zone‑neighbor pair.

    The output shape is ``(max_zones * max_neighbors,)`` which will be reshaped
    by the PPOAgent to ``(max_zones, max_neighbors)``.
    """

    def __init__(self, obs_dim: int, max_zones: int, max_neighbors: int, hidden_dim: int = 128):
        super().__init__()
        self.max_zones = max_zones
        self.max_neighbors = max_neighbors
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, max_zones * max_neighbors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns flat logits; caller reshapes.
        return self.net(x)


class Critic(nn.Module):
    """Value function estimator returning a scalar value for a given observation."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # shape (batch,)
