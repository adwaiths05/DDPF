# src/mappo/centralized_critic.py
"""Centralized critic for hierarchical MAPPO.
The critic takes the full joint observation (concatenated observations of all agents)
and outputs a single state‑value estimate.  It is a simple MLP that can be shared
across all roles.
"""
import torch
import torch.nn as nn


class CentralizedCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))  # scalar value
        self.net = nn.Sequential(*layers)

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        """Return a scalar value for each batch element.
        ``joint_obs`` shape: (batch, input_dim)
        """
        return self.net(joint_obs).squeeze(-1)
