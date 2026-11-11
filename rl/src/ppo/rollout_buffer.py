# src/ppo/rollout_buffer.py
"""Rollout buffer for PPO with optional GAE.
The buffer stores trajectories for a single episode (or a batch of steps) and
provides methods to compute advantages and retrieve batched tensors for the
training update.
"""
import numpy as np
import torch
from collections import deque


class RolloutBuffer:
    def __init__(self, obs_dim: int, max_zones: int, max_neighbors: int, buffer_size: int = 2048, gamma: float = 0.99, lam: float = 0.95):
        self.obs_dim = obs_dim
        self.max_zones = max_zones
        self.max_neighbors = max_neighbors
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lam = lam
        # pre‑allocate lists
        self.observations = []
        self.actions = []          # shape (max_zones,) per step (int indices per zone)
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.masks = []            # action mask per step (max_zones, max_neighbors)

    def add(self, obs: torch.Tensor, action: np.ndarray, log_prob: torch.Tensor, reward: float, done: bool, value: torch.Tensor, mask: np.ndarray):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.masks.append(mask)
        # truncate oldest entry if over capacity (FIFO)
        if len(self.observations) > self.buffer_size:
            self.observations.pop(0)
            self.actions.pop(0)
            self.log_probs.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.values.pop(0)
            self.masks.pop(0)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.masks.clear()

    def compute_returns_and_advantages(self):
        """Compute GAE returns and advantages.
        Handles empty buffers and correctly concatenates scalar value tensors for bootstrapping.
        """
        if len(self.rewards) == 0:
            return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)
        device = self.values[0].device if self.values else torch.device('cpu')
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        values = torch.stack(self.values).detach().to(device)
        if dones[-1]:
            next_value = torch.tensor([0.0], dtype=torch.float32, device=device)
        else:
            next_value = values[-1].unsqueeze(0)
        values = torch.cat([values, next_value])
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages[step] = gae
        returns = advantages + values[:-1]
        return returns, advantages



    def get_batch(self):
        returns, advantages = self.compute_returns_and_advantages()
        batch = {
            "obs": torch.stack(self.observations),
            "actions": torch.from_numpy(np.array(self.actions)).long(),
            "log_probs": torch.stack(self.log_probs),
            "returns": returns,
            "advantages": advantages,
            "masks": torch.from_numpy(np.array(self.masks)).float(),
        }
        return batch
