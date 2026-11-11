# src/ppo/ppo_agent.py
"""PPO agent encapsulating policy, value network, and update logic.
The agent works with the RolloutBuffer defined in ``rollout_buffer.py``.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from .actor_critic import Actor, Critic
from .rollout_buffer import RolloutBuffer


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        max_zones: int,
        max_neighbors: int,
        device: torch.device,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.device = device
        self.actor = Actor(obs_dim, max_zones, max_neighbors).to(device)
        self.critic = Critic(obs_dim).to(device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        # buffer will be created per rollout cycle
        self.buffer = RolloutBuffer(obs_dim, max_zones, max_neighbors)
        self.max_zones = max_zones
        self.max_neighbors = max_neighbors

    def select_action(self, obs: torch.Tensor, action_mask: torch.Tensor, deterministic: bool = False):
        """Return action, log_prob and value for a single observation.
        ``action_mask`` shape: (max_zones, max_neighbors) with 1 for valid actions.
        """
        obs = obs.to(self.device)
        logits_flat = self.actor(obs)
        logits = logits_flat.view(self.max_zones, self.max_neighbors)
        # mask invalid actions by large negative value
        masked_logits = logits.masked_fill(action_mask == 0, -1e10)
        dist = torch.distributions.Categorical(logits=masked_logits)
        if deterministic:
            action = torch.argmax(masked_logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(obs)
        return action.cpu().numpy(), log_prob, value

    def update(self, epochs: int = 4, batch_size: int = 64):
        """Perform PPO update.
        If the buffer is empty (e.g., before enough steps have been collected),
        skip the update and return zeroed metrics.
        """
        if len(self.buffer.observations) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        batch = self.buffer.get_batch()
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)  # shape (N, max_zones)
        old_log_probs = batch["log_probs"].to(self.device)
        returns = batch["returns"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        masks = batch["masks"].to(self.device)

        dataset_size = obs.shape[0]
        for epoch in range(epochs):
            # simple shuffling by random indices
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                obs_batch = obs[idx]
                act_batch = actions[idx]
                old_lp_batch = old_log_probs[idx].detach()
                ret_batch = returns[idx]
                adv_batch = advantages[idx]
                mask_batch = masks[idx]

                # forward pass
                logits_flat = self.actor(obs_batch)
                logits = logits_flat.view(-1, self.max_zones, self.max_neighbors)
                masked_logits = logits.masked_fill(mask_batch == 0, -1e10)
                dist = torch.distributions.Categorical(logits=masked_logits)
                new_log_probs = dist.log_prob(act_batch)
                entropy = dist.entropy().mean()
                # obtain value estimates (scalar per observation)
                values = self.critic(obs_batch).squeeze(-1)  # shape (batch,)

                # ratio for PPO clipping
                ratio = torch.exp(new_log_probs - old_lp_batch)
                # broadcast advantage to per-zone shape
                adv_expanded = adv_batch.unsqueeze(-1).expand(-1, self.max_zones)
                surr1 = ratio * adv_expanded
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_expanded
                policy_loss = -torch.min(surr1, surr2).mean()
                # value loss using returns (scalar) vs values (scalar)
                value_loss = nn.functional.mse_loss(values, ret_batch)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.optimizer.step()

        # clear buffer after update
        self.buffer.clear()
        # return metrics for logging (optional)
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
