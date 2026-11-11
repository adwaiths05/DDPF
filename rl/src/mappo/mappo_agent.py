# src/mappo/mappo_agent.py
"""Hierarchical MAPPO agent implementation.
The agent maintains a separate policy network per role (Leader, Zone, Exit, Signage)
but shares a centralized critic for value estimation.  For simplicity we implement a
single ``PolicyNet`` that can be instantiated for any role; the role determines the
observation dimension.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .centralized_critic import CentralizedCritic
from .agent_roles import AgentRoles


class PolicyNet(nn.Module):
    """MLP policy for a given role.
    Returns logits over the role's discrete action space.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MAPPOAgent:
    def __init__(
        self,
        roles: AgentRoles,
        device: torch.device,
        leader_obs_dim: int,
        zone_obs_dim: int,
        exit_obs_dim: int,
        signage_obs_dim: int,
        max_zones: int,
        max_neighbors: int,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        critic_lr: float = 1e-3,
    ):
        self.device = device
        self.roles = roles
        self.max_zones = max_zones
        self.max_neighbors = max_neighbors
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # centralized critic: input is the concatenation of all per-zone obs only
        self.joint_dim = max_zones * zone_obs_dim

        # policies per role
        # leader_policy input = full joint obs (max_zones * zone_obs_dim)
        self.leader_policy = PolicyNet(self.joint_dim, output_dim=5).to(device)  # 5 high-level strategies
        self.zone_policy = PolicyNet(zone_obs_dim, output_dim=max_neighbors).to(device)
        self.exit_policy = PolicyNet(exit_obs_dim, output_dim=3).to(device)       # e.g. throttle levels
        self.signage_policy = PolicyNet(signage_obs_dim, output_dim=max_neighbors).to(device)

        self.critic = CentralizedCritic(self.joint_dim).to(device)

        # single optimizer for all params
        self.actor_params = (
            list(self.leader_policy.parameters())
            + list(self.zone_policy.parameters())
            + list(self.exit_policy.parameters())
            + list(self.signage_policy.parameters())
        )
        self.critic_params = list(self.critic.parameters())
        self.all_params = self.actor_params + self.critic_params
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor_params, "lr": lr},
                {"params": self.critic_params, "lr": critic_lr},
            ]
        )

        # trajectory buffers
        self.obs_buffer: list = []
        self.action_buffer: list = []
        self.logp_buffer: list = []
        self.mask_buffer: list = []
        self.reward_buffer: list = []
        self.done_buffer: list = []
        self.value_buffer: list = []
        self.joint_obs_buffer: list = []
        self.zone_obs_buffer: list = []   # store per-zone obs tensors for zone policy update

    def select_actions(self, per_agent_obs: dict, masks: dict):
        """Select actions for all agents given their observations and masks.
        Returns a dict mapping agent ids to actions, plus log‑probs and values.
        """
        actions: dict = {}
        logps: dict = {}
        values: dict = {}

        joint_obs = self._build_joint_obs(per_agent_obs)

        # Leader: uses joint obs as input
        leader_logits = self.leader_policy(joint_obs)
        leader_dist = torch.distributions.Categorical(logits=leader_logits)
        leader_action = leader_dist.sample()
        actions[-1] = int(leader_action.item())
        logps[-1] = leader_dist.log_prob(leader_action)
        values[-1] = self.critic(joint_obs)

        # Per-zone agents
        for zid, obs in per_agent_obs.items():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(masks[zid], dtype=torch.bool, device=self.device)
            logits = self.zone_policy(obs_t)
            masked_logits = logits.masked_fill(~mask, -1e10)
            dist = torch.distributions.Categorical(logits=masked_logits)
            act = dist.sample()
            actions[zid] = int(act.item())
            logps[zid] = dist.log_prob(act)
            values[zid] = self.critic(joint_obs)

        # Store step data
        self.obs_buffer.append(per_agent_obs)
        self.action_buffer.append(actions)
        self.logp_buffer.append(logps)
        self.mask_buffer.append([
            torch.as_tensor(
                masks.get(zid, np.ones(self.max_neighbors)),
                dtype=torch.bool,
                device=self.device,
            )
            for zid in range(self.max_zones)
        ])
        self.value_buffer.append(values)
        self.joint_obs_buffer.append(joint_obs.detach())
        self.zone_obs_buffer.append([
            torch.as_tensor(per_agent_obs[zid], dtype=torch.float32, device=self.device).detach()
            for zid in sorted(per_agent_obs)[:self.max_zones]
        ])
        return actions, logps, values

    def _build_joint_obs(self, per_agent_obs: dict) -> torch.Tensor:
        """Build the fixed-size centralized-critic input."""
        sorted_ids = sorted(per_agent_obs.keys())
        zone_tensors = [torch.as_tensor(per_agent_obs[i], dtype=torch.float32, device=self.device) for i in sorted_ids]

        # Pad to max_zones if fewer zones than expected
        zone_obs_dim = zone_tensors[0].shape[-1] if zone_tensors else 1
        while len(zone_tensors) < self.max_zones:
            zone_tensors.append(torch.zeros(zone_obs_dim, device=self.device))

        return torch.cat(zone_tensors[:self.max_zones], dim=-1)

    @torch.no_grad()
    def estimate_value(self, per_agent_obs: dict) -> float:
        """Estimate the value of the state at a rollout boundary."""
        joint_obs = self._build_joint_obs(per_agent_obs)
        return float(self.critic(joint_obs).item())

    def store_reward(self, reward: float, done: bool):
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)

    def update(self, epochs: int = 4, batch_size: int = 64, last_value: float = 0.0):
        """Perform MAPPO update across all policies using the centralized critic."""
        N = len(self.reward_buffer)
        if N == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Value estimates: average per-agent values for each step
        values_t = torch.stack([
            torch.stack(list(v.values())).mean()
            for v in self.value_buffer
        ]).detach()

        # Compute generalized advantages, resetting the trace at episode ends.
        rewards_t = torch.tensor(self.reward_buffer, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(self.done_buffer, dtype=torch.float32, device=self.device)
        advantages = torch.zeros(N, dtype=torch.float32, device=self.device)
        gae = torch.tensor(0.0, device=self.device)
        for index in reversed(range(N)):
            next_value = values_t[index + 1] if index + 1 < N else torch.tensor(last_value, device=self.device)
            delta = rewards_t[index] + self.gamma * (1.0 - dones_t[index]) * next_value - values_t[index]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones_t[index]) * gae
            advantages[index] = gae
        returns_t = advantages + values_t
        advs_t = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # Stack joint observations into a tensor
        joint_obs_t = torch.stack(self.joint_obs_buffer)  # (N, joint_dim)

        # Stack zone-level observations: (N, max_zones, zone_obs_dim)
        zone_obs_t = torch.stack([
            torch.stack(obs_list)
            for obs_list in self.zone_obs_buffer
        ])  # (N, max_zones, zone_obs_dim)

        # Old log-probs (leader only for ratio; zone handled separately)
        old_leader_logps = torch.stack([lp[-1] for lp in self.logp_buffer]).detach()
        old_leader_acts = torch.tensor(
            [acts[-1] for acts in self.action_buffer],
            dtype=torch.long,
            device=self.device,
        )
        # Old zone log-probs: shape (N, max_zones)
        old_zone_logps = torch.stack([
            torch.stack([lp.get(zid, torch.tensor(0.0, device=self.device)) for zid in range(self.max_zones)])
            for lp in self.logp_buffer
        ]).detach()

        # Old zone actions: shape (N, max_zones)
        old_zone_acts = torch.tensor([
            [acts.get(zid, 0) for zid in range(self.max_zones)]
            for acts in self.action_buffer
        ], dtype=torch.long, device=self.device)
        zone_masks = torch.stack([
            torch.stack(mask_list)
            for mask_list in self.mask_buffer
        ])

        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        minibatch_count = 0

        for epoch in range(epochs):
            idxs = torch.randperm(N)
            for start in range(0, N, batch_size):
                batch_idx = idxs[start:start + batch_size].tolist()  # convert to Python list
                adv_batch = advs_t[batch_idx]
                ret_batch = returns_t[batch_idx]
                jt_batch = joint_obs_t[batch_idx]        # (B, joint_dim)
                zo_batch = zone_obs_t[batch_idx]         # (B, max_zones, zone_obs_dim)
                mask_batch = zone_masks[batch_idx]

                # ---- Leader policy loss ----
                leader_logits_new = self.leader_policy(jt_batch)
                leader_dist_new = torch.distributions.Categorical(logits=leader_logits_new)
                leader_acts_batch = old_leader_acts[batch_idx]
                new_lp_leader = leader_dist_new.log_prob(leader_acts_batch)
                old_lp_leader = old_leader_logps[batch_idx]
                ratio_leader = torch.exp(new_lp_leader - old_lp_leader)
                surr1 = ratio_leader * adv_batch
                surr2 = torch.clamp(ratio_leader, 1 - self.clip_eps, 1 + self.clip_eps) * adv_batch
                leader_loss = -torch.min(surr1, surr2).mean()
                leader_entropy = leader_dist_new.entropy().mean()

                # ---- Zone policy loss ----
                # Reshape zone obs: (B*max_zones, zone_obs_dim)
                B = zo_batch.shape[0]
                zo_flat = zo_batch.view(B * self.max_zones, -1)
                zone_logits_new = self.zone_policy(zo_flat).view(B, self.max_zones, self.max_neighbors)
                zone_logits_new = zone_logits_new.masked_fill(~mask_batch, -1e10)
                zone_dist_new = torch.distributions.Categorical(logits=zone_logits_new)
                zone_acts_batch = old_zone_acts[batch_idx]   # (B, max_zones)
                new_lp_zone = zone_dist_new.log_prob(zone_acts_batch)  # (B, max_zones)
                old_lp_zone = old_zone_logps[batch_idx]     # (B, max_zones)
                ratio_zone = torch.exp(new_lp_zone - old_lp_zone)
                adv_zone = adv_batch.unsqueeze(-1).expand(-1, self.max_zones)
                surr1z = ratio_zone * adv_zone
                surr2z = torch.clamp(ratio_zone, 1 - self.clip_eps, 1 + self.clip_eps) * adv_zone
                zone_loss = -torch.min(surr1z, surr2z).mean()
                zone_entropy = zone_dist_new.entropy().mean()

                # ---- Value loss ----
                values_pred = self.critic(jt_batch).squeeze(-1)  # (B,)
                value_loss = F.smooth_l1_loss(values_pred, ret_batch)

                # The current environment executes only zone actions, so the
                # sampled leader action has no causal effect on the reward.
                policy_loss = zone_loss
                entropy = zone_entropy
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.all_params, self.max_grad_norm)
                self.optimizer.step()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_total += entropy.item()
                minibatch_count += 1

        # Clear all buffers
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.logp_buffer.clear()
        self.mask_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()
        self.value_buffer.clear()
        self.joint_obs_buffer.clear()
        self.zone_obs_buffer.clear()

        return {
            "policy_loss": policy_loss_total / minibatch_count,
            "value_loss": value_loss_total / minibatch_count,
            "entropy": entropy_total / minibatch_count,
        }
