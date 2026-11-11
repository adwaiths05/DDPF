# src/integration_adapter.py
"""Adapter layer between the RL module and the broader crowd‑risk system.
The adapter exposes a clean API that the higher‑level pipeline can call:

* ``initialize(seed)`` – creates a venue, environment and loads a trained policy.
* ``step(state)`` – given the current crowd state (positions, densities, etc.)
  returns the next evacuation actions for each zone.
* ``reset()`` – resets the environment for a new episode.
* ``close()`` – cleans up resources.

The adapter hides the internal details of whether the underlying policy is PPO
or MAPPO.  It also performs any necessary observation encoding and reward
computation so that downstream modules can operate on high‑level tensors.
"""
import os
import numpy as np
import torch
from .venue_generator import VenueGenerator
from .evacuation_env import EvacuationEnv
from .observation_encoder import encode_observation
from .reward_function import compute_reward
from .ppo.actor_critic import Actor
from .ppo.trainer import train_ppo
from .mappo.mappo_agent import MAPPOAgent
from .mappo.trainer import train_mappo


class IntegrationAdapter:
    def __init__(self, policy_type: str = "ppo", checkpoint_path: str = None, device: str = "cpu"):
        self.policy_type = policy_type.lower()
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.env: EvacuationEnv | None = None
        self.policy = None
        self.checkpoint_path = checkpoint_path

    def initialize(self, seed: int = 0):
        """Create a fresh venue, environment and load the policy.
        Returns the initial observation tensor (encoded).
        """
        # generate venue – use random; fallback to grid inside Env if needed
        try:
            venue = VenueGenerator.generate_random_venue(seed=seed)
        except Exception:
            venue = VenueGenerator.generate_grid_venue(seed=seed)
        self.env = EvacuationEnv(venue, seed=seed)
        obs, _ = self.env.reset(seed=seed)
        # load policy based on type
        if self.policy_type == "ppo":
            # infer dimensions from env
            obs_dim = obs.shape[0]
            actor = Actor(obs_dim, self.env.max_zones, self.env.max_neighbors).to(self.device)
            if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                actor.load_state_dict(ckpt.get("actor", {}))
            self.policy = actor
        elif self.policy_type == "mappo":
            from .mappo.agent_roles import AgentRoles
            roles = AgentRoles(self.env.max_zones, self.env.max_neighbors)
            # dimension placeholders – must match training configuration
            leader_obs_dim = encode_observation(self.env).shape[0]
            zone_obs_dim = 8 + self.env.max_neighbors * 2
            exit_obs_dim = 5
            signage_obs_dim = zone_obs_dim
            self.policy = MAPPOAgent(
                roles=roles,
                device=self.device,
                leader_obs_dim=leader_obs_dim,
                zone_obs_dim=zone_obs_dim,
                exit_obs_dim=exit_obs_dim,
                signage_obs_dim=signage_obs_dim,
                max_zones=self.env.max_zones,
                max_neighbors=self.env.max_neighbors,
            )
            if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                self.policy.leader_policy.load_state_dict(ckpt.get("leader_policy", {}))
                self.policy.zone_policy.load_state_dict(ckpt.get("zone_policy", {}))
                self.policy.critic.load_state_dict(ckpt.get("critic", {}))
        else:
            raise ValueError(f"Unsupported policy_type: {self.policy_type}")
        # return encoded observation for downstream use
        return encode_observation(self.env)

    def step(self, external_state=None):
        """Perform a single environment step using the loaded policy.
        ``external_state`` is ignored for now – the adapter works directly with the
        internal ``EvacuationEnv``.  It returns the action dict (zone_id → neighbor
        index) and the encoded observation after the step.
        """
        if self.env is None or self.policy is None:
            raise RuntimeError("Adapter not initialized – call initialize() first.")
        # encode current observation
        obs_enc = encode_observation(self.env)
        if self.policy_type == "ppo":
            obs_tensor = torch.from_numpy(obs_enc).float().to(self.device)
            logits = self.policy(obs_tensor)
            logits = logits.view(self.env.max_zones, self.env.max_neighbors)
            mask = torch.from_numpy(self.env.action_masks()).to(self.device)
            masked = logits.masked_fill(mask == 0, -1e10)
            dist = torch.distributions.Categorical(logits=masked)
            action = dist.sample().cpu().numpy()
            # step env with full action vector
            next_obs, reward, done, trunc, info = self.env.step(action)
        else:  # mappo
            # build per‑agent observation dict and masks
            full_obs = encode_observation(self.env)
            per_agent_obs = {}
            for zid in range(self.env.max_zones):
                start = zid * 8
                per = full_obs[start:start+8]
                neigh_start = self.env.max_zones * 8 + zid * self.env.max_neighbors * 2
                neigh = full_obs[neigh_start:neigh_start + self.env.max_neighbors * 2]
                per_agent_obs[zid] = torch.from_numpy(np.concatenate([per, neigh])).float().to(self.device)
            masks = {zid: torch.from_numpy(self.env.action_masks()[zid]).to(self.device) for zid in range(self.env.max_zones)}
            actions, _, _ = self.policy.select_actions(per_agent_obs, masks)
            # strip leader action
            zone_actions = {aid: act for aid, act in actions.items() if aid != -1}
            # convert zone_actions dict to list for env.step
            action_list = [zone_actions.get(zid, 0) for zid in range(self.env.max_zones)]
            next_obs, reward, done, trunc, info = self.env.step(action_list)
        # compute reward components (optional, for logging)
        total_reward, components = compute_reward(self.env, info.get("invalid_actions", 0))
        return {
            "action": action if self.policy_type == "ppo" else zone_actions,
            "reward": total_reward,
            "done": done or trunc,
            "observation": encode_observation(self.env),
        }

    def reset(self, seed: int = None):
        if self.env is None:
            raise RuntimeError("Adapter not initialized – call initialize() first.")
        obs, _ = self.env.reset(seed=seed)
        return encode_observation(self.env)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        self.policy = None
