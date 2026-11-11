# src/mappo/multi_agent_env.py
"""Multi‑agent wrapper around the single‑agent ``EvacuationEnv``.
The environment is exposed as a list of agents, each with its own observation
and action space.  For simplicity we treat each zone as an independent agent.
The wrapper returns a tuple ``(obs_dict, reward, done, info)`` where ``obs_dict``
maps agent IDs (integers) to their respective observations.
"""
import numpy as np
from gymnasium import spaces
from ..evacuation_env import EvacuationEnv
from ..observation_encoder import encode_observation


class MultiAgentEnv:
    def __init__(self, venue, seed: int = 0):
        self.base_env = EvacuationEnv(venue, seed=seed)
        self.max_zones = self.base_env.max_zones
        self.max_neighbors = self.base_env.max_neighbors
        # each agent corresponds to a zone index
        self.agent_ids = list(range(self.max_zones))
        # observation space for a single agent (per‑zone slice)
        per_zone_dim = 8 + self.max_neighbors * 2  # matches AgentRoles.zone_observation
        self.observation_spaces = {i: spaces.Box(low=-np.inf, high=np.inf, shape=(per_zone_dim,), dtype=np.float32) for i in self.agent_ids}
        # action space per zone: choose one neighbor (or stay) -> max_neighbors actions
        self.action_spaces = {i: spaces.Discrete(self.max_neighbors) for i in self.agent_ids}

    def reset(self, seed: int = None):
        obs, _ = self.base_env.reset(seed=seed)
        full_obs = encode_observation(self.base_env)
        # slice per zone using the same logic as in AgentRoles.zone_observation
        per_zone_obs = {}
        for zid in self.agent_ids:
            start = zid * 8
            zone_feat = full_obs[start:start+8]
            neigh_start = self.max_zones * 8 + zid * self.max_neighbors * 2
            neigh_feat = full_obs[neigh_start:neigh_start + self.max_neighbors * 2]
            per_zone_obs[zid] = np.concatenate([zone_feat, neigh_feat]).astype(np.float32)
        return per_zone_obs, {}

    def step(self, actions: dict):
        """Accept a dict mapping agent IDs to chosen neighbor index.
        The underlying ``EvacuationEnv`` expects a full action vector of length
        ``max_zones`` where each entry is the neighbor index for that zone.
        Returns (obs_dict, reward_dict, done, trunc, info) — same shape as Gymnasium.
        """
        # Build full action vector
        full_action = [0] * self.max_zones
        for zid, act in actions.items():
            full_action[zid] = int(act)
        obs, reward, done, trunc, info = self.base_env.step(full_action)
        # recompute per-agent observations
        full_obs = encode_observation(self.base_env)
        per_zone_obs = {}
        for zid in self.agent_ids:
            start = zid * 8
            zone_feat = full_obs[start:start+8]
            neigh_start = self.max_zones * 8 + zid * self.max_neighbors * 2
            neigh_feat = full_obs[neigh_start:neigh_start + self.max_neighbors * 2]
            per_zone_obs[zid] = np.concatenate([zone_feat, neigh_feat]).astype(np.float32)
        # reward is shared among agents
        reward_dict = {zid: reward for zid in self.agent_ids}
        return per_zone_obs, reward_dict, done, trunc, info

    def action_masks(self):
        """Return a dict of action masks for each agent.
        ``1`` indicates the neighbor is reachable; ``0`` otherwise.
        """
        masks = {}
        base_masks = self.base_env.action_masks()  # shape (max_zones, max_neighbors)
        for zid in self.agent_ids:
            masks[zid] = base_masks[zid].astype(np.int32)
        return masks

    def render(self, mode="human"):
        return self.base_env.render(mode=mode)

    def close(self):
        self.base_env.close()
