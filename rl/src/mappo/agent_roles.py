# src/mappo/agent_roles.py
"""Definition of agent roles for hierarchical MAPPO.
We model four roles (Leader, Zone, Exit, Signage) as simple data classes.
Each role knows how to slice the global observation vector into its local view.
The observation vector is produced by ``src/observation_encoder.py`` and has the
following layout (see that module for details):

- Per‑zone features (8 values) for each of ``max_zones`` zones.
- Neighbor‑level features (capacity, congestion) for each zone and each of
  ``max_neighbors`` neighbor slots.
- Global features (5 values).

Roles extract the relevant slices:
* Leader – all global features + aggregated zone statistics.
* Zone – its own per‑zone features + neighbor features for that zone.
* Exit – global features + exit‑specific flags (derived from zone IDs).
* Signage – same as Zone but may include an extra "signage" flag (placeholder).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class AgentRoles:
    max_zones: int
    max_neighbors: int

    def leader_observation(self, obs: np.ndarray) -> np.ndarray:
        """Leader sees the entire observation plus aggregated summaries.
        For simplicity we just return the full vector.
        """
        return obs.copy()

    def zone_observation(self, obs: np.ndarray, zone_idx: int) -> np.ndarray:
        """Extract observation for a single zone (index 0‑based)."""
        # per‑zone features occupy the first ``max_zones * 8`` entries
        start = zone_idx * 8
        per_zone = obs[start : start + 8]
        # neighbor features follow after all per‑zone blocks
        neigh_start = self.max_zones * 8 + zone_idx * self.max_neighbors * 2
        neigh_feats = obs[neigh_start : neigh_start + self.max_neighbors * 2]
        return np.concatenate([per_zone, neigh_feats])

    def exit_observation(self, obs: np.ndarray, exit_idx: int) -> np.ndarray:
        """For now exit observation is just the global tail (last 5 values)."""
        return obs[-5:]

    def signage_observation(self, obs: np.ndarray, zone_idx: int) -> np.ndarray:
        """Same as zone observation – placeholder for future signage features."""
        return self.zone_observation(obs, zone_idx)
