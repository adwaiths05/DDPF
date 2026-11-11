# src/baselines.py
"""Heuristic baseline policies for the evacuation environment.
Both policies return a NumPy array of length ``env.max_zones`` where each entry
is the index of the selected outgoing neighbor for the corresponding zone.
"""
import numpy as np
import networkx as nx
from .venue_generator import NODE_TYPE


class BasePolicy:
    """Simple interface shared by all baseline policies."""

    def __init__(self, env=None):
        self.env = env

    def select_action(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return an action array for the given observation.
        Sub-classes must implement ``_choose`` which receives the env instance.
        """
        action = self._choose()
        # Ensure actions respect the mask (fallback to first valid action if needed)
        for i, a in enumerate(action):
            if mask[i, a] == 0:
                # pick first valid neighbor
                valid = np.where(mask[i] == 1)[0]
                action[i] = valid[0] if valid.size > 0 else 0
        return action

    def predict(self, env) -> np.ndarray:
        """Convenience stateless interface: set env and return action array."""
        self.env = env
        mask = env.action_masks()
        obs = env._encode_observation()
        return self.select_action(obs, mask)

    def _choose(self) -> np.ndarray:
        raise NotImplementedError



class ShortestPathPolicy(BasePolicy):
    """Route each zone toward the nearest exit using Dijkstra shortest‑path distances.
    The policy ignores congestion and hazards – it is a pure shortest‑path heuristic.
    """

    def _choose(self) -> np.ndarray:
        env = self.env
        # Pre‑compute shortest travel‑time distance from every node to any exit
        distances = {}
        for node in env.zone_ids:
            min_dist = np.inf
            for exit_id in env.exit_ids:
                try:
                    d = nx.shortest_path_length(
                        env.venue.graph, source=node, target=exit_id, weight="travel_time"
                    )
                    if d < min_dist:
                        min_dist = d
                except nx.NetworkXNoPath:
                    continue
            distances[node] = min_dist
        # For each zone pick the neighbor that lies on a shortest‑path toward an exit
        actions = []
        for zid in env.zone_ids:
            neighs = env.adj.get(zid, [])
            if not neighs:
                actions.append(0)
                continue
            # Evaluate each neighbor's distance to an exit
            best_idx = 0
            best_val = np.inf
            for idx, (nbr, _) in enumerate(neighs):
                if distances.get(nbr, np.inf) < best_val:
                    best_val = distances[nbr]
                    best_idx = idx
            actions.append(best_idx)
        # Pad to max_zones
        pad = env.max_zones - len(env.zone_ids)
        if pad > 0:
            actions.extend([0] * pad)
        return np.array(actions, dtype=int)


class CongestionAwarePolicy(BasePolicy):
    """Weighted Dijkstra where edge weight = travel_time * (1 + congestion_factor).
    Congestion factor = occupancy / capacity of the edge (capped at 1).
    """

    def _choose(self) -> np.ndarray:
        env = self.env
        # Build a weight function that incorporates current congestion
        def edge_weight(u, v, data):
            base = data.get("travel_time", 1.0)
            cap = data.get("capacity", 1.0)
            cong = env.edge_congestion.get((u, v), 0) / cap if cap > 0 else 0.0
            factor = 1.0 + min(cong, 1.0)  # cap at 2x
            return base * factor

        actions = []
        for zid in env.zone_ids:
            neighs = env.adj.get(zid, [])
            if not neighs:
                actions.append(0)
                continue
            # Compute shortest path distances using the dynamic edge weights
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    env.venue.graph, source=zid, weight=edge_weight
                )
            except Exception:
                lengths = {}
            # Choose neighbor that leads to the smallest distance to any exit
            best_idx = 0
            best_val = np.inf
            for idx, (nbr, _) in enumerate(neighs):
                # distance from neighbor to nearest exit
                dist_to_exit = np.inf
                for exit_id in env.exit_ids:
                    if exit_id in lengths:
                        # distance from zid to exit passes through nbr automatically
                        # we approximate by length to exit from neighbor
                        try:
                            d = nx.shortest_path_length(
                                env.venue.graph,
                                source=nbr,
                                target=exit_id,
                                weight="travel_time",
                            )
                            if d < dist_to_exit:
                                dist_to_exit = d
                        except nx.NetworkXNoPath:
                            continue
                if dist_to_exit < best_val:
                    best_val = dist_to_exit
                    best_idx = idx
            actions.append(best_idx)
        # Pad
        pad = env.max_zones - len(env.zone_ids)
        if pad > 0:
            actions.extend([0] * pad)
        return np.array(actions, dtype=int)
