# src/evacuation_env.py
"""Gymnasium environment for synthetic crowd evacuation.

The environment models aggregate crowd flow on a directed venue graph. At each
step the policy provides a routing decision for each *decision zone* – i.e. a
choice of which neighboring node occupants should attempt to move toward.

Key design points
-----------------
* **State** – occupancy per zone, edge congestion, exit capacities, blocked
  flags, elapsed time, and total evacuated count.
* **Action space** – ``gymnasium.spaces.MultiDiscrete`` where each entry
  corresponds to a zone and encodes the index of the chosen outgoing edge.
* **Action masks** – ``action_masks()`` returns a binary matrix indicating which
  neighbor indices are valid for each zone (blocked edges/exits are excluded).
* **Population conservation** – total number of people is invariant except for
  those that have evacuated.
* **Termination** – episode ends when all evacuable occupants have exited or a
  maximum step limit is reached.

The implementation is deliberately simple (aggregate flow) to keep CPU
training fast while still providing a realistic enough challenge for PPO.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import networkx as nx

from .venue_generator import VenueGraph, NODE_TYPE, CAPACITY, OCCUPANCY, IS_BLOCKED, IS_HAZARD, RISK, TRAVEL_TIME, EDGE_CAPACITY, POS


class EvacuationEnv(gym.Env):
    """A single‑agent evacuation environment compatible with Gymnasium.

    Observation and action spaces are fixed‑size to simplify the initial PPO
    implementation. Variable‑size graphs are padded to a configurable maximum
    number of zones (``max_zones``) and a maximum number of outgoing edges per
    zone (``max_neighbors``).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        venue: VenueGraph,
        max_zones: int = 15,
        max_neighbors: int = 6,
        max_steps: int = 100,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.venue = venue
        self.max_zones = max_zones
        self.max_neighbors = max_neighbors
        self.max_steps = max_steps
        self.current_step = 0
        self.rng = random.Random(seed)

        # ----- Prepare graph topology for fast lookup -----
        self.zone_ids = self.venue.zones
        self.exit_ids = self.venue.exits
        self.num_zones = len(self.zone_ids)
        self.num_exits = len(self.exit_ids)

        # Mapping from zone -> list of (neighbor_id, edge_data)
        self.adj: Dict[int, List[Tuple[int, dict]]] = {
            z: [(nbr, self.venue.graph.edges[(z, nbr)]) for nbr in self.venue.graph.successors(z)] for z in self.zone_ids
        }

        # ----- Define observation & action spaces -----
        # Observation: flattened vector – see ``_encode_observation``
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.max_zones * 8 + self.max_zones * self.max_neighbors * 2 + 5,),
            dtype=np.float32,
        )
        # Action: for each zone, an integer selecting one of its outgoing neighbors (0..max_neighbors-1)
        self.action_space = gym.spaces.MultiDiscrete([self.max_neighbors] * self.max_zones)

        # Internal state containers
        self.zone_occupancy: Dict[int, int] = {}
        self.edge_congestion: Dict[Tuple[int, int], int] = {}
        self.total_evacuated: int = 0
        self.initial_population: int = 0
        self.blocked_edges: List[Tuple[int, int]] = []
        self.unavailable_exits: List[int] = []
        self.hazard_zones: List[int] = []

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    def _reset_state(self) -> None:
        """Initialize occupancy, congestion, and incident flags.

        For simplicity we use a uniform random occupancy respecting zone capacity.
        """
        self.zone_occupancy = {}
        self.edge_congestion = {(u, v): 0 for u, v in self.venue.graph.edges}
        self.total_evacuated = 0
        self._prev_evacuated = 0
        self.current_step = 0
        self.blocked_edges = []
        self.unavailable_exits = []
        self.hazard_zones = []
        self._last_reward_components = {}
        # Random occupancy respecting capacity (max 80% of capacity)
        for zid in self.zone_ids:
            cap = self.venue.graph.nodes[zid][CAPACITY]
            occ = int(cap * self.rng.uniform(0.2, 0.8))
            self.zone_occupancy[zid] = occ
        self.initial_population = sum(self.zone_occupancy.values())

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng.seed(seed)
        self._reset_state()
        obs = self._encode_observation()
        info = {"initial_population": self.initial_population}
        return obs, info

    # ---------------------------------------------------------------------
    # Observation encoding (fixed‑size vector)
    # ---------------------------------------------------------------------
    def _encode_observation(self) -> np.ndarray:
        """Encode the current environment state into a flat numpy array.

        The layout mirrors the definition in ``observation_encoder.py`` – this
        method is kept here for a minimal first version.
        """
        vec = []
        # Per‑zone features (8 numbers per zone)
        for zid in self.zone_ids:
            occ = self.zone_occupancy.get(zid, 0)
            cap = self.venue.graph.nodes[zid][CAPACITY]
            norm_occ = occ / cap if cap > 0 else 0.0
            risk = self.venue.graph.nodes[zid][RISK]
            is_hazard = 1.0 if self.venue.graph.nodes[zid][IS_HAZARD] else 0.0
            is_exit = 1.0 if self.venue.graph.nodes[zid][NODE_TYPE] == "exit" else 0.0
            min_dist = self._min_distance_to_exit(zid)
            n_neighbors = len(self.adj.get(zid, [])) / self.max_neighbors
            vec.extend([occ, norm_occ, cap, risk, is_hazard, is_exit, min_dist, n_neighbors])
        # Pad zones up to max_zones
        pad_zones = self.max_zones - self.num_zones
        if pad_zones > 0:
            vec.extend([0.0] * 8 * pad_zones)
        # Neighbor‑level features: for each zone, for each possible neighbor slot
        for zid in self.zone_ids:
            neighs = self.adj.get(zid, [])
            for slot in range(self.max_neighbors):
                if slot < len(neighs):
                    nbr, edge = neighs[slot]
                    cap = edge[EDGE_CAPACITY]
                    cong = self.edge_congestion.get((zid, nbr), 0) / cap if cap > 0 else 0.0
                else:
                    cap = 0.0
                    cong = 0.0
                vec.extend([cap, cong])
        # Pad remaining zones' neighbor slots
        if pad_zones > 0:
            vec.extend([0.0] * self.max_neighbors * 2 * pad_zones)
        # Global features (5 numbers)
        remaining = self.initial_population - self.total_evacuated
        frac_evacuated = self.total_evacuated / self.initial_population if self.initial_population > 0 else 0.0
        time_norm = self.current_step / self.max_steps
        n_blocked = len(self.blocked_edges) / max(1, self.venue.graph.number_of_edges())
        n_unavail = len(self.unavailable_exits) / max(1, self.num_exits)
        vec.extend([remaining, frac_evacuated, time_norm, n_blocked, n_unavail])
        return np.array(vec, dtype=np.float32)

    def _min_distance_to_exit(self, node_id: int) -> float:
        """Shortest‑path travel‑time distance to any exit (ignoring blockages)."""
        distances = []
        for exit_id in self.exit_ids:
            try:
                length = nx.shortest_path_length(
                    self.venue.graph,
                    source=node_id,
                    target=exit_id,
                    weight=TRAVEL_TIME,
                )
                distances.append(length)
            except nx.NetworkXNoPath:
                continue
        if not distances:
            return 0.0
        # Normalise by a heuristic max distance (e.g., 20 steps)
        return min(distances) / 20.0

    # ---------------------------------------------------------------------
    # Action handling
    # ---------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """Return a binary mask of shape (max_zones, max_neighbors).

        ``1`` indicates the neighbor index is a valid routing choice for the zone.
        """
        mask = np.zeros((self.max_zones, self.max_neighbors), dtype=np.int8)
        for i, zid in enumerate(self.zone_ids):
            neighs = self.adj.get(zid, [])
            for j in range(min(len(neighs), self.max_neighbors)):
                nbr, edge = neighs[j]
                if edge[IS_BLOCKED]:
                    continue
                if self.venue.graph.nodes[nbr][NODE_TYPE] == "exit" and nbr in self.unavailable_exits:
                    continue
                mask[i, j] = 1
        return mask

    # ---------------------------------------------------------------------
    # Step logic
    # ---------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """Execute one simulation step.

        Parameters
        ----------
        action: np.ndarray of shape (max_zones,)
            For each zone the index of the chosen outgoing neighbor (0‑based).
        """
        assert isinstance(action, (list, np.ndarray)), "Action must be array‑like"
        action = np.asarray(action, dtype=int)
        # Clip actions to valid range
        action = np.clip(action, 0, self.max_neighbors - 1)
        invalid_actions = 0
        # Determine attempted flows per edge
        attempted_flow: Dict[Tuple[int, int], int] = {}
        for i, zid in enumerate(self.zone_ids):
            occ = self.zone_occupancy.get(zid, 0)
            if occ == 0:
                continue
            neighs = self.adj.get(zid, [])
            if not neighs:
                continue
            neigh_idx = action[i]
            if neigh_idx >= len(neighs):
                invalid_actions += 1
                continue
            nbr, edge = neighs[neigh_idx]
            if edge[IS_BLOCKED]:
                invalid_actions += 1
                continue
            if self.venue.graph.nodes[nbr][NODE_TYPE] == "exit" and nbr in self.unavailable_exits:
                invalid_actions += 1
                continue
            # Move all occupants toward chosen neighbour (aggregate flow)
            attempted_flow[(zid, nbr)] = attempted_flow.get((zid, nbr), 0) + occ
            # After decision, zone is emptied; overflow handled later
            self.zone_occupancy[zid] = 0
        # Apply capacities and update congestion counters
        for (src, dst), people in attempted_flow.items():
            cap = self.venue.graph.edges[(src, dst)][EDGE_CAPACITY]
            moved = min(people, cap)
            overflow = people - moved
            if overflow > 0:
                self.zone_occupancy[src] = self.zone_occupancy.get(src, 0) + overflow
            if self.venue.graph.nodes[dst][NODE_TYPE] == "exit":
                self.total_evacuated += moved
            else:
                self.zone_occupancy[dst] = self.zone_occupancy.get(dst, 0) + moved
            self.edge_congestion[(src, dst)] = people
        self.current_step += 1
        success = self.total_evacuated >= self.initial_population
        timeout = self.current_step >= self.max_steps
        done = success or timeout     # terminated
        trunc = timeout and not success  # truncated (timeout without completion)
        reward = self._compute_reward(invalid_actions)
        self._prev_evacuated = self.total_evacuated
        obs = self._encode_observation()
        info = {
            "step": self.current_step,
            "evacuated": self.total_evacuated,
            "remaining": self.initial_population - self.total_evacuated,
            "invalid_actions": invalid_actions,
            "reward_components": self._last_reward_components,
        }
        return obs, reward, done, trunc, info

    # ---------------------------------------------------------------------
    # Reward function (simple version; later a dedicated module can replace this)
    # ---------------------------------------------------------------------
    def _compute_reward(self, invalid_actions: int) -> float:
        """Compute a decomposed reward and store components for ``info``.

        Uses incremental progress (delta evacuated) instead of cumulative
        fraction to avoid rewarding stagnation.
        """
        pop = self.initial_population if self.initial_population > 0 else 1.0

        # Incremental progress – only reward NEW evacuations this step
        delta_evacuated = self.total_evacuated - self._prev_evacuated
        w_progress = 20.0
        progress = w_progress * (delta_evacuated / pop)

        # Stagnation penalty – penalise steps where no one was evacuated
        w_stagnation = -1.0
        stagnation = w_stagnation if delta_evacuated == 0 else 0.0

        # Congestion penalty – average occupancy / capacity across zones
        w_cong = -0.5
        congestion_vals = []
        for zid in self.zone_ids:
            occ = self.zone_occupancy.get(zid, 0)
            cap = self.venue.graph.nodes[zid]["capacity"]
            if cap > 0:
                congestion_vals.append(occ / cap)
        avg_cong = sum(congestion_vals) / len(congestion_vals) if congestion_vals else 0.0
        congestion_pen = w_cong * avg_cong

        # Hazard exposure – people in hazardous zones
        w_hazard = -1.0
        hazard_people = sum(self.zone_occupancy.get(zid, 0) for zid in self.hazard_zones)
        hazard_pen = w_hazard * (hazard_people / pop)

        # Invalid action penalty
        w_invalid = -0.5
        invalid_pen = w_invalid * invalid_actions

        # Time penalty – constant per step to encourage speed
        w_time = -0.1
        time_pen = w_time

        # Completion bonus / failure penalty
        w_completion = 100.0
        completion_bonus = w_completion if self.total_evacuated >= self.initial_population else 0.0

        w_failure = -10.0
        failure_pen = w_failure if (self.current_step >= self.max_steps and self.total_evacuated < self.initial_population) else 0.0

        total = progress + stagnation + congestion_pen + hazard_pen + invalid_pen + time_pen + completion_bonus + failure_pen
        self._last_reward_components = {
            "progress": progress,
            "stagnation": stagnation,
            "congestion_pen": congestion_pen,
            "hazard_pen": hazard_pen,
            "invalid_pen": invalid_pen,
            "time_pen": time_pen,
            "completion_bonus": completion_bonus,
            "failure_pen": failure_pen,
        }
        return total

    # ---------------------------------------------------------------------
    # Rendering – simple matplotlib plot of the graph with node colors reflecting occupancy
    # ---------------------------------------------------------------------
    def render(self, mode: str = "human"):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        # Update node colors based on occupancy ratio
        colors = []
        for nid in self.venue.graph.nodes:
            if self.venue.graph.nodes[nid][NODE_TYPE] == "exit":
                colors.append("gold")
            else:
                occ = self.zone_occupancy.get(nid, 0)
                cap = self.venue.graph.nodes[nid][CAPACITY]
                ratio = occ / cap if cap > 0 else 0.0
                colors.append((ratio, 1 - ratio, 0))
        pos = {n: d["pos"] for n, d in self.venue.graph.nodes(data=True)}
        nx.draw(self.venue.graph, pos, node_color=colors, with_labels=True, ax=ax, node_size=500)
        ax.set_title(f"Step {self.current_step} – Evacuated: {self.total_evacuated}/{self.initial_population}")
        if mode == "human":
            plt.show()
        else:
            canvas = fig.canvas
            canvas.draw()
            image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image

    def close(self):
        pass

# Simple demo for manual testing
if __name__ == "__main__":
    from .venue_generator import VenueGenerator

    vg = VenueGenerator.generate_random_venue(seed=202)
    env = EvacuationEnv(vg, seed=202)
    obs, info = env.reset(seed=202)
    done = False
    while not done:
        masks = env.action_masks()
        actions = []
        for i in range(env.max_zones):
            valid = np.where(masks[i] == 1)[0]
            if len(valid) == 0:
                actions.append(0)
            else:
                actions.append(int(env.rng.choice(valid)))
        obs, reward, done, trunc, info = env.step(actions)
        print(f"Step {env.current_step}, reward {reward:.2f}, evacuated {info['evacuated']}")
    env.render()
