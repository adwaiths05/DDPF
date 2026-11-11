# src/observation_encoder.py
"""Observation encoder for the evacuation environment.
The encoder produces a fixed‑size flat NumPy vector that can be used by any RL
algorithm. It mirrors the logic previously embedded in ``evacuation_env.py``
but is now a standalone utility.
"""
import numpy as np
from .venue_generator import (
    CAPACITY,
    OCCUPANCY,
    RISK,
    IS_HAZARD,
    NODE_TYPE,
    EDGE_CAPACITY,
    IS_BLOCKED,
    POS,
)


def encode_observation(env, max_zones: int = 15, max_neighbors: int = 6) -> np.ndarray:
    """Encode the current ``EvacuationEnv`` state into a 1‑D float32 array.

    Parameters
    ----------
    env: EvacuationEnv instance – must expose the same attributes as the original
        implementation (e.g., ``zone_occupancy``, ``edge_congestion``, etc.).
    max_zones: maximum number of zones the model expects.
    max_neighbors: maximum number of outgoing edges per zone.

    Returns
    -------
    np.ndarray
        Flat observation vector of shape ``(max_zones * 8 + max_zones * max_neighbors * 2 + 5,)``.
    """
    # Per‑zone features (8 per zone)
    vec = []
    for zid in env.zone_ids:
        occ = env.zone_occupancy.get(zid, 0)
        cap = env.venue.graph.nodes[zid][CAPACITY]
        norm_occ = occ / cap if cap > 0 else 0.0
        risk = env.venue.graph.nodes[zid][RISK]
        is_hazard = 1.0 if env.venue.graph.nodes[zid][IS_HAZARD] else 0.0
        is_exit = 1.0 if env.venue.graph.nodes[zid][NODE_TYPE] == "exit" else 0.0
        min_dist = env._min_distance_to_exit(zid)
        n_neighbors = len(env.adj.get(zid, [])) / max_neighbors
        vec.extend([occ, norm_occ, cap, risk, is_hazard, is_exit, min_dist, n_neighbors])

    # Pad zones up to max_zones
    pad_zones = max_zones - len(env.zone_ids)
    if pad_zones > 0:
        vec.extend([0.0] * 8 * pad_zones)

    # Neighbor‑level features: (capacity, congestion) for each possible neighbor slot
    for zid in env.zone_ids:
        neighs = env.adj.get(zid, [])
        for slot in range(max_neighbors):
            if slot < len(neighs):
                nbr, edge = neighs[slot]
                cap = edge[EDGE_CAPACITY]
                cong = env.edge_congestion.get((zid, nbr), 0) / cap if cap > 0 else 0.0
            else:
                cap = 0.0
                cong = 0.0
            vec.extend([cap, cong])
    if pad_zones > 0:
        vec.extend([0.0] * max_neighbors * 2 * pad_zones)

    # Global features (5)
    remaining = env.initial_population - env.total_evacuated
    frac_evacuated = (
        env.total_evacuated / env.initial_population if env.initial_population > 0 else 0.0
    )
    time_norm = env.current_step / env.max_steps
    n_blocked = len(env.blocked_edges) / max(1, env.venue.graph.number_of_edges())
    n_unavail = len(env.unavailable_exits) / max(1, env.num_exits)
    vec.extend([remaining, frac_evacuated, time_norm, n_blocked, n_unavail])
    return np.array(vec, dtype=np.float32)
