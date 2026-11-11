# src/reward_function.py
"""Reward function for the evacuation environment.
The function returns a scalar reward and a dictionary of component values for
logging/analysis. It mirrors the logic that was previously embedded in
``evacuation_env._compute_reward``.
"""
from typing import Tuple, Dict
import numpy as np


def compute_reward(
    env,
    invalid_actions: int,
    prev_evacuated: int = 0,
    w_progress: float = 20.0,
    w_cong: float = -0.5,
    w_hazard: float = -1.0,
    w_invalid: float = -0.5,
    w_time: float = -0.1,
    w_completion: float = 100.0,
    w_failure: float = -10.0,
    w_stagnation: float = -1.0,
) -> Tuple[float, Dict[str, float]]:
    """Calculate the decomposed reward.

    Parameters
    ----------
    env: EvacuationEnv instance – needs ``total_evacuated``, ``initial_population``,
        ``zone_occupancy``, ``venue`` and other attributes used below.
    invalid_actions: Number of invalid actions taken this step.
    prev_evacuated: Number of people evacuated at the *previous* step.
    w_*: Weights for each component (defaults match the original design).

    Returns
    -------
    reward: float – total reward for the step.
    components: dict – named components for logging.
    """
    pop = env.initial_population if env.initial_population > 0 else 1.0

    # Incremental progress – only reward NEW evacuations this step
    delta_evacuated = env.total_evacuated - prev_evacuated
    progress = w_progress * (delta_evacuated / pop)

    # Stagnation penalty – penalise steps where no one was evacuated
    stagnation = w_stagnation if delta_evacuated == 0 else 0.0

    # Congestion penalty – average occupancy / capacity across zones
    congestion_vals = []
    for zid in env.zone_ids:
        occ = env.zone_occupancy.get(zid, 0)
        cap = env.venue.graph.nodes[zid]["capacity"]
        if cap > 0:
            congestion_vals.append(occ / cap)
    avg_cong = sum(congestion_vals) / len(congestion_vals) if congestion_vals else 0.0
    congestion_pen = w_cong * avg_cong

    # Hazard penalty – people in hazardous zones
    hazard_people = sum(
        env.zone_occupancy.get(zid, 0)
        for zid in env.hazard_zones
    )
    hazard_pen = w_hazard * (hazard_people / pop)

    # Invalid action penalty
    invalid_pen = w_invalid * invalid_actions

    # Time penalty – grows linearly with each step
    time_pen = w_time

    # Completion bonus / failure penalty
    completion_bonus = w_completion if env.total_evacuated >= env.initial_population else 0.0
    failure_pen = w_failure if (env.current_step >= env.max_steps and env.total_evacuated < env.initial_population) else 0.0

    total = (
        progress
        + stagnation
        + congestion_pen
        + hazard_pen
        + invalid_pen
        + time_pen
        + completion_bonus
        + failure_pen
    )
    components = {
        "progress": progress,
        "stagnation": stagnation,
        "congestion_pen": congestion_pen,
        "hazard_pen": hazard_pen,
        "invalid_pen": invalid_pen,
        "time_pen": time_pen,
        "completion_bonus": completion_bonus,
        "failure_pen": failure_pen,
    }
    return float(total), components
