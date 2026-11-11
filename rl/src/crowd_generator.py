# src/crowd_generator.py
"""Synthetic crowd state generator for the evacuation environment.

The module defines a ``CrowdScenario`` dataclass containing the per‑zone
population counts and optional incident flags (blocked corridors, unavailable
exits, hazards). The ``generate_crowd_scenario`` function produces a scenario
consistent with a given ``VenueGraph`` and a chosen ``ScenarioType``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple

from .venue_generator import VenueGraph, NODE_TYPE, CAPACITY, OCCUPANCY, IS_HAZARD, RISK, IS_BLOCKED


class ScenarioType(Enum):
    """High‑level categories of synthetic evacuation situations.

    The enum members are used by ``generate_crowd_scenario`` to decide how to
    distribute the crowd and which incident flags to toggle.
    """

    NORMAL = auto()
    UNEVEN = auto()
    CONGESTED = auto()
    EXIT_BLOCKED = auto()
    CORRIDOR_BLOCKED = auto()
    HAZARD_ZONE = auto()
    MIXED = auto()


@dataclass
class CrowdScenario:
    """Container for a synthetic crowd configuration.

    Attributes
    ----------
    zone_occupancy: Dict[int, int]
        Mapping from zone node id to initial number of people.
    blocked_edges: List[Tuple[int, int]]
        List of edges (source, target) that are temporarily blocked.
    unavailable_exits: List[int]
        Exit node ids that are unavailable for this episode.
    hazard_zones: List[int]
        Zones flagged as hazardous (risk > 0).
    total_population: int
        Sum of all zone occupancies – used for invariant checks.
    """

    zone_occupancy: Dict[int, int]
    blocked_edges: List[Tuple[int, int]]
    unavailable_exits: List[int]
    hazard_zones: List[int]
    total_population: int


def _sample_population_for_zone(max_capacity: int, rng: random.Random) -> int:
    """Sample a realistic initial crowd count for a zone.

    The distribution is a truncated normal‑like shape: most zones get 20‑80% of
    their capacity, but we allow occasional over‑capacity to model pre‑event
    congestion.
    """
    # Base probability – uniform 30‑70% of capacity
    base = int(max_capacity * rng.uniform(0.3, 0.7))
    # With small probability add extra people (over‑capacity)
    if rng.random() < 0.15:
        base += int(max_capacity * rng.uniform(0.1, 0.3))
    return min(base, max_capacity * 2)  # allow up to double capacity for stress tests


def generate_crowd_scenario(
    venue: VenueGraph,
    scenario_type: ScenarioType = ScenarioType.NORMAL,
    seed: int = 0,
) -> CrowdScenario:
    """Create a synthetic crowd configuration for a given venue.

    Parameters
    ----------
    venue: VenueGraph
        The venue where the scenario will unfold.
    scenario_type: ScenarioType, optional
        Desired scenario category. ``NORMAL`` is a balanced distribution with
        no incidents. Other types introduce uneven crowds, blocked corridors, etc.
    seed: int
        Random seed for reproducibility.
    """
    rng = random.Random(seed)
    # ---------------------------------------------------------------------
    # Base occupancy – start from a normal distribution for all zones
    # ---------------------------------------------------------------------
    zone_occupancy: Dict[int, int] = {}
    total_pop = 0
    for zone_id in venue.zones:
        cap = venue.graph.nodes[zone_id][CAPACITY]
        occ = _sample_population_for_zone(cap, rng)
        zone_occupancy[zone_id] = occ
        total_pop += occ

    # ---------------------------------------------------------------------
    # Apply scenario‑specific modifications
    # ---------------------------------------------------------------------
    blocked_edges: List[Tuple[int, int]] = []
    unavailable_exits: List[int] = []
    hazard_zones: List[int] = []

    if scenario_type == ScenarioType.UNEVEN:
        # Skew population toward a random subset of zones (30‑50% of zones)
        heavy_zones = rng.sample(venue.zones, k=max(1, len(venue.zones) // 3))
        for zid in heavy_zones:
            cap = venue.graph.nodes[zid][CAPACITY]
            # Increase occupancy up to 150% of capacity
            zone_occupancy[zid] = min(int(cap * 1.5), cap * 2)
        total_pop = sum(zone_occupancy.values())

    elif scenario_type == ScenarioType.CONGESTED:
        # Reduce capacities on a random subset of edges to create congestion
        edges = list(venue.graph.edges)
        num_to_reduce = max(1, int(0.2 * len(edges)))
        for e in rng.sample(edges, k=num_to_reduce):
            orig = venue.graph.edges[e]["capacity"]
            new_cap = max(1, int(orig * rng.uniform(0.3, 0.5)))
            venue.graph.edges[e]["capacity"] = new_cap

    elif scenario_type == ScenarioType.EXIT_BLOCKED:
        if venue.exits:
            exit_to_block = rng.choice(venue.exits)
            unavailable_exits.append(exit_to_block)
            for src, _ in list(venue.graph.in_edges(exit_to_block)):
                venue.graph.edges[(src, exit_to_block)][IS_BLOCKED] = True

    elif scenario_type == ScenarioType.CORRIDOR_BLOCKED:
        if venue.graph.edges:
            edge_to_block = rng.choice(list(venue.graph.edges))
            blocked_edges.append(edge_to_block)
            venue.graph.edges[edge_to_block][IS_BLOCKED] = True

    elif scenario_type == ScenarioType.HAZARD_ZONE:
        if venue.zones:
            hazard_zone = rng.choice(venue.zones)
            hazard_zones.append(hazard_zone)
            venue.graph.nodes[hazard_zone][IS_HAZARD] = True
            venue.graph.nodes[hazard_zone][RISK] = rng.uniform(0.5, 1.0)
            cap = venue.graph.nodes[hazard_zone][CAPACITY]
            zone_occupancy[hazard_zone] = min(int(cap * 1.3), cap * 2)

    elif scenario_type == ScenarioType.MIXED:
        # Combine a few effects: uneven crowd + a blocked corridor + a hazard zone
        heavy_zones = rng.sample(venue.zones, k=max(1, len(venue.zones) // 4))
        for zid in heavy_zones:
            cap = venue.graph.nodes[zid][CAPACITY]
            zone_occupancy[zid] = min(int(cap * 1.4), cap * 2)
        if venue.graph.edges:
            edge_to_block = rng.choice(list(venue.graph.edges))
            blocked_edges.append(edge_to_block)
            venue.graph.edges[edge_to_block][IS_BLOCKED] = True
        if venue.zones:
            hazard_zone = rng.choice(venue.zones)
            hazard_zones.append(hazard_zone)
            venue.graph.nodes[hazard_zone][IS_HAZARD] = True
            venue.graph.nodes[hazard_zone][RISK] = rng.uniform(0.5, 1.0)
            cap = venue.graph.nodes[hazard_zone][CAPACITY]
            zone_occupancy[hazard_zone] = min(int(cap * 1.2), cap * 2)
        total_pop = sum(zone_occupancy.values())

    total_pop = sum(zone_occupancy.values())

    return CrowdScenario(
        zone_occupancy=zone_occupancy,
        blocked_edges=blocked_edges,
        unavailable_exits=unavailable_exits,
        hazard_zones=hazard_zones,
        total_population=total_pop,
    )


if __name__ == "__main__":
    from .venue_generator import VenueGenerator

    vg = VenueGenerator.generate_random_venue(seed=101)
    cs = generate_crowd_scenario(vg, scenario_type=ScenarioType.MIXED, seed=202)
    print("Total population:", cs.total_population)
    print("Blocked edges:", cs.blocked_edges)
    print("Unavailable exits:", cs.unavailable_exits)
    print("Hazard zones:", cs.hazard_zones)
    for zid, occ in cs.zone_occupancy.items():
        print(f"Zone {zid}: {occ} people (capacity {vg.graph.nodes[zid][CAPACITY]})")
