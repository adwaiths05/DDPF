# src/venue_generator.py
"""Synthetic venue graph generator.

Provides utilities to create artificial indoor venue layouts represented as a directed
NetworkX graph. Nodes correspond to zones (including exits) and edges correspond to
traversable corridors. The generator supports reproducible random seeds and ensures
that every ordinary zone has a path to at least one exit under normal (unblocked)
conditions.

Key classes:
- `VenueGraph`: thin wrapper around a `networkx.DiGraph` exposing convenient helper
  methods and validation utilities.
- `VenueGenerator`: static methods to build venues using different strategies
  (random Erdős–Rényi, grid layout, or a multi‑wing complex layout).

The implementation focuses on clarity and extensibility – later phases can replace
or augment the graph generation logic without touching the rest of the codebase.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants & type aliases
# ---------------------------------------------------------------------------
NodeId = int
EdgeKey = Tuple[NodeId, NodeId]

# Node attribute keys (all stored in the NetworkX node dict)
NODE_TYPE = "node_type"  # "zone" or "exit"
CAPACITY = "capacity"  # max people that can reside in the zone
OCCUPANCY = "occupancy"  # current number of people (initially 0)
RISK = "risk"  # scalar risk score, 0.0–1.0
IS_HAZARD = "is_hazard"  # bool flag for hazardous zones
POS = "pos"  # (x, y) tuple for visualization

# Edge attribute keys
TRAVEL_TIME = "travel_time"  # base traversal time (steps)
EDGE_CAPACITY = "capacity"  # max people per step that can traverse
IS_BLOCKED = "is_blocked"  # bool, true if corridor is blocked
CONGESTION = "congestion"  # dynamic metric, updated during simulation

# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------
@dataclass
class VenueGraph:
    """Wrapper around a NetworkX DiGraph with venue‑specific helpers.

    The wrapper makes it easy to query exits, ordinary zones and to validate the
    connectivity constraints required by the RL environment.
    """

    graph: nx.DiGraph

    @property
    def nodes(self) -> List[NodeId]:
        return list(self.graph.nodes)

    @property
    def exits(self) -> List[NodeId]:
        return [n for n, d in self.graph.nodes(data=True) if d.get(NODE_TYPE) == "exit"]

    @property
    def zones(self) -> List[NodeId]:
        return [n for n, d in self.graph.nodes(data=True) if d.get(NODE_TYPE) == "zone"]

    def is_connected_to_exit(self) -> bool:
        """Return ``True`` if **every** zone can reach at least one exit
        via directed edges (ignoring blocked flags).
        """
        for zone in self.zones:
            reachable = False
            for exit_node in self.exits:
                if nx.has_path(self.graph, zone, exit_node):
                    reachable = True
                    break
            if not reachable:
                return False
        return True

    def validate(self) -> None:
        """Validate structural invariants.

        Raises ``ValueError`` with a helpful message if any invariant is violated.
        """
        if len(self.exits) < 2:
            raise ValueError("Venue must contain at least two exit nodes.")
        if len(self.zones) < 6:
            raise ValueError("Venue must contain at least six ordinary zones.")
        if not self.is_connected_to_exit():
            raise ValueError("At least one zone cannot reach any exit.")
        # Ensure every edge has the required attributes
        for u, v, d in self.graph.edges(data=True):
            missing = [key for key in (TRAVEL_TIME, EDGE_CAPACITY, IS_BLOCKED, CONGESTION) if key not in d]
            if missing:
                raise ValueError(f"Edge {(u, v)} missing attributes: {missing}")
        # Node attribute sanity checks
        for n, d in self.graph.nodes(data=True):
            if NODE_TYPE not in d:
                raise ValueError(f"Node {n} missing '{NODE_TYPE}' attribute.")
            if CAPACITY not in d:
                raise ValueError(f"Node {n} missing '{CAPACITY}' attribute.")
            # occupancy starts at zero
            d[OCCUPANCY] = 0
            d.setdefault(RISK, 0.0)
            d.setdefault(IS_HAZARD, False)
            d.setdefault(POS, (0.0, 0.0))

    # ---------------------------------------------------------------------
    # Visualization helpers
    # ---------------------------------------------------------------------
    def draw(self, ax: plt.Axes | None = None, show: bool = False) -> plt.Axes:
        """Render the venue graph.

        *Zones* are drawn as circles, *exits* as squares. Node colour encodes risk –
        low risk (green) to high risk (red). Edge width corresponds to capacity.
        ``ax`` can be supplied for embedding in larger figures.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        pos = {n: d[POS] for n, d in self.graph.nodes(data=True)}
        # Node styling
        node_colors = []
        node_shapes = {"zone": "o", "exit": "s"}
        for n in self.graph.nodes:
            node_type = self.graph.nodes[n][NODE_TYPE]
            risk = self.graph.nodes[n][RISK]
            # Map risk 0..1 to green→red gradient
            color = (risk, 1 - risk, 0)  # (R, G, B)
            node_colors.append(color)
        # Separate draws for each shape because NetworkX does not support per‑node shape
        for node_type, shape in node_shapes.items():
            nodelist = [n for n, d in self.graph.nodes(data=True) if d[NODE_TYPE] == node_type]
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=nodelist,
                node_shape=shape,
                node_color=[node_colors[i] for i, n in enumerate(self.graph.nodes) if n in nodelist],
                node_size=500,
                ax=ax,
            )
        # Edge styling – width proportional to capacity, colour dark if blocked
        widths = []
        edge_colors = []
        for u, v, d in self.graph.edges(data=True):
            cap = d[EDGE_CAPACITY]
            widths.append(2 + 3 * (cap - 1) / (max(30, cap) - 1))  # simple scaling
            edge_colors.append("gray" if not d[IS_BLOCKED] else "black")
        nx.draw_networkx_edges(self.graph, pos, width=widths, edge_color=edge_colors, ax=ax)
        # Labels
        labels = {n: f"{n}\n{'E' if self.graph.nodes[n][NODE_TYPE]=='exit' else ''}" for n in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8, ax=ax)
        ax.set_axis_off()
        if show:
            plt.show()
        return ax

# ---------------------------------------------------------------------------
# Core generator implementation
# ---------------------------------------------------------------------------
class VenueGenerator:
    """Factory class providing static methods to generate synthetic venues.

    All methods accept an explicit ``seed`` for reproducibility. The returned
    ``VenueGraph`` is already validated.
    """

    @staticmethod
    def _initialize_graph(num_zones: int, num_exits: int, seed: int) -> nx.DiGraph:
        random.seed(seed)
        g = nx.DiGraph()
        # Assign deterministic node ids: zones 0..num_zones-1, exits after that
        for i in range(num_zones):
            g.add_node(i, node_type="zone", capacity=random.randint(30, 60), risk=0.0, is_hazard=False)
        for e in range(num_exits):
            nid = num_zones + e
            g.add_node(
                nid,
                node_type="exit",
                capacity=random.randint(20, 40),  # exit throughput per step
                risk=0.0,
                is_hazard=False,
            )
        return g

    @staticmethod
    def _add_random_edges(g: nx.DiGraph, avg_degree: float, seed: int) -> None:
        random.seed(seed)
        nodes = list(g.nodes)
        n = len(nodes)
        # Ensure connectivity by creating a random spanning tree first
        tree_edges = list(nx.random_tree(n, seed=seed).edges())
        for u, v in tree_edges:
            # Random direction for each undirected edge
            if random.random() < 0.5:
                src, dst = u, v
            else:
                src, dst = v, u
            g.add_edge(
                src,
                dst,
                travel_time=random.randint(1, 4),
                capacity=random.randint(5, 30),
                is_blocked=False,
                congestion=0.0,
            )
        # Add extra edges to reach desired average degree
        target_edges = int(avg_degree * n / 2)  # each directed edge counts as 1
        while g.number_of_edges() < target_edges:
            u, v = random.sample(nodes, 2)
            if g.has_edge(u, v):
                continue
            g.add_edge(
                u,
                v,
                travel_time=random.randint(1, 4),
                capacity=random.randint(5, 30),
                is_blocked=False,
                congestion=0.0,
            )

    @staticmethod
    def _place_exits(g: nx.DiGraph, num_exits: int, seed: int) -> None:
        # Already added as isolated nodes; now connect each exit to at least one zone
        random.seed(seed)
        zone_ids = [n for n, d in g.nodes(data=True) if d[NODE_TYPE] == "zone"]
        exit_ids = [n for n, d in g.nodes(data=True) if d[NODE_TYPE] == "exit"]
        for exit_id in exit_ids:
            # Connect exit to a random zone (bidirectional for realism)
            neighbour = random.choice(zone_ids)
            # Edge from zone to exit (people leave) – high capacity
            g.add_edge(
                neighbour,
                exit_id,
                travel_time=1,
                capacity=random.randint(15, 35),
                is_blocked=False,
                congestion=0.0,
            )
            # Optional reverse edge (unlikely to be used) with low capacity
            if random.random() < 0.3:
                g.add_edge(
                    exit_id,
                    neighbour,
                    travel_time=1,
                    capacity=2,
                    is_blocked=False,
                    congestion=0.0,
                )

    @staticmethod
    def _inject_bottleneck(g: nx.DiGraph, seed: int) -> None:
        """Force at least one narrow corridor (capacity ≤ 5).
        """
        random.seed(seed)
        edges = list(g.edges)
        if not edges:
            return
        bottleneck_edge = random.choice(edges)
        g.edges[bottleneck_edge][EDGE_CAPACITY] = random.randint(2, 5)
        g.edges[bottleneck_edge][TRAVEL_TIME] = random.randint(2, 4)

    @staticmethod
    def _assign_positions(g: nx.DiGraph, seed: int) -> None:
        """Simple 2‑D layout for visualization using spring layout.
        """
        random.seed(seed)
        pos = nx.spring_layout(g, seed=seed, k=0.5)
        for n, p in pos.items():
            g.nodes[n][POS] = (p[0] * 10, p[1] * 10)

    # ---------------------------------------------------------------------
    # Public factory methods
    # ---------------------------------------------------------------------
    @staticmethod
    def generate_random_venue(
        num_zones: int = 8,
        num_exits: int = 2,
        avg_degree: float = 2.5,
        seed: int = 42,
    ) -> VenueGraph:
        """Generate a random connected venue using an Erdős–Rényi‑like approach.

        Parameters
        ----------
        num_zones: int
            Number of ordinary zones (default 8, within 6‑10 range).
        num_exits: int
            Number of exit nodes (default 2).
        avg_degree: float
            Desired average degree of the underlying directed graph.
        seed: int
            Random seed for reproducibility.
        """
        g = VenueGenerator._initialize_graph(num_zones, num_exits, seed)
        VenueGenerator._add_random_edges(g, avg_degree, seed + 1)
        VenueGenerator._place_exits(g, num_exits, seed + 2)
        VenueGenerator._inject_bottleneck(g, seed + 3)
        VenueGenerator._assign_positions(g, seed + 4)
        vg = VenueGraph(g)
        vg.validate()
        return vg

    @staticmethod
    def generate_grid_venue(
        rows: int = 3,
        cols: int = 3,
        num_exits: int = 2,
        seed: int = 123,
    ) -> VenueGraph:
        """Create a rectangular grid layout and then decorate it with exits.
        The grid yields ``rows * cols`` zones. Exits are attached to random border
        zones.
        """
        random.seed(seed)
        g = nx.DiGraph()
        # Create grid nodes
        zone_id = 0
        grid_map: Dict[Tuple[int, int], NodeId] = {}
        for r in range(rows):
            for c in range(cols):
                g.add_node(
                    zone_id,
                    node_type="zone",
                    capacity=random.randint(30, 60),
                    risk=0.0,
                    is_hazard=False,
                )
                grid_map[(r, c)] = zone_id
                zone_id += 1
        # Connect neighbours (4‑connected grid) with directed edges both ways
        for (r, c), nid in grid_map.items():
            for dr, dc in [(0, 1), (1, 0)]:  # right and down to avoid duplicates
                nr, nc = r + dr, c + dc
                if (nr, nc) in grid_map:
                    neighbor_id = grid_map[(nr, nc)]
                    # Add both directions with random capacities
                    for src, dst in [(nid, neighbor_id), (neighbor_id, nid)]:
                        g.add_edge(
                            src,
                            dst,
                            travel_time=1,
                            capacity=random.randint(10, 30),
                            is_blocked=False,
                            congestion=0.0,
                        )
        # Add exits as new nodes
        exit_ids: List[NodeId] = []
        for i in range(num_exits):
            exit_node = zone_id + i
            g.add_node(
                exit_node,
                node_type="exit",
                capacity=random.randint(20, 40),
                risk=0.0,
                is_hazard=False,
            )
            exit_ids.append(exit_node)
        # Connect each exit to a random border zone (border = r==0 or r==rows-1 or c==0 or c==cols-1)
        border_zones = [nid for (r, c), nid in grid_map.items() if r == 0 or r == rows - 1 or c == 0 or c == cols - 1]
        for exit_node in exit_ids:
            neighbour = random.choice(border_zones)
            g.add_edge(
                neighbour,
                exit_node,
                travel_time=1,
                capacity=random.randint(15, 35),
                is_blocked=False,
                congestion=0.0,
            )
        # Position nodes using a simple lattice for readability
        for (r, c), nid in grid_map.items():
            g.nodes[nid][POS] = (c * 2.0, -r * 2.0)
        for i, exit_node in enumerate(exit_ids):
            # Place exits slightly outside the grid on the bottom side
            g.nodes[exit_node][POS] = ((cols - 1) * 2.0 + (i + 1) * 1.5, -(rows - 1) * 2.0)
        vg = VenueGraph(g)
        vg.validate()
        return vg

    @staticmethod
    def generate_complex_venue(seed: int = 999) -> VenueGraph:
        """Generate a multi‑wing building with a clear bottleneck.

        The procedure:
        1. Create three wing sub‑graphs (each a small random graph).
        2. Connect wings through a narrow corridor (the bottleneck).
        3. Attach exits to different wings.
        4. Add optional hazard zones.
        """
        random.seed(seed)
        g = nx.DiGraph()
        zone_counter = 0
        wing_nodes: List[List[NodeId]] = []
        # 3 wings, each with 4‑6 zones
        for wing in range(3):
            n_zones = random.randint(4, 6)
            nodes = []
            for _ in range(n_zones):
                g.add_node(
                    zone_counter,
                    node_type="zone",
                    capacity=random.randint(30, 60),
                    risk=0.0,
                    is_hazard=False,
                )
                nodes.append(zone_counter)
                zone_counter += 1
            # Connect wing zones internally (random tree + extra edges)
            VenueGenerator._add_random_edges(g, avg_degree=2.0, seed=seed + wing)
            wing_nodes.append(nodes)
        # Connect wings via a bottleneck corridor (single edge with low capacity)
        b_src = random.choice(wing_nodes[0])
        b_dst = random.choice(wing_nodes[1])
        g.add_edge(
            b_src,
            b_dst,
            travel_time=3,
            capacity=2,  # narrow corridor
            is_blocked=False,
            congestion=0.0,
        )
        # Also connect wing 1 to wing 2 (higher capacity)
        g.add_edge(
            random.choice(wing_nodes[1]),
            random.choice(wing_nodes[2]),
            travel_time=2,
            capacity=random.randint(10, 20),
            is_blocked=False,
            congestion=0.0,
        )
        # Add exits – one per wing (except middle wing may share)
        exit_ids = []
        for i in range(3):
            exit_node = zone_counter + i
            g.add_node(
                exit_node,
                node_type="exit",
                capacity=random.randint(20, 40),
                risk=0.0,
                is_hazard=False,
            )
            exit_ids.append(exit_node)
            g.add_edge(
                random.choice(wing_nodes[i]),
                exit_node,
                travel_time=1,
                capacity=random.randint(15, 35),
                is_blocked=False,
                congestion=0.0,
            )
        # Optionally mark a random zone as hazardous
        if random.random() < 0.4:
            hazard_zone = random.choice(list(g.nodes))
            if g.nodes[hazard_zone][NODE_TYPE] == "zone":
                g.nodes[hazard_zone][IS_HAZARD] = True
                g.nodes[hazard_zone][RISK] = random.uniform(0.5, 1.0)
        # Assign positions – each wing placed in a distinct quadrant for clarity
        offset = [(0, 0), (30, 0), (0, -30)]
        for wing_idx, nodes in enumerate(wing_nodes):
            for n in nodes:
                base = offset[wing_idx]
                g.nodes[n][POS] = (base[0] + random.uniform(-5, 5), base[1] + random.uniform(-5, 5))
        for i, exit_node in enumerate(exit_ids):
            base = offset[i]
            g.nodes[exit_node][POS] = (base[0] + 8, base[1] - 8)
        vg = VenueGraph(g)
        vg.validate()
        return vg

# ---------------------------------------------------------------------------
# Simple demo when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vg = VenueGenerator.generate_random_venue(seed=123)
    ax = vg.draw(show=True)
    plt.title("Random Synthetic Venue (seed=123)")
    plt.show()
