# tests/test_venue_generator.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from src.venue_generator import VenueGenerator, VenueGraph


def test_grid_venue_creates_graph():
    venue = VenueGenerator.generate_grid_venue(rows=3, cols=3, num_exits=2, seed=0)
    assert isinstance(venue, VenueGraph)
    assert len(venue.zones) > 0
    assert len(venue.exits) >= 1


def test_grid_venue_connectivity():
    venue = VenueGenerator.generate_grid_venue(rows=3, cols=3, num_exits=2, seed=1)
    import networkx as nx
    # every zone should reach at least one exit
    for zone in venue.zones:
        lengths = nx.single_source_shortest_path_length(venue.graph, zone)
        reachable_exits = [e for e in venue.exits if e in lengths]
        assert len(reachable_exits) > 0, f"Zone {zone} cannot reach any exit"


def test_reproducibility():
    v1 = VenueGenerator.generate_grid_venue(seed=42)
    v2 = VenueGenerator.generate_grid_venue(seed=42)
    assert list(v1.graph.nodes) == list(v2.graph.nodes)
    assert list(v1.graph.edges) == list(v2.graph.edges)


def test_exits_are_subset_of_nodes():
    venue = VenueGenerator.generate_grid_venue(seed=7)
    for exit_id in venue.exits:
        assert exit_id in venue.graph.nodes


def test_node_attributes():
    venue = VenueGenerator.generate_grid_venue(seed=3)
    for n, data in venue.graph.nodes(data=True):
        assert "capacity" in data
        assert data["capacity"] > 0
