# tests/test_crowd_generator.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from src.venue_generator import VenueGenerator
from src.crowd_generator import generate_crowd_scenario, ScenarioType, CrowdScenario


@pytest.fixture
def venue():
    return VenueGenerator.generate_grid_venue(rows=3, cols=3, num_exits=2, seed=0)


def test_normal_scenario(venue):
    scenario = generate_crowd_scenario(venue, ScenarioType.NORMAL, seed=0)
    assert isinstance(scenario, CrowdScenario)
    total = sum(scenario.zone_occupancy.values())
    assert total > 0


def test_random_scenario_types(venue):
    for stype in ScenarioType:
        scenario = generate_crowd_scenario(venue, stype, seed=5)
        assert scenario is not None


def test_crowd_non_negative(venue):
    scenario = generate_crowd_scenario(venue, ScenarioType.UNEVEN, seed=1)
    for zone_id, count in scenario.zone_occupancy.items():
        assert count >= 0, f"Negative crowd at zone {zone_id}"


def test_total_population_consistent(venue):
    scenario = generate_crowd_scenario(venue, ScenarioType.NORMAL, seed=2)
    assert scenario.total_population == sum(scenario.zone_occupancy.values())


def test_reproducibility(venue):
    s1 = generate_crowd_scenario(venue, ScenarioType.NORMAL, seed=99)
    s2 = generate_crowd_scenario(venue, ScenarioType.NORMAL, seed=99)
    assert s1.zone_occupancy == s2.zone_occupancy
