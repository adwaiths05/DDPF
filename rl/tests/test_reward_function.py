# tests/test_reward_function.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from src.venue_generator import VenueGenerator
from src.evacuation_env import EvacuationEnv
from src.reward_function import compute_reward


@pytest.fixture
def env():
    venue = VenueGenerator.generate_grid_venue(rows=3, cols=3, num_exits=2, seed=0)
    e = EvacuationEnv(venue, seed=42)
    e.reset(seed=0)
    return e


def test_reward_is_finite(env):
    reward, components = compute_reward(env, invalid_actions=0)
    assert isinstance(reward, float)
    assert not (reward != reward)  # not NaN
    assert abs(reward) < 1e6  # finite


def test_reward_components_keys(env):
    _, components = compute_reward(env, invalid_actions=0)
    expected_keys = {"progress", "stagnation", "congestion_pen", "hazard_pen", "invalid_pen", "time_pen", "completion_bonus", "failure_pen"}
    assert set(components.keys()) == expected_keys


def test_invalid_action_penalty(env):
    _, c_no_invalid = compute_reward(env, invalid_actions=0)
    _, c_with_invalid = compute_reward(env, invalid_actions=5)
    # more invalid actions → lower (more negative) invalid_pen
    assert c_with_invalid["invalid_pen"] < c_no_invalid["invalid_pen"]


def test_completion_bonus(env):
    # simulate full evacuation
    env.total_evacuated = env.initial_population
    reward, components = compute_reward(env, invalid_actions=0)
    assert components["completion_bonus"] > 0
