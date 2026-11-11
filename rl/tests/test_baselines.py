# tests/test_baselines.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
import numpy as np
from src.venue_generator import VenueGenerator
from src.evacuation_env import EvacuationEnv
from src.baselines import ShortestPathPolicy, CongestionAwarePolicy


@pytest.fixture
def env():
    venue = VenueGenerator.generate_grid_venue(rows=3, cols=3, num_exits=2, seed=0)
    e = EvacuationEnv(venue, seed=42)
    e.reset(seed=0)
    return e


def test_shortest_path_produces_valid_action(env):
    policy = ShortestPathPolicy()
    action = policy.predict(env)
    assert len(action) == env.max_zones
    mask = env.action_masks()
    for zid_idx, act in enumerate(action):
        assert 0 <= act < env.max_neighbors


def test_congestion_aware_produces_valid_action(env):
    policy = CongestionAwarePolicy()
    action = policy.predict(env)
    assert len(action) == env.max_zones


def test_shortest_path_completes_episode():
    venue = VenueGenerator.generate_grid_venue(rows=3, cols=3, num_exits=2, seed=1)
    env = EvacuationEnv(venue, seed=1, max_steps=200)
    policy = ShortestPathPolicy()
    obs, _ = env.reset(seed=1)
    done, trunc = False, False
    steps = 0
    while not done and not trunc:
        action = policy.predict(env)
        obs, reward, done, trunc, info = env.step(action)
        steps += 1
    assert done or trunc
    assert steps <= 200


def test_congestion_aware_completes_episode():
    venue = VenueGenerator.generate_grid_venue(rows=3, cols=3, num_exits=2, seed=2)
    env = EvacuationEnv(venue, seed=2, max_steps=200)
    policy = CongestionAwarePolicy()
    obs, _ = env.reset(seed=2)
    done, trunc = False, False
    steps = 0
    while not done and not trunc:
        action = policy.predict(env)
        obs, reward, done, trunc, info = env.step(action)
        steps += 1
    assert done or trunc
