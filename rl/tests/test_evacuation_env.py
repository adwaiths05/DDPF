# tests/test_evacuation_env.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
import numpy as np
from src.venue_generator import VenueGenerator
from src.evacuation_env import EvacuationEnv


@pytest.fixture
def env():
    venue = VenueGenerator.generate_grid_venue(rows=3, cols=3, num_exits=2, seed=0)
    return EvacuationEnv(venue, seed=42)


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert "initial_population" in info


def test_obs_shape_consistent(env):
    obs, _ = env.reset(seed=1)
    assert obs.ndim == 1
    expected = env.max_zones * 8 + env.max_zones * env.max_neighbors * 2 + 5
    assert obs.shape[0] == expected


def test_action_mask_shape(env):
    env.reset(seed=2)
    mask = env.action_masks()
    assert mask.shape == (env.max_zones, env.max_neighbors)
    assert mask.dtype in (np.int8, np.int32, np.int64, np.bool_)


def test_step_returns_five_tuple(env):
    env.reset(seed=3)
    action = np.zeros(env.max_zones, dtype=int)
    result = env.step(action)
    assert len(result) == 5
    obs, reward, done, trunc, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(trunc, bool)


def test_population_conservation(env):
    """People should only leave the system by exiting — never appear from nowhere."""
    env.reset(seed=4)
    initial = env.initial_population
    for _ in range(10):
        action = np.zeros(env.max_zones, dtype=int)
        env.step(action)
        in_zones = sum(env.zone_occupancy.values())
        total = in_zones + env.total_evacuated
        assert total <= initial, f"Population not conserved: {total} > {initial}"


def test_episode_terminates(env):
    """Episode should terminate within max_steps."""
    env.reset(seed=5)
    done, trunc = False, False
    steps = 0
    while not done and not trunc:
        action = np.zeros(env.max_zones, dtype=int)
        _, _, done, trunc, _ = env.step(action)
        steps += 1
        if steps > env.max_steps + 5:
            break
    assert done or trunc, "Episode did not terminate"


def test_invalid_action_mask_respected(env):
    """Actions respecting mask should never trigger invalid_action > 0."""
    obs, _ = env.reset(seed=6)
    mask = env.action_masks()  # (max_zones, max_neighbors)
    # choose first valid action for each zone
    action = np.argmax(mask, axis=-1)
    _, _, _, _, info = env.step(action)
    assert info.get("invalid_actions", 0) == 0
