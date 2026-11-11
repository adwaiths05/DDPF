# tests/test_observation_encoder.py
import numpy as np
from src.observation_encoder import encode_observation
from src.venue_generator import VenueGenerator
from src.evacuation_env import EvacuationEnv

def test_encode_observation_shape():
    venue = VenueGenerator.generate_grid_venue(seed=0)
    env = EvacuationEnv(venue, seed=0)
    env.reset()
    obs = encode_observation(env)
    assert isinstance(obs, np.ndarray)
    # Expected dimension: max_zones * 8 + max_zones * max_neighbors * 2 + 5 (global)
    expected_dim = env.max_zones * 8 + env.max_zones * env.max_neighbors * 2 + 5
    assert obs.shape == (expected_dim,)
