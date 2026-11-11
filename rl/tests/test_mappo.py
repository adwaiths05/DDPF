# tests/test_mappo.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
import torch
from src.venue_generator import VenueGenerator
from src.evacuation_env import EvacuationEnv
from src.mappo.multi_agent_env import MultiAgentEnv
from src.mappo.mappo_agent import MAPPOAgent


@pytest.fixture
def ma_env():
    venue = VenueGenerator.generate_grid_venue(rows=3, cols=3, num_exits=2, seed=0)
    return MultiAgentEnv(venue, seed=42)


def test_multi_agent_env_reset(ma_env):
    obs_dict, info = ma_env.reset(seed=0)
    assert isinstance(obs_dict, dict)
    assert len(obs_dict) == ma_env.max_zones
    for zid, obs in obs_dict.items():
        assert obs.ndim == 1
        assert obs.shape[0] > 0


def test_multi_agent_env_step(ma_env):
    obs_dict, _ = ma_env.reset(seed=0)
    # take first valid action for each zone
    masks = ma_env.action_masks()
    import numpy as np
    actions = {zid: int(np.argmax(masks[zid])) for zid in obs_dict}
    next_obs, reward_dict, done, trunc, info = ma_env.step(actions)
    assert isinstance(next_obs, dict)
    assert isinstance(done, bool)


def test_mappo_agent_select_actions(ma_env):
    obs_dict, _ = ma_env.reset(seed=0)
    import numpy as np
    zone_obs_dim = list(obs_dict.values())[0].shape[0]
    agent = MAPPOAgent(
        roles=None,
        device=torch.device("cpu"),
        leader_obs_dim=zone_obs_dim,
        zone_obs_dim=zone_obs_dim,
        exit_obs_dim=5,
        signage_obs_dim=zone_obs_dim,
        max_zones=ma_env.max_zones,
        max_neighbors=ma_env.max_neighbors,
    )
    masks_raw = ma_env.action_masks()
    obs_t = {zid: torch.from_numpy(obs.copy()).float() for zid, obs in obs_dict.items()}
    masks_t = {zid: torch.from_numpy(masks_raw[zid].astype('float32')) for zid in obs_dict}
    actions, logps, values = agent.select_actions(obs_t, masks_t)
    assert -1 in actions  # leader action present
    for zid in obs_dict:
        assert zid in actions
        assert 0 <= actions[zid] < ma_env.max_neighbors


def test_mappo_agent_update_smoke(ma_env):
    """Run a few steps and perform an update without crashing."""
    obs_dict, _ = ma_env.reset(seed=0)
    import numpy as np
    zone_obs_dim = list(obs_dict.values())[0].shape[0]
    agent = MAPPOAgent(
        roles=None,
        device=torch.device("cpu"),
        leader_obs_dim=zone_obs_dim,
        zone_obs_dim=zone_obs_dim,
        exit_obs_dim=5,
        signage_obs_dim=zone_obs_dim,
        max_zones=ma_env.max_zones,
        max_neighbors=ma_env.max_neighbors,
    )
    for step in range(20):
        masks_raw = ma_env.action_masks()
        obs_t = {zid: torch.from_numpy(obs.copy()).float() for zid, obs in obs_dict.items()}
        masks_t = {zid: torch.from_numpy(masks_raw[zid].astype('float32')) for zid in obs_dict}
        actions, _, _ = agent.select_actions(obs_t, masks_t)
        zone_acts = {k: v for k, v in actions.items() if k != -1}
        obs_dict, reward_dict, done, trunc, _ = ma_env.step(zone_acts)
        agent.store_reward(sum(reward_dict.values()), done or trunc)
        if done or trunc:
            obs_dict, _ = ma_env.reset(seed=step)
    metrics = agent.update(epochs=1, batch_size=8)
    assert "policy_loss" in metrics
