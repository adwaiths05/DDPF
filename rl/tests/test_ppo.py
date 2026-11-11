# tests/test_ppo.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
import torch
import numpy as np
from src.venue_generator import VenueGenerator
from src.evacuation_env import EvacuationEnv
from src.ppo.actor_critic import Actor, Critic
from src.ppo.rollout_buffer import RolloutBuffer
from src.ppo.ppo_agent import PPOAgent


@pytest.fixture
def env():
    venue = VenueGenerator.generate_grid_venue(rows=3, cols=3, num_exits=2, seed=0)
    e = EvacuationEnv(venue, seed=42)
    e.reset(seed=0)
    return e


def test_actor_output_shape(env):
    obs_dim = env.observation_space.shape[0]
    actor = Actor(obs_dim, env.max_zones, env.max_neighbors)
    obs = torch.zeros(obs_dim)
    out = actor(obs)
    assert out.shape == (env.max_zones * env.max_neighbors,)


def test_critic_output_shape(env):
    obs_dim = env.observation_space.shape[0]
    critic = Critic(obs_dim)
    obs = torch.zeros(obs_dim)
    val = critic(obs)
    assert val.shape == ()  # scalar


def test_rollout_buffer_add_and_batch(env):
    obs_dim = env.observation_space.shape[0]
    buf = RolloutBuffer(obs_dim, env.max_zones, env.max_neighbors)
    obs = torch.zeros(obs_dim)
    action = np.zeros(env.max_zones, dtype=int)
    log_prob = torch.tensor(0.0)
    value = torch.tensor(0.5)
    mask = np.ones((env.max_zones, env.max_neighbors), dtype=np.float32)
    buf.add(obs, action, log_prob, 1.0, False, value, mask)
    buf.add(obs, action, log_prob, 2.0, True, value, mask)
    batch = buf.get_batch()
    assert batch["obs"].shape[0] == 2
    assert batch["returns"].shape[0] == 2
    assert batch["advantages"].shape[0] == 2


def test_ppo_agent_select_action(env):
    obs_dim = env.observation_space.shape[0]
    device = torch.device("cpu")
    agent = PPOAgent(obs_dim, env.max_zones, env.max_neighbors, device)
    obs_t = torch.from_numpy(env.reset(seed=0)[0]).float()
    mask_t = torch.from_numpy(env.action_masks())
    action, log_prob, value = agent.select_action(obs_t, mask_t)
    assert action.shape == (env.max_zones,)


def test_ppo_agent_update_smoke(env):
    """Run a few steps and perform an update without crashing."""
    obs_dim = env.observation_space.shape[0]
    device = torch.device("cpu")
    agent = PPOAgent(obs_dim, env.max_zones, env.max_neighbors, device)
    obs, _ = env.reset(seed=0)
    for _ in range(20):
        obs_t = torch.from_numpy(obs).float()
        mask = env.action_masks()
        mask_t = torch.from_numpy(mask)
        action, log_prob, value = agent.select_action(obs_t, mask_t)
        obs, reward, done, trunc, _ = env.step(action)
        agent.buffer.add(obs_t, action, log_prob, reward, done or trunc, value, mask)
        if done or trunc:
            obs, _ = env.reset(seed=1)
    metrics = agent.update(epochs=1, batch_size=8)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics


def test_checkpoint_save_load(env, tmp_path):
    """Ensure checkpoint can be saved and loaded."""
    obs_dim = env.observation_space.shape[0]
    device = torch.device("cpu")
    agent = PPOAgent(obs_dim, env.max_zones, env.max_neighbors, device)
    ckpt_path = tmp_path / "ppo_test.pt"
    torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()}, ckpt_path)
    ckpt = torch.load(ckpt_path)
    agent2 = PPOAgent(obs_dim, env.max_zones, env.max_neighbors, device)
    agent2.actor.load_state_dict(ckpt["actor"])
    agent2.critic.load_state_dict(ckpt["critic"])
    # verify weights are identical
    for p1, p2 in zip(agent.actor.parameters(), agent2.actor.parameters()):
        assert torch.allclose(p1, p2)
