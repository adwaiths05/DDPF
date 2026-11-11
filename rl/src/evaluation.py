# src/evaluation.py
"""Evaluation utilities for the RL module.
Provides a simple Monte‑Carlo style evaluation that runs a given policy (PPO or MAPPO)
for a number of episodes and reports average reward, success rate (all agents evacuated),
and optional per‑step statistics.
"""
import os
import numpy as np
import torch
from .evacuation_env import EvacuationEnv
from .venue_generator import VenueGenerator
from .observation_encoder import encode_observation
from .reward_function import compute_reward


def _generate_evaluation_venue(seed: int):
    """Generate a valid random venue, falling back to a validated grid."""
    try:
        return VenueGenerator.generate_random_venue(seed=seed)
    except ValueError:
        return VenueGenerator.generate_grid_venue(seed=seed)


def evaluate_policy(
    policy_type: str = "ppo",
    checkpoint_path: str = "results/ppo/ppo_policy_final.pt",
    num_episodes: int = 10,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Run ``num_episodes`` episodes using the specified policy checkpoint.

    Parameters
    ----------
    policy_type: ``"ppo"`` or ``"mappo"``
    checkpoint_path: path to the saved model file.
    num_episodes: number of evaluation roll‑outs.
    device: ``"cpu"`` or ``"cuda"`` – will fall back to CPU if CUDA unavailable.
    seed: random seed for reproducibility.

    Returns
    -------
    dict with keys ``avg_reward``, ``success_rate`` and ``episode_rewards``.
    """
    torch.manual_seed(seed)
    device_ = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    # load model
    ckpt = torch.load(checkpoint_path, map_location=device_)
    if policy_type == "ppo":
        from .ppo.actor_critic import Actor
        # reconstruct actor network (observation dim, max_zones, max_neighbors)
        # we infer dimensions from checkpoint if possible; otherwise use env defaults
        # For simplicity we rebuild using env defaults
        venue = VenueGenerator.generate_grid_venue(seed=seed)
        env = EvacuationEnv(venue, seed=seed)
        obs_dim = env.observation_space.shape[0]
        max_zones = env.max_zones
        max_neighbors = env.max_neighbors
        actor = Actor(obs_dim, max_zones, max_neighbors).to(device_)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
    elif policy_type == "mappo":
        from .mappo.mappo_agent import MAPPOAgent, PolicyNet
        # We'll reconstruct a minimal MAPPOAgent using same dimensions as training
        venue = VenueGenerator.generate_grid_venue(seed=seed)
        env = EvacuationEnv(venue, seed=seed)
        max_zones = env.max_zones
        max_neighbors = env.max_neighbors
        # observation dimensions
        leader_obs_dim = encode_observation(env).shape[0]
        zone_obs_dim = 8 + max_neighbors * 2
        exit_obs_dim = 5
        signage_obs_dim = zone_obs_dim
        from .mappo.agent_roles import AgentRoles
        roles = AgentRoles(max_zones, max_neighbors)
        agent = MAPPOAgent(
            roles=roles,
            device=device_,
            leader_obs_dim=leader_obs_dim,
            zone_obs_dim=zone_obs_dim,
            exit_obs_dim=exit_obs_dim,
            signage_obs_dim=signage_obs_dim,
            max_zones=max_zones,
            max_neighbors=max_neighbors,
        )
        agent.leader_policy.load_state_dict(ckpt["leader_policy"])
        agent.zone_policy.load_state_dict(ckpt["zone_policy"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.leader_policy.eval()
        agent.zone_policy.eval()
        agent.critic.eval()
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")

    episode_rewards = []
    successes = 0
    for ep in range(num_episodes):
        # fresh venue each episode
        venue = _generate_evaluation_venue(seed + ep)
        env = EvacuationEnv(venue, seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        while not done:
            obs_tensor = torch.from_numpy(obs).float().to(device_)
            if policy_type == "ppo":
                with torch.no_grad():
                    logits = actor(obs_tensor)
                    # reshape and mask using env masks
                    logits = logits.view(max_zones, max_neighbors)
                    mask = torch.from_numpy(env.action_masks()).to(device_)
                    masked = logits.masked_fill(mask == 0, -1e10)
                    dist = torch.distributions.Categorical(logits=masked)
                    action = dist.sample().cpu().numpy()
            else:  # mappo
                # build multi‑agent observation dict
                full_obs = encode_observation(env)
                per_agent_obs = {}
                for zid in range(max_zones):
                    start = zid * 8
                    per = full_obs[start:start+8]
                    neigh_start = max_zones * 8 + zid * max_neighbors * 2
                    neigh = full_obs[neigh_start:neigh_start + max_neighbors * 2]
                    per_agent_obs[zid] = torch.from_numpy(np.concatenate([per, neigh])).float().to(device_)
                masks = {zid: torch.from_numpy(env.action_masks()[zid]).to(device_) for zid in range(max_zones)}
                with torch.no_grad():
                    actions, _, _ = agent.select_actions(per_agent_obs, masks)
                # strip leader action
                action = {k: v for k, v in actions.items() if k != -1}
            # step env (action may be a dict for MAPPO — convert to list)
            if isinstance(action, dict):
                action_arr = [action.get(z, 0) for z in range(max_zones)]
            else:
                action_arr = action
            next_obs, reward, done, trunc, info = env.step(action_arr)
            total_reward += reward
            obs = next_obs
            if trunc:
                done = True
        episode_rewards.append(total_reward)
        # success if all persons evacuated (reward > 0 and no remaining crowd)
        if env.total_evacuated >= env.initial_population:
            successes += 1
    avg_reward = float(np.mean(episode_rewards))
    success_rate = successes / num_episodes
    return {
        "avg_reward": avg_reward,
        "std_reward": float(np.std(episode_rewards)),
        "success_rate": success_rate,
        "episode_rewards": episode_rewards,
    }
