# scripts/run_demo.py
"""Interactive demo: run a complete evacuation episode with a random policy
and print step-by-step results.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --policy shortest_path
    python scripts/run_demo.py --policy ppo --checkpoint results/ppo/ppo_policy_final.pt
    python scripts/run_demo.py --policy mappo --checkpoint results/mappo/mappo_policy_final.pt
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.venue_generator import VenueGenerator
from src.evacuation_env import EvacuationEnv
from src.baselines import ShortestPathPolicy, CongestionAwarePolicy


def run_demo(policy_name="shortest_path", checkpoint=None, seed=42, max_steps=50):
    print(f"\n{'='*60}")
    print(f"  Crowd Evacuation Demo  |  Policy: {policy_name.upper()}")
    print(f"{'='*60}")

    # Generate venue
    try:
        venue = VenueGenerator.generate_random_venue(seed=seed)
    except Exception:
        venue = VenueGenerator.generate_grid_venue(seed=seed)
    print(f"Venue: {len(venue.zones)} zones, {len(venue.exits)} exits")

    if policy_name == "mappo":
        from src.mappo.multi_agent_env import MultiAgentEnv
        env = MultiAgentEnv(venue, seed=seed)
        env.base_env.max_steps = max_steps
        base_env = env.base_env
    else:
        env = EvacuationEnv(venue, seed=seed, max_steps=max_steps)
        base_env = env
    obs, info = env.reset(seed=seed)
    print(f"Initial population: {base_env.initial_population}")
    print(f"Zones: {base_env.zone_ids}")
    print(f"Exits: {base_env.exit_ids}")
    print()

    # Setup policy
    mappo_agent = None
    if policy_name == "shortest_path":
        policy = ShortestPathPolicy()
    elif policy_name == "congestion_aware":
        policy = CongestionAwarePolicy()
    elif policy_name == "mappo" and checkpoint:
        import torch
        from src.mappo.mappo_agent import MAPPOAgent
        device = torch.device("cpu")
        zone_obs_dim = next(iter(obs.values())).shape[0]
        mappo_agent = MAPPOAgent(
            roles=None,
            device=device,
            leader_obs_dim=zone_obs_dim,
            zone_obs_dim=zone_obs_dim,
            exit_obs_dim=5,
            signage_obs_dim=zone_obs_dim,
            max_zones=env.max_zones,
            max_neighbors=env.max_neighbors,
        )
        ckpt = torch.load(checkpoint, map_location=device)
        mappo_agent.leader_policy.load_state_dict(ckpt["leader_policy"])
        mappo_agent.zone_policy.load_state_dict(ckpt["zone_policy"])
        mappo_agent.leader_policy.eval()
        mappo_agent.zone_policy.eval()
        policy = None
    elif policy_name == "ppo" and checkpoint:
        import torch
        from src.ppo.actor_critic import Actor
        device = torch.device("cpu")
        actor = Actor(obs.shape[0], base_env.max_zones, base_env.max_neighbors)
        ckpt = torch.load(checkpoint, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        policy = None  # handled inline below
    else:
        policy = ShortestPathPolicy()  # default

    total_reward = 0.0
    done, trunc = False, False
    step = 0

    while not done and not trunc:
        step += 1
        if policy is not None:
            action = policy.predict(env)
        elif mappo_agent is not None:
            import torch
            masks = env.action_masks()
            obs_t = {zid: torch.from_numpy(zone_obs).float()
                     for zid, zone_obs in obs.items()}
            masks_t = {zid: torch.from_numpy(mask.astype(np.float32))
                       for zid, mask in masks.items()}
            with torch.no_grad():
                actions, _, _ = mappo_agent.select_actions(obs_t, masks_t)
            action = {zid: act for zid, act in actions.items() if zid != -1}
        else:
            import torch
            obs_t = torch.from_numpy(obs).float()
            with torch.no_grad():
                logits = actor(obs_t).view(base_env.max_zones, base_env.max_neighbors)
                mask = torch.from_numpy(base_env.action_masks())
                logits = logits.masked_fill(mask == 0, -1e10)
                action = logits.argmax(dim=-1).numpy()

        if mappo_agent is not None:
            obs, reward_dict, done, trunc, info = env.step(action)
            reward = next(iter(reward_dict.values()), 0.0)
        else:
            obs, reward, done, trunc, info = env.step(action)
        total_reward += reward

        if mappo_agent is not None:
            info = {
                "evacuated": base_env.total_evacuated,
                "remaining": base_env.initial_population - base_env.total_evacuated,
            }
        evacuated = info["evacuated"]
        remaining = info["remaining"]
        pct = evacuated / base_env.initial_population * 100
        print(f"Step {step:3d} | Evacuated: {evacuated:4d} ({pct:5.1f}%) | "
              f"Remaining: {remaining:4d} | Reward: {reward:+.2f}")

        if step % 10 == 0:
            print(f"         Zone occupancies: { {zid: base_env.zone_occupancy.get(zid,0) for zid in list(base_env.zone_ids)[:5]} }")

    print()
    print(f"Episode finished at step {step}")
    if done and base_env.total_evacuated >= base_env.initial_population:
        print("OUTCOME: SUCCESS - all occupants evacuated!")
    else:
        pct_evac = base_env.total_evacuated / base_env.initial_population * 100
        print(f"OUTCOME: TIMEOUT - evacuated {base_env.total_evacuated}/{base_env.initial_population} ({pct_evac:.1f}%)")
    print(f"Total reward: {total_reward:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="shortest_path",
                        choices=["shortest_path", "congestion_aware", "ppo", "mappo", "random"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()
    run_demo(args.policy, args.checkpoint, args.seed, args.max_steps)
