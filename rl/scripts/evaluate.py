# scripts/evaluate.py
"""Evaluation script: compare all trained policies on held-out venues.

Usage:
    python scripts/evaluate.py --ppo-ckpt results/ppo/ppo_policy_final.pt
    python scripts/evaluate.py --mappo-ckpt results/mappo/mappo_policy_final.pt --n-episodes 20
    python scripts/evaluate.py --all  # evaluate heuristics + PPO + MAPPO together
"""
import sys
import pathlib
import json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.baselines import ShortestPathPolicy, CongestionAwarePolicy
from src.evacuation_env import EvacuationEnv
from src.venue_generator import VenueGenerator
import numpy as np


def evaluate_heuristic(policy_cls, n_episodes=10, seed=10000):
    rewards = []
    successes = 0
    for ep in range(n_episodes):
        try:
            venue = VenueGenerator.generate_random_venue(seed=seed + ep)
        except Exception:
            venue = VenueGenerator.generate_grid_venue(seed=seed + ep)
        env = EvacuationEnv(venue, seed=seed + ep)
        policy = policy_cls()
        obs, _ = env.reset(seed=seed + ep)
        done, trunc = False, False
        total_reward = 0.0
        while not done and not trunc:
            action = policy.predict(env)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        if env.total_evacuated >= env.initial_population:
            successes += 1
    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": successes / n_episodes,
        "episode_rewards": rewards,
    }


def evaluate_ppo(checkpoint_path, n_episodes=10, seed=10000, device="cpu"):
    from src.evaluation import evaluate_policy
    return evaluate_policy(
        policy_type="ppo",
        checkpoint_path=checkpoint_path,
        num_episodes=n_episodes,
        device=device,
        seed=seed,
    )


def evaluate_mappo(checkpoint_path, n_episodes=10, seed=10000, device="cpu"):
    from src.evaluation import evaluate_policy
    return evaluate_policy(
        policy_type="mappo",
        checkpoint_path=checkpoint_path,
        num_episodes=n_episodes,
        device=device,
        seed=seed,
    )


def print_results(name, results):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Avg Reward   : {results['avg_reward']:.3f} (+/- {results.get('std_reward', 0):.3f})")
    print(f"  Success Rate : {results['success_rate']*100:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained policies")
    parser.add_argument("--ppo-ckpt", type=str, default=None)
    parser.add_argument("--mappo-ckpt", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--all", action="store_true", help="Evaluate all available policies")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    all_results = {}

    # Always evaluate heuristics
    print("Evaluating ShortestPath baseline...")
    sp_results = evaluate_heuristic(ShortestPathPolicy, n_episodes=args.n_episodes, seed=args.seed)
    print_results("Shortest Path Baseline", sp_results)
    all_results["shortest_path"] = sp_results

    print("Evaluating CongestionAware baseline...")
    ca_results = evaluate_heuristic(CongestionAwarePolicy, n_episodes=args.n_episodes, seed=args.seed)
    print_results("Congestion Aware Baseline", ca_results)
    all_results["congestion_aware"] = ca_results

    if args.ppo_ckpt or args.all:
        ckpt = args.ppo_ckpt or "results/ppo/ppo_policy_final.pt"
        if pathlib.Path(ckpt).exists():
            print(f"Evaluating PPO ({ckpt})...")
            ppo_results = evaluate_ppo(ckpt, n_episodes=args.n_episodes, seed=args.seed, device=args.device)
            print_results("PPO Policy", ppo_results)
            all_results["ppo"] = ppo_results
        else:
            print(f"PPO checkpoint not found: {ckpt}")

    if args.mappo_ckpt or args.all:
        ckpt = args.mappo_ckpt or "results/mappo/mappo_policy_final.pt"
        if pathlib.Path(ckpt).exists():
            print(f"Evaluating MAPPO ({ckpt})...")
            mappo_results = evaluate_mappo(ckpt, n_episodes=args.n_episodes, seed=args.seed, device=args.device)
            print_results("MAPPO Policy", mappo_results)
            all_results["mappo"] = mappo_results
        else:
            print(f"MAPPO checkpoint not found: {ckpt}")

    if args.output:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
