# src/ppo/trainer.py
"""Training loop for PPO.
Uses the PPOAgent, environment, and optional evaluation callbacks.

Improvements:
* Venue randomisation every episode for generalisation
* Frequent updates (every ``update_interval`` steps)
* LR scheduling
* Episode-level metric logging
"""
import os
import pathlib
from datetime import datetime
import numpy as np
import torch
from ..evacuation_env import EvacuationEnv
from ..venue_generator import VenueGenerator
from .ppo_agent import PPOAgent
from .rollout_buffer import RolloutBuffer
from ..observation_encoder import encode_observation
from ..reward_function import compute_reward


def _make_venue(seed: int):
    try:
        return VenueGenerator.generate_random_venue(seed=seed)
    except Exception:
        return VenueGenerator.generate_grid_venue(seed=seed)


def train_ppo(
    seed: int = 42,
    max_steps: int = 50000,
    log_dir: str = "results/ppo",
    device: str = "cpu",
    eval_interval: int = 2000,
    update_interval: int = 256,
    lr: float = 5e-4,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_ = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Training PPO on {device_}")
    print(f"  max_steps={max_steps}  update_interval={update_interval}  lr={lr}")

    # Generate initial venue
    venue = _make_venue(seed)
    env = EvacuationEnv(venue, seed=seed)

    obs, _ = env.reset(seed=seed)
    obs_dim = obs.shape[0]
    max_zones = env.max_zones
    max_neighbors = env.max_neighbors

    agent = PPOAgent(obs_dim, max_zones, max_neighbors, device_, lr=lr)

    # LR schedule: linear decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        agent.optimizer, lr_lambda=lambda step: max(1.0 - step / max_steps, 0.1)
    )

    # Episode tracking
    episode_rewards = []
    episode_successes = []
    current_episode_reward = 0.0
    episode_count = 0
    venue_seed = seed

    for step in range(max_steps):
        obs_tensor = torch.from_numpy(obs).float().to(device_)
        mask = env.action_masks()
        action, log_prob, value = agent.select_action(obs_tensor, torch.from_numpy(mask).to(device_))
        next_obs, reward, done, trunc, info = env.step(action)

        current_episode_reward += reward
        agent.buffer.add(
            obs=obs_tensor,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done or trunc,
            value=value,
            mask=mask,
        )
        obs = next_obs

        if done or trunc:
            episode_rewards.append(current_episode_reward)
            success = env.total_evacuated >= env.initial_population
            episode_successes.append(1 if success else 0)
            current_episode_reward = 0.0
            episode_count += 1

            # Randomise venue every episode
            venue_seed += 1
            venue = _make_venue(venue_seed)
            env = EvacuationEnv(venue, seed=venue_seed)
            obs, _ = env.reset(seed=venue_seed)

        # Periodic update
        if (step + 1) % update_interval == 0:
            metrics = agent.update()
            scheduler.step()

        # Periodic logging and checkpointing
        if (step + 1) % eval_interval == 0:
            ckpt_dir = pathlib.Path(log_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"ppo_step_{step+1}.pt"
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "step": step + 1,
                },
                ckpt_path,
            )
            recent = episode_rewards[-50:] if episode_rewards else [0]
            recent_success = episode_successes[-50:] if episode_successes else [0]
            avg_return = np.mean(recent)
            success_rate = np.mean(recent_success)
            current_lr = agent.optimizer.param_groups[0]["lr"]
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{timestamp}] Step {step+1:>6d}/{max_steps} | "
                f"Episodes: {episode_count:>4d} | "
                f"AvgReturn(50): {avg_return:>8.1f} | "
                f"SuccessRate(50): {success_rate:.0%} | "
                f"LR: {current_lr:.2e}"
            )

    # Final save
    ckpt_dir = pathlib.Path(log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_path = ckpt_dir / "ppo_policy_final.pt"
    torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()}, final_path)
    print(f"\nPPO final checkpoint saved to {final_path}")
    if episode_rewards:
        print(f"Final stats: {episode_count} episodes, avg return={np.mean(episode_rewards[-50:]):.1f}, "
              f"success rate={np.mean(episode_successes[-50:]):.0%}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO for crowd evacuation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--log-dir", type=str, default="results/ppo")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--update-interval", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    train_ppo(
        seed=args.seed,
        max_steps=args.max_steps,
        log_dir=args.log_dir,
        device=args.device,
        eval_interval=args.eval_interval,
        update_interval=args.update_interval,
        lr=args.lr,
    )
