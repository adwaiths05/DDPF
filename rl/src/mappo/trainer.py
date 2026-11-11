# src/mappo/trainer.py
"""Training loop for hierarchical MAPPO.
Utilises ``MultiAgentEnv`` and ``MAPPOAgent``.
The loop collects trajectories for a fixed number of steps, then performs PPO-style
updates for all policies using a shared centralized critic.

Key improvements over the initial version:
* Venue randomisation every episode (better generalisation)
* Frequent updates (every ``update_interval`` steps, default 256)
* Learning rate schedule (linear decay)
* Reward normalisation via running statistics
* Logging of episode-level metrics (success rate, avg return)
"""
import os
import pathlib
from datetime import datetime
import numpy as np
import torch
from .mappo_agent import MAPPOAgent
from .multi_agent_env import MultiAgentEnv
from ..venue_generator import VenueGenerator


class RunningMeanStd:
    """Welford's online algorithm for running reward normalisation."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        return (x - self.mean) / (max(np.sqrt(self.var), 1e-8))


def _make_venue(seed: int):
    """Generate a random venue for training, falling back to grid."""
    try:
        return VenueGenerator.generate_random_venue(seed=seed)
    except Exception:
        return VenueGenerator.generate_grid_venue(seed=seed)


def train_mappo(
    seed: int = 42,
    max_steps: int = 50000,
    log_dir: str = "results/mappo",
    device: str = "cpu",
    eval_interval: int = 2000,
    update_interval: int = 256,
    lr: float = 5e-4,
    critic_lr: float = 1e-3,
    entropy_coef: float = 0.02,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ppo_epochs: int = 6,
    batch_size: int = 64,
    max_grad_norm: float = 0.5,
    normalize_rewards: bool = True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_ = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Training MAPPO on {device_}")
    print(f"  max_steps={max_steps}  update_interval={update_interval}  lr={lr}")
    print(f"  entropy_coef={entropy_coef}  gamma={gamma}  gae_lambda={gae_lambda}")

    # Generate initial venue
    venue = _make_venue(seed)
    env = MultiAgentEnv(venue, seed=seed)

    # Initial observation dict
    per_agent_obs, _ = env.reset(seed=seed)
    sample_obs = next(iter(per_agent_obs.values()))
    zone_obs_dim = sample_obs.shape[0]
    exit_obs_dim = 5
    signage_obs_dim = zone_obs_dim

    agent = MAPPOAgent(
        roles=None,
        device=device_,
        leader_obs_dim=zone_obs_dim,
        zone_obs_dim=zone_obs_dim,
        exit_obs_dim=exit_obs_dim,
        signage_obs_dim=signage_obs_dim,
        max_zones=env.max_zones,
        max_neighbors=env.max_neighbors,
        lr=lr,
        critic_lr=critic_lr,
        entropy_coef=entropy_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
    )

    # LR schedule: linear decay to 10% of initial
    def lr_schedule(step):
        frac = 1.0 - step / max_steps
        return max(frac, 0.1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        agent.optimizer, lr_lambda=lr_schedule
    )

    # Running reward normalisation
    reward_rms = RunningMeanStd()

    # Episode tracking
    episode_rewards = []
    episode_successes = []
    current_episode_reward = 0.0
    episode_count = 0
    venue_seed = seed

    for step in range(max_steps):
        # Get masks for each zone agent
        masks = env.action_masks()
        # Select actions
        actions, logps, values = agent.select_actions(per_agent_obs, masks)
        # Step env (remove leader action -1)
        zone_actions = {aid: act for aid, act in actions.items() if aid != -1}
        next_obs, reward_dict, done, trunc, step_info = env.step(zone_actions)
        is_terminal = done or trunc

        total_reward = next(iter(reward_dict.values()), 0.0)

        # Optionally normalise reward
        if normalize_rewards:
            reward_rms.update_from_moments(total_reward, 0.0, 1)
            norm_reward = reward_rms.normalize(total_reward)
        else:
            norm_reward = total_reward

        agent.store_reward(norm_reward, is_terminal)
        current_episode_reward += total_reward  # track un-normalised for logging
        per_agent_obs = next_obs

        if is_terminal:
            episode_rewards.append(current_episode_reward)
            success = env.base_env.total_evacuated >= env.base_env.initial_population
            episode_successes.append(1 if success else 0)
            current_episode_reward = 0.0
            episode_count += 1

            # Randomise venue every episode for generalisation
            venue_seed += 1
            venue = _make_venue(venue_seed)
            env = MultiAgentEnv(venue, seed=venue_seed)
            per_agent_obs, _ = env.reset(seed=venue_seed)

        # Periodic PPO update
        if (step + 1) % update_interval == 0:
            last_value = 0.0 if is_terminal else agent.estimate_value(per_agent_obs)
            metrics = agent.update(
                last_value=last_value,
                epochs=ppo_epochs,
                batch_size=batch_size,
            )
            scheduler.step()

        # Periodic logging and checkpointing
        if (step + 1) % eval_interval == 0:
            ckpt_dir = pathlib.Path(log_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"mappo_step_{step+1}.pt"
            torch.save(
                {
                    "leader_policy": agent.leader_policy.state_dict(),
                    "zone_policy": agent.zone_policy.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "step": step + 1,
                },
                ckpt_path,
            )
            # Compute recent stats
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
                f"LR: {current_lr:.2e} | "
                f"PL: {metrics.get('policy_loss',0):.4f} | "
                f"VL: {metrics.get('value_loss',0):.4f} | "
                f"Ent: {metrics.get('entropy',0):.4f}"
            )

    # Final save
    final_dir = pathlib.Path(log_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / "mappo_policy_final.pt"
    torch.save(
        {
            "leader_policy": agent.leader_policy.state_dict(),
            "zone_policy": agent.zone_policy.state_dict(),
            "critic": agent.critic.state_dict(),
        },
        final_path,
    )
    print(f"\nMAPPO final checkpoint saved to {final_path}")
    if episode_rewards:
        print(f"Final stats: {episode_count} episodes, avg return={np.mean(episode_rewards[-50:]):.1f}, "
              f"success rate={np.mean(episode_successes[-50:]):.0%}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Hierarchical MAPPO for crowd evacuation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--log-dir", type=str, default="results/mappo")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--update-interval", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--ppo-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    train_mappo(
        seed=args.seed,
        max_steps=args.max_steps,
        log_dir=args.log_dir,
        device=args.device,
        eval_interval=args.eval_interval,
        update_interval=args.update_interval,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
    )
