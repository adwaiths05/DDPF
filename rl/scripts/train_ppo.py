# scripts/train_ppo.py
"""
Simple PPO training entry‑point with explicit GPU support.

The script automatically selects ``cuda`` if ``--device cuda`` is passed **and** a
compatible GPU is detected via ``torch.cuda.is_available()``.  All tensors –
observations, actions and model parameters – are moved to the chosen device.

Usage example (GPU):
    python scripts/train_ppo.py \
        --seed 123 \
        --max-steps 50000 \
        --log-dir results/ppo \
        --device cuda
"""

import argparse
import pathlib
import random
import sys

import numpy as np
import torch
from torch import nn, optim

# ---- local imports ---------------------------------------------------------
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from src.venue_generator import VenueGenerator
from src.evacuation_env import EvacuationEnv


# ---- a tiny MLP policy ----------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, max_zones: int, max_neighbors: int):
        super().__init__()
        self.max_zones = max_zones
        self.max_neighbors = max_neighbors
        # Output logits for each zone separately (max_zones * max_neighbors)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, max_zones * max_neighbors),  # raw logits per zone
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits shaped (max_zones, max_neighbors)
        logits_flat = self.net(x)
        return logits_flat.view(self.max_zones, self.max_neighbors)

# ---- very small REINFORCE‑style loop (illustrative) -----------------------
def train(args):
    # ---------------------------------------------------------------------
    # Device selection – respect user request and fall back safely
    # ---------------------------------------------------------------------
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on {device}")

    # ---------------------------------------------------------------------
    # Build deterministic venue & environment (seed ensures reproducibility)
    # ---------------------------------------------------------------------
    # Build a deterministic venue; fall back to a guaranteed‑connected grid if random generation fails
    try:
        venue = VenueGenerator.generate_random_venue(seed=args.seed)
    except ValueError as e:
        print(f"[train_ppo] Random venue invalid ({e}); using grid layout instead.")
        venue = VenueGenerator.generate_grid_venue(seed=args.seed)
    env = EvacuationEnv(venue, seed=args.seed)

    obs, _ = env.reset(seed=args.seed)
    obs_dim = obs.shape[0]
    # Action dimensions per zone
    max_zones = env.max_zones
    max_neighbors = env.max_neighbors

    policy = PolicyNet(obs_dim, max_zones, max_neighbors).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # Simple rollout buffers (no GAE, no minibatching – just a sanity‑check)
    observations, actions, rewards = [], [], []

    for step in range(args.max_steps):
        # -----------------------------------------------------------------
        # Forward pass – observation -> logits on the chosen device
        # -----------------------------------------------------------------
        obs_tensor = torch.from_numpy(obs).float().to(device)
        logits = policy(obs_tensor)  # shape (max_zones, max_neighbors)
        # Sample an action per zone independently
        dist = torch.distributions.Categorical(logits=logits)
        actions_per_zone = dist.sample().cpu().numpy()  # length max_zones
        multi_action = actions_per_zone.tolist()

        # -----------------------------------------------------------------
        # Environment step (still operates with NumPy, returns NumPy arrays)
        # -----------------------------------------------------------------
        next_obs, reward, done, trunc, info = env.step(multi_action)

        # Store trajectory items for the REINFORCE update later
        observations.append(obs_tensor)
        actions.append(actions_per_zone.tolist())
        rewards.append(reward)

        obs = next_obs
        if done or trunc:
            # -----------------------------------------------------------------
            # Very naive policy‑gradient update: total return * log‑prob gradient
            # -----------------------------------------------------------------
            total_return = sum(rewards)
            loss = 0.0
            for o, a in zip(observations, actions):
                logits = policy(o)  # (max_zones, max_neighbors)
                a_tensor = torch.tensor(a, dtype=torch.long, device=device)  # shape (max_zones,)
                logp = torch.distributions.Categorical(logits=logits).log_prob(a_tensor).sum()
                loss += -logp * total_return
            loss = loss / len(observations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Reset episode buffers
            observations, actions, rewards = [], [], []
            obs, _ = env.reset(seed=args.seed)

        if (step + 1) % 1000 == 0:
            print(f"Step {step+1}/{args.max_steps} – latest reward {reward:.2f}")

    # ---------------------------------------------------------------------
    # Save the trained policy checkpoint (still a PyTorch state_dict)
    # ---------------------------------------------------------------------
    ckpt_dir = pathlib.Path(args.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ppo_policy.pt"
    torch.save(policy.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--log-dir", type=str, default="results/ppo", help="Where to store checkpoint and logs")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for torch tensors (cuda will be used only if a GPU is detected)",
    )
    args = parser.parse_args()
    train(args)
