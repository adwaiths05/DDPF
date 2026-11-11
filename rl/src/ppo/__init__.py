# src/ppo/__init__.py
"""PPO sub‑package exposing core components and training entry point."""

from .actor_critic import Actor, Critic
from .rollout_buffer import RolloutBuffer
from .ppo_agent import PPOAgent
from .trainer import train_ppo
