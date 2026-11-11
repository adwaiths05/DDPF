# src/mappo/__init__.py
"""MAPPO sub-package exposing core classes.
Exports:
    AgentRoles, CentralizedCritic, MultiAgentEnv, MAPPOAgent, train_mappo
"""
from .agent_roles import AgentRoles
from .centralized_critic import CentralizedCritic
from .multi_agent_env import MultiAgentEnv
from .mappo_agent import MAPPOAgent
from .trainer import train_mappo
