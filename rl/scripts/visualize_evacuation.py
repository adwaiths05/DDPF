"""Animate initial crowd distribution and PPO/MAPPO evacuation side by side.

Examples:
    python scripts/visualize_evacuation.py
    python scripts/visualize_evacuation.py --max-steps 50 --save results/evacuation.gif
    python scripts/visualize_evacuation.py --no-show --save results/evacuation.gif
"""
import pathlib
import sys
from dataclasses import dataclass
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.evacuation_env import EvacuationEnv
from src.mappo.mappo_agent import MAPPOAgent
from src.mappo.multi_agent_env import MultiAgentEnv
from src.ppo.actor_critic import Actor
from src.venue_generator import VenueGenerator, EDGE_CAPACITY, IS_BLOCKED, NODE_TYPE, POS


@dataclass
class State:
    step: int
    occupancy: dict
    evacuated: int
    initial_population: int
    edge_congestion: dict


def generate_venue(seed: int):
    try:
        return VenueGenerator.generate_random_venue(seed=seed)
    except ValueError:
        return VenueGenerator.generate_grid_venue(seed=seed)


def snapshot(env, step: int) -> State:
    return State(
        step=step,
        occupancy=dict(env.zone_occupancy),
        evacuated=env.total_evacuated,
        initial_population=env.initial_population,
        edge_congestion=dict(env.edge_congestion),
    )


def load_ppo(checkpoint_path: str, env: EvacuationEnv, device: torch.device):
    actor = Actor(
        env.observation_space.shape[0],
        env.max_zones,
        env.max_neighbors,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


def load_mappo(checkpoint_path: str, env: MultiAgentEnv, obs: dict, device: torch.device):
    zone_obs_dim = next(iter(obs.values())).shape[0]
    agent = MAPPOAgent(
        roles=None,
        device=device,
        leader_obs_dim=zone_obs_dim,
        zone_obs_dim=zone_obs_dim,
        exit_obs_dim=5,
        signage_obs_dim=zone_obs_dim,
        max_zones=env.max_zones,
        max_neighbors=env.max_neighbors,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.zone_policy.load_state_dict(checkpoint["zone_policy"])
    agent.zone_policy.eval()
    return agent


def run_ppo(env: EvacuationEnv, actor, seed: int, max_steps: int, device: torch.device, deterministic: bool = False):
    obs, _ = env.reset(seed=seed)
    states = [snapshot(env, 0)]
    for step in range(1, max_steps + 1):
        with torch.no_grad():
            logits = actor(torch.from_numpy(obs).float().to(device))
            logits = logits.view(env.max_zones, env.max_neighbors)
            mask = torch.from_numpy(env.action_masks()).to(device)
            masked = logits.masked_fill(mask == 0, -1e10)
            if deterministic:
                action = masked.argmax(dim=-1).cpu().numpy()
            else:
                action = torch.distributions.Categorical(logits=masked).sample().cpu().numpy()
        obs, _, done, trunc, _ = env.step(action)
        states.append(snapshot(env, step))
        if done or trunc:
            break
    return states


def run_mappo(env: MultiAgentEnv, agent, seed: int, max_steps: int, device: torch.device, deterministic: bool = False):
    obs, _ = env.reset(seed=seed)
    states = [snapshot(env.base_env, 0)]
    for step in range(1, max_steps + 1):
        masks = env.action_masks()
        zone_actions = {}
        with torch.no_grad():
            for zid, zone_obs in obs.items():
                logits = agent.zone_policy(
                    torch.from_numpy(zone_obs).float().to(device)
                )
                mask = torch.from_numpy(masks[zid]).float().to(device)
                masked = logits.masked_fill(mask == 0, -1e10)
                if deterministic:
                    zone_actions[zid] = int(masked.argmax().item())
                else:
                    zone_actions[zid] = int(
                        torch.distributions.Categorical(logits=masked).sample().item()
                    )
        obs, _, done, trunc, _ = env.step(zone_actions)
        states.append(snapshot(env.base_env, step))
        if done or trunc:
            break
    return states


def draw_state(ax, venue, state: State, title: str):
    ax.clear()
    graph = venue.graph
    positions = {node: graph.nodes[node][POS] for node in graph.nodes}

    edge_colors = []
    edge_widths = []
    for source, target, data in graph.edges(data=True):
        congestion = state.edge_congestion.get((source, target), 0)
        capacity = max(1, data[EDGE_CAPACITY])
        ratio = min(1.0, congestion / capacity)
        edge_colors.append((0.2 + 0.8 * ratio, 0.65 - 0.4 * ratio, 0.2))
        edge_widths.append(1.0 + 2.5 * ratio)

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=11,
        edge_color=edge_colors,
        width=edge_widths,
        connectionstyle="arc3,rad=0.04",
    )

    zone_nodes = list(venue.zones)
    exit_nodes = list(venue.exits)
    zone_values = [state.occupancy.get(node, 0) for node in zone_nodes]
    zone_colors = plt.cm.Oranges(
        np.asarray(zone_values) / max(1, max(zone_values, default=1))
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=zone_nodes,
        node_color=zone_colors,
        node_size=850,
        node_shape="o",
        edgecolors="#7a3e00",
        linewidths=1.2,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=exit_nodes,
        node_color="#2ca25f",
        node_size=950,
        node_shape="s",
        edgecolors="#145a32",
        linewidths=1.5,
        ax=ax,
    )

    labels = {
        node: f"Z{node}\n{state.occupancy.get(node, 0)}"
        for node in zone_nodes
    }
    labels.update({node: f"EXIT\n{node}" for node in exit_nodes})
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=8, ax=ax)

    evacuated_pct = 100.0 * state.evacuated / max(1, state.initial_population)
    ax.set_title(
        f"{title}\nStep {state.step} | Evacuated: {state.evacuated}/"
        f"{state.initial_population} ({evacuated_pct:.1f}%)",
        fontsize=11,
    )
    ax.set_axis_off()


def visualize(
    ppo_checkpoint: str,
    mappo_checkpoint: str,
    seed: int,
    max_steps: int,
    save_path: str | None,
    show: bool,
    deterministic: bool = False,
):
    device = torch.device("cpu")
    venue = generate_venue(seed)

    ppo_env = EvacuationEnv(venue, seed=seed, max_steps=max_steps)
    mappo_env = MultiAgentEnv(venue, seed=seed)
    mappo_env.base_env.max_steps = max_steps
    mappo_obs, _ = mappo_env.reset(seed=seed)

    ppo_actor = load_ppo(ppo_checkpoint, ppo_env, device)
    mappo_agent = load_mappo(mappo_checkpoint, mappo_env, mappo_obs, device)
    ppo_states = run_ppo(ppo_env, ppo_actor, seed, max_steps, device, deterministic=deterministic)
    mappo_states = run_mappo(mappo_env, mappo_agent, seed, max_steps, device, deterministic=deterministic)
    frame_count = max(len(ppo_states), len(mappo_states))
    initial_frames = 8

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    fig.suptitle(
        f"Crowd Evacuation | {len(venue.zones)} zones, {len(venue.exits)} exits | "
        "Orange = crowd, green = exit",
        fontsize=14,
    )

    def get_state(states, index):
        return states[min(index, len(states) - 1)]

    def update(frame):
        if frame < initial_frames:
            ppo_state = get_state(ppo_states, 0)
            mappo_state = get_state(mappo_states, 0)
            fig.suptitle(
                "Initial crowd distribution | zone labels show people in each zone",
                fontsize=14,
            )
        else:
            state_index = frame - initial_frames
            ppo_state = get_state(ppo_states, state_index)
            mappo_state = get_state(mappo_states, state_index)
            mode = "deterministic" if deterministic else "stochastic"
            fig.suptitle(
                f"Evacuation comparison | {mode} PPO vs {mode} MAPPO",
                fontsize=14,
            )
        draw_state(axes[0], venue, ppo_state, "PPO")
        draw_state(axes[1], venue, mappo_state, "MAPPO")
        return axes

    animation = FuncAnimation(
        fig,
        update,
        frames=initial_frames + frame_count,
        interval=350,
        repeat=False,
        blit=False,
    )
    if save_path:
        output = pathlib.Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() != ".gif":
            raise ValueError("Animation output must use the .gif extension")
        animation.save(output, writer=PillowWriter(fps=3))
        print(f"Evacuation animation saved to {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Animate PPO and MAPPO evacuation")
    parser.add_argument(
        "--ppo-checkpoint",
        default="results/ppo/ppo_policy_final.pt",
        help="PPO checkpoint path",
    )
    parser.add_argument(
        "--mappo-checkpoint",
        default="results/mappo/mappo_policy_final.pt",
        help="MAPPO checkpoint path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--save", default=None, help="GIF output path")
    parser.add_argument("--no-show", action="store_true", help="Do not open a window")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use greedy argmax instead of stochastic sampling (default: stochastic)")
    args = parser.parse_args()

    visualize(
        ppo_checkpoint=args.ppo_checkpoint,
        mappo_checkpoint=args.mappo_checkpoint,
        seed=args.seed,
        max_steps=args.max_steps,
        save_path=args.save,
        show=not args.no_show,
        deterministic=args.deterministic,
    )
