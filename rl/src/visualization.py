# src/visualization.py
"""Visualization utilities for the crowd‑evacuation RL module.
Provides quick plots for:
* Venue layout (rooms, exits, corridors)
* Training reward curve (from a CSV log file)
* Episode trajectories (agent positions over time – placeholder).
The functions rely only on ``matplotlib`` and ``seaborn`` which are already in the
project dependencies.  They are deliberately lightweight so they can be imported
without triggering heavyweight simulation code.
"""
import os
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_venue(venue, save_path: str | None = None):
    """Render a simple 2‑D representation of a venue.
    ``venue`` is expected to be the object returned by ``VenueGenerator`` and must
    provide ``rooms`` (list of ``(x, y, w, h)``) and ``exits`` (list of ``(x, y)``).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for (x, y, w, h) in getattr(venue, "rooms", []):
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="#4a90e2", linewidth=2)
        ax.add_patch(rect)
    for (ex, ey) in getattr(venue, "exits", []):
        ax.plot(ex, ey, "s", markersize=12, color="#e94e77")
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Synthetic Venue Layout")
    ax.grid(True, linestyle="--", alpha=0.5)
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Venue plot saved to {save_path}")
    plt.show()
    return fig, ax


def plot_training_curve(csv_path: str, metric: str = "reward", save_path: str | None = None):
    """Plot a training metric over time from a CSV log.
    The CSV is expected to have a ``step`` column and the chosen ``metric`` column.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "step" not in df.columns or metric not in df.columns:
        raise ValueError(f"CSV must contain 'step' and '{metric}' columns")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=df, x="step", y=metric, ax=ax, errorbar=None)
    ax.set_title(f"Training {metric.capitalize()} Curve")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(metric.capitalize())
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Training curve saved to {save_path}")
    plt.show()
    return fig, ax


def plot_episode_trajectory(positions: np.ndarray, venue=None, save_path: str | None = None):
    """Visualize crowd positions over a single episode.
    ``positions`` should be an array of shape ``(T, N, 2)`` where ``T`` is the
    number of timesteps, ``N`` agents, and the last dimension are ``(x, y)``
    coordinates.  If a ``venue`` object is provided, its layout is drawn underneath.
    """
    T, N, _ = positions.shape
    fig, ax = plt.subplots(figsize=(8, 6))
    if venue is not None:
        # draw simple venue outline (reuse plot_venue logic without showing)
        for (x, y, w, h) in getattr(venue, "rooms", []):
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="#4a90e2", linewidth=1)
            ax.add_patch(rect)
        for (ex, ey) in getattr(venue, "exits", []):
            ax.plot(ex, ey, "s", markersize=8, color="#e94e77")
    # animate by drawing each timestep as a scatter (static for now)
    for t in range(T):
        ax.cla()
        if venue is not None:
            for (x, y, w, h) in getattr(venue, "rooms", []):
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="#4a90e2", linewidth=1)
                ax.add_patch(rect)
            for (ex, ey) in getattr(venue, "exits", []):
                ax.plot(ex, ey, "s", markersize=8, color="#e94e77")
        ax.scatter(positions[t, :, 0], positions[t, :, 1], s=20, alpha=0.7, color="#ff7f0e")
        ax.set_title(f"Episode trajectory – step {t+1}/{T}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.set_xlim(0, getattr(venue, "width", 20))
        ax.set_ylim(0, getattr(venue, "height", 20))
        plt.pause(0.05)
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Trajectory plot saved to {save_path}")
    plt.show()
    return fig, ax
