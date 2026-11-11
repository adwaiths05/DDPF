# scripts/train_mappo.py
"""Standalone script to train the hierarchical MAPPO policy.
Runs the full MAPPO training loop via src.mappo.trainer.

Usage:
    python scripts/train_mappo.py --seed 42 --max-steps 10000 --device cuda
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.mappo.trainer import train_mappo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Hierarchical MAPPO for crowd evacuation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--log-dir", type=str, default="results/mappo", help="Checkpoint output directory")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Training device")
    parser.add_argument("--eval-interval", type=int, default=1000, help="Steps between checkpoints")
    args = parser.parse_args()
    train_mappo(
        seed=args.seed,
        max_steps=args.max_steps,
        log_dir=args.log_dir,
        device=args.device,
        eval_interval=args.eval_interval,
    )
