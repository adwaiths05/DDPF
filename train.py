import argparse
import config as cfg
from engine import run_training_and_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPF-Net Training & Evaluation")
    parser.add_argument("--fold", type=int, default=cfg.CURRENT_FOLD, help="Fold index to run.")
    args = parser.parse_args()
    
    cfg.CURRENT_FOLD = args.fold
    run_training_and_eval()
