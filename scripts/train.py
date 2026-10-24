"""Train the ViT-5/YOLO12n hybrid for seeds 0, 1, and 2."""

from __future__ import annotations

import argparse
from pathlib import Path

from vit5_yolo12.config import load_experiment_config
from vit5_yolo12.train import train_one_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Local experiment YAML")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), help="Run one seed (required for resume)")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if config.train.get("resume") and args.seed is None:
        parser.error("--seed is required when resuming a checkpoint")
    for seed in (args.seed,) if args.seed is not None else config.seeds:
        print(f"Seed {seed}: {train_one_seed(config, seed)}")


if __name__ == "__main__":
    main()
