"""Validate a YOLO dataset manifest before starting a training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vit5_yolo12.config import ConfigError, check_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_yaml", type=Path, help="Path to dataset data.yaml")
    args = parser.parse_args()
    try:
        result = check_dataset(args.data_yaml)
    except ConfigError as exc:
        parser.exit(2, f"Dataset check failed: {exc}\n")
    print(json.dumps(result, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
