"""Keep Ultralytics test settings inside this repository's ignored workspace."""

import os
from pathlib import Path


os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))
