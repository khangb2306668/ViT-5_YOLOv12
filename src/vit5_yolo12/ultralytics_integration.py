"""Register the hybrid backbone with the pinned Ultralytics YAML parser."""

from __future__ import annotations

import ultralytics
from ultralytics.nn import tasks

from .hybrid import ViT5HybridBackbone


SUPPORTED_ULTRALYTICS_VERSION = "8.4.185"


def register_ultralytics_modules() -> None:
    """Make ViT5HybridBackbone resolvable by Ultralytics parse_model()."""
    if ultralytics.__version__ != SUPPORTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Hybrid YAML requires Ultralytics {SUPPORTED_ULTRALYTICS_VERSION}; "
            f"found {ultralytics.__version__}"
        )
    tasks.ViT5HybridBackbone = ViT5HybridBackbone
