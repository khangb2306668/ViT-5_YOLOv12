"""Register the hybrid backbone with the pinned Ultralytics YAML parser."""

from __future__ import annotations

import ultralytics
from ultralytics.data.build import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn import tasks
from ultralytics.utils.torch_utils import unwrap_model

from .checkpoint import load_vit5_checkpoint
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


def _require_square(batch: dict, imgsz: int) -> None:
    if tuple(batch["img"].shape[-2:]) != (imgsz, imgsz):
        raise ValueError(f"Hybrid model requires {imgsz}x{imgsz} batches; got {tuple(batch['img'].shape[-2:])}")


class HybridDetectionValidator(DetectionValidator):
    """Use Ultralytics detection metrics with fixed square images."""

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode,
            rect=False, stride=self.stride,
        )

    def preprocess(self, batch: dict) -> dict:
        _require_square(batch, self.args.imgsz)
        return super().preprocess(batch)


class HybridDetectionTrainer(DetectionTrainer):
    """Use Ultralytics training with the official ViT initialization."""

    def __init__(self, *, vit5_checkpoint, **kwargs):
        self.vit5_checkpoint = vit5_checkpoint
        super().__init__(**kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        register_ultralytics_modules()
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        backbone = unwrap_model(model).model[0]
        load_vit5_checkpoint(backbone, self.vit5_checkpoint)
        if weights is not None:
            model.load(weights)  # Resume/finetuned weights take priority over base ViT.
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode,
            rect=False, stride=gs,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        _require_square(batch, self.args.imgsz)
        batch = super().preprocess_batch(batch)
        _require_square(batch, self.args.imgsz)
        return batch

    def get_validator(self):
        from copy import copy

        return HybridDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
