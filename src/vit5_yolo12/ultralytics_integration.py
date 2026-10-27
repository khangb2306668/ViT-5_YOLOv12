"""Register the hybrid backbone with the pinned Ultralytics YAML parser."""

from __future__ import annotations

from dataclasses import asdict

import ultralytics
from ultralytics.data.build import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn import tasks
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils import LOGGER

from .checkpoint import load_vit5_checkpoint
from .hybrid import ViT5HybridBackbone
from .yolo12_checkpoint import load_yolo12_checkpoint


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

    def __init__(self, *, vit5_checkpoint, yolo12_checkpoint=None, **kwargs):
        self.vit5_checkpoint = vit5_checkpoint
        self.yolo12_checkpoint = yolo12_checkpoint
        self.initialization_report = {"mode": "pending", "yolo12_applied": False}
        if yolo12_checkpoint is not None and kwargs.get("overrides", {}).get("pretrained", True) is not True:
            raise ValueError("pretrained conflicts with yolo12_checkpoint; omit it or use true")
        super().__init__(**kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        register_ultralytics_modules()
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        backbone = unwrap_model(model).model[0]
        vit5_report = load_vit5_checkpoint(backbone, self.vit5_checkpoint)
        if weights is not None:
            if self.yolo12_checkpoint is not None:
                source = unwrap_model(weights)
                source_state = source.state_dict()
                target_state = model.state_dict()
                if (not hasattr(source, "model") or not isinstance(source.model[0], ViT5HybridBackbone)
                        or source_state.keys() != target_state.keys()
                        or any(source_state[key].shape != value.shape for key, value in target_state.items())):
                    raise ValueError("Initialization B requires full compatible hybrid weights for resume/finetune")
            model.load(weights)  # Resume/finetuned weights take priority over base ViT.
            self.initialization_report = {
                "mode": "hybrid_weights", "yolo12_applied": False,
                "source": str(getattr(weights, "pt_path", "provided hybrid model")),
            }
        elif self.yolo12_checkpoint is not None:
            report = load_yolo12_checkpoint(model, self.yolo12_checkpoint)
            self.initialization_report = {
                "mode": "vit5_yolo12_b", "yolo12_applied": True,
                "vit5": asdict(vit5_report), "yolo12": asdict(report),
            }
            LOGGER.info(
                f"Initialization B: {sum(len(group.mappings) for group in report.groups)} tensors, "
                f"{sum(group.parameters for group in report.groups):,} parameters from {report.source} "
                "(P3, neck, box; classifier and adapters initialized anew)"
            )
        else:
            self.initialization_report = {
                "mode": "vit5_only", "yolo12_applied": False, "vit5": asdict(vit5_report),
            }
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
