"""Strict P3/neck/box transfer from an unfused COCO YOLO12n checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import ultralytics
from ultralytics.nn.tasks import DetectionModel, load_checkpoint

from .hybrid import ViT5HybridBackbone


class Yolo12CheckpointError(ValueError):
    """A checkpoint cannot provide initialization B for the hybrid graph."""


@dataclass(frozen=True)
class TransferGroup:
    name: str
    mappings: tuple[tuple[str, str], ...]
    parameters: int


@dataclass(frozen=True)
class Yolo12LoadReport:
    source: str
    groups: tuple[TransferGroup, ...]
    skipped_source: tuple[str, ...]
    untouched_target: tuple[str, ...]


def _module_signature(module: nn.Module) -> tuple:
    attributes = ("kernel_size", "stride", "padding", "dilation", "groups",
                  "num_features", "eps", "momentum", "affine", "track_running_stats",
                  "scale_factor", "mode", "align_corners", "padding_mode",
                  "c", "add", "d", "area", "num_heads", "head_dim", "all_head_dim",
                  "nc", "nl", "reg_max", "end2end")

    def value(child, key):
        item = getattr(child, key)
        return tuple(item.tolist()) if isinstance(item, torch.Tensor) else item

    return tuple(
        (name, type(child), tuple((key, value(child, key)) for key in attributes if hasattr(child, key)))
        for name, child in module.named_modules()
    )


def _check_module(source: nn.Module, target: nn.Module, label: str, *, allowed_extra=frozenset()) -> None:
    if _module_signature(source) != _module_signature(target):
        raise Yolo12CheckpointError(f"Module structure mismatch: {label}")
    source_state, target_state = source.state_dict(), target.state_dict()
    extra = set(source_state) - set(target_state)
    if set(target_state) - set(source_state) or extra - allowed_extra:
        raise Yolo12CheckpointError(f"State keys mismatch: {label}; expected an unfused model")
    for key, tensor in source_state.items():
        if key in extra:
            conv = target.get_submodule(key.rsplit(".", 1)[0])
            if tensor.shape != (conv.out_channels,) or tensor.layout != torch.strided:
                raise Yolo12CheckpointError(f"Legacy positional bias mismatch: {label}.{key}")
            continue
        other = target_state[key]
        if tensor.shape != other.shape or tensor.layout != other.layout:
            raise Yolo12CheckpointError(f"Tensor shape/layout mismatch: {label}.{key}")


def _validate_source(source: nn.Module) -> None:
    # Reference construction must not perturb the run's RNG or newly initialized groups.
    with torch.random.fork_rng(devices=[]):
        reference = DetectionModel("yolo12n.yaml", nc=80, verbose=False)
    if not isinstance(source, DetectionModel) or len(source.model) != len(reference.model):
        raise Yolo12CheckpointError("Source must be an unfused COCO YOLO12n detection model")
    if source.model[-1].nc != 80 or source.stride.tolist() != [8.0, 16.0, 32.0]:
        raise Yolo12CheckpointError("Source must have 80 COCO classes and strides [8,16,32]")
    for index, (actual, expected) in enumerate(zip(source.model, reference.model)):
        if actual.f != expected.f:
            raise Yolo12CheckpointError(f"Source graph mismatch at layer {index}")
        # Official older checkpoints have PE convolution biases in the two unused
        # attention backbone stages. They are never copied into this hybrid.
        allowed = {
            f"{name}.bias" for name, module in expected.named_modules()
            if index in (6, 8) and name.endswith("attn.pe.conv") and isinstance(module, nn.Conv2d)
        }
        _check_module(actual, expected, f"source.model.{index}", allowed_extra=allowed)


def load_yolo12_checkpoint(model: DetectionModel, path: Path) -> Yolo12LoadReport:
    """Preflight the complete transfer, then atomically copy parameters and buffers."""
    if ultralytics.__version__ != "8.4.185":
        raise Yolo12CheckpointError("Initialization B requires Ultralytics 8.4.185")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise Yolo12CheckpointError(f"YOLO12n checkpoint does not exist: {path}")
    try:
        source, _ = load_checkpoint(str(path), device=torch.device("cpu"), fuse=False)
    except Exception as exc:
        raise Yolo12CheckpointError(f"Cannot read YOLO12n checkpoint {path}: {exc}") from exc
    _validate_source(source)
    if (not isinstance(model, DetectionModel) or len(model.model) != 17
            or not isinstance(model.model[0], ViT5HybridBackbone)
            or model.model[-1].nc != 6 or model.stride.tolist() != [8.0, 16.0, 32.0]):
        raise Yolo12CheckpointError("Target must be the six-class ViT-5/YOLO12n hybrid")

    groups = (
        ("p3", [(f"model.{i}", f"model.0.p3.{i}") for i in range(5)]),
        ("neck", [(f"model.{i}", f"model.{i - 5}") for i in range(9, 21)]),
        ("box", [("model.21.cv2", "model.16.cv2")]),
    )
    backbone_refs = {4: 1, 6: 2, 8: 3}

    def remap(index):
        return -1 if index == -1 else index - 5 if index >= 9 else backbone_refs[index]

    for old_index in range(9, 22):
        old_from = source.model[old_index].f
        expected = [remap(i) for i in old_from] if isinstance(old_from, list) else remap(old_from)
        if model.model[old_index - 5].f != expected:
            raise Yolo12CheckpointError(f"Target graph mismatch at layer {old_index - 5}")

    prepared: dict[str, torch.Tensor] = {}
    reports = []
    used_source = set()
    target_state = model.state_dict()
    for name, pairs in groups:
        mappings, parameters = [], 0
        for source_prefix, target_prefix in pairs:
            original = source.get_submodule(source_prefix)
            destination = model.get_submodule(target_prefix)
            _check_module(original, destination, f"{source_prefix} -> {target_prefix}")
            parameters += sum(param.numel() for param in destination.parameters())
            for key, tensor in original.state_dict().items():
                source_key, target_key = f"{source_prefix}.{key}", f"{target_prefix}.{key}"
                other = target_state[target_key]
                value = tensor.detach().to(device=other.device, dtype=other.dtype).clone()
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise Yolo12CheckpointError(f"Nonfinite source tensor: {source_key}")
                prepared[target_key] = value
                used_source.add(source_key)
                mappings.append((source_key, target_key))
        reports.append(TransferGroup(name, tuple(mappings), parameters))

    backups = {key: target_state[key].clone() for key in prepared}
    with torch.no_grad():
        try:
            for key, value in prepared.items():
                target_state[key].copy_(value)
        except (RuntimeError, TypeError) as exc:
            for key, value in backups.items():
                target_state[key].copy_(value)
            raise Yolo12CheckpointError(f"Cannot commit initialization B: {exc}") from exc
    return Yolo12LoadReport(
        source=str(path), groups=tuple(reports),
        skipped_source=tuple(sorted(set(source.state_dict()) - used_source)),
        untouched_target=tuple(sorted(set(target_state) - set(prepared))),
    )
