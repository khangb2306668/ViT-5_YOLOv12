"""Strict, tensor-only loading of the official ViT-5 checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
from pathlib import Path
import pickle

import torch

from .hybrid import ViT5HybridBackbone


class CheckpointError(ValueError):
    """The checkpoint does not match the ViT-5 encoder contract."""


@dataclass(frozen=True)
class LoadReport:
    source: str
    prefix: str
    loaded: tuple[str, ...]
    skipped: tuple[str, ...]
    unexpected: tuple[str, ...]


_PREFIXES = (
    "",
    "module.",
    "model.",
    "vit5.",
    "backbone.vit5.",
    "module.vit5.",
    "module.backbone.vit5.",
)
_CLASSIFIER_KEYS = {"head.weight", "head.bias"}


def _select_state(checkpoint: object, prefer_ema: bool) -> tuple[str, Mapping[str, torch.Tensor]]:
    if not isinstance(checkpoint, Mapping):
        raise CheckpointError("Checkpoint must contain a tensor mapping")
    if prefer_ema:
        if not isinstance(checkpoint.get("model_ema"), Mapping):
            raise CheckpointError("EMA weights requested but model_ema is missing")
        return "model_ema", checkpoint["model_ema"]
    for key in ("model", "state_dict"):
        if isinstance(checkpoint.get(key), Mapping):
            return key, checkpoint[key]
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return "root", checkpoint
    raise CheckpointError("Checkpoint has no model or state_dict tensor mapping")


def _strip_prefix(
    source: Mapping[str, torch.Tensor], target_keys: set[str]
) -> tuple[str, dict[str, torch.Tensor]]:
    if not source or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in source.items()):
        raise CheckpointError("Checkpoint state must map string keys to tensors")
    best_prefix = ""
    best_score = -1
    best_state: dict[str, torch.Tensor] = {}
    for prefix in _PREFIXES:
        if prefix and not all(key.startswith(prefix) for key in source):
            continue
        stripped = {key[len(prefix) :]: value for key, value in source.items()}
        score = len(target_keys.intersection(stripped))
        if score > best_score:
            best_prefix, best_score, best_state = prefix, score, stripped
    if best_score == 0:
        raise CheckpointError("Checkpoint contains no matching ViT-5 backbone weights")
    return best_prefix, best_state


def load_vit5_checkpoint(
    backbone: ViT5HybridBackbone, path: Path, *, prefer_ema: bool = False
) -> LoadReport:
    """Load matching encoder tensors only; reject all unexplained differences."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise CheckpointError(f"Checkpoint file does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"Cannot read tensor-only checkpoint {path}: {exc}") from exc

    source_name, source_state = _select_state(checkpoint, prefer_ema)
    target = backbone.vit5.state_dict()
    prefix, source_state = _strip_prefix(source_state, set(target))
    skipped = tuple(sorted(key for key in source_state if key in _CLASSIFIER_KEYS))
    weights = {key: value for key, value in source_state.items() if key not in _CLASSIFIER_KEYS}
    unexpected = tuple(sorted(set(weights) - set(target)))
    missing = tuple(sorted(set(target) - set(weights)))
    mismatch = tuple(
        sorted(key for key in set(weights) & set(target) if weights[key].shape != target[key].shape)
    )
    if unexpected:
        raise CheckpointError(f"Unexpected ViT-5 keys: {unexpected}")
    if missing:
        raise CheckpointError(f"Missing ViT-5 keys: {missing}")
    if mismatch:
        raise CheckpointError(f"ViT-5 tensor shape mismatch: {mismatch}")

    layout_mismatch = tuple(
        sorted(key for key in weights if weights[key].layout != target[key].layout)
    )
    if layout_mismatch:
        raise CheckpointError(f"ViT-5 tensor layout mismatch: {layout_mismatch}")

    # Prove all copies work on an isolated encoder before changing the live one.
    candidate = copy.deepcopy(backbone.vit5)
    try:
        candidate.load_state_dict(weights, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise CheckpointError(f"Cannot apply ViT-5 weights: {exc}") from exc
    original = {key: value.detach().clone() for key, value in target.items()}
    try:
        backbone.vit5.load_state_dict(candidate.state_dict(), strict=True)
    except (RuntimeError, TypeError) as exc:
        backbone.vit5.load_state_dict(original, strict=True)
        raise CheckpointError(f"Cannot commit ViT-5 weights: {exc}") from exc
    return LoadReport(
        source=source_name,
        prefix=prefix,
        loaded=tuple(sorted(weights)),
        skipped=skipped,
        unexpected=unexpected,
    )
