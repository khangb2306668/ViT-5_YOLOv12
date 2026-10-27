"""Validated dataset and experiment inputs for the hybrid detector."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


CLASS_NAMES = (
    "alternaria_leaf_spot",
    "colletotrichum_anthracnose",
    "glomerella_anthracnose",
    "healthy",
    "pestalotia_leaf_spot",
    "pink_disease",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ConfigError(ValueError):
    """The dataset or experiment configuration cannot be used safely."""


@dataclass(frozen=True)
class ExperimentConfig:
    data: Path
    model_yaml: Path
    vit5_checkpoint: Path
    output_dir: Path
    seeds: tuple[int, int, int]
    imgsz: int
    train: dict[str, Any]
    validation: dict[str, Any]
    baseline_ultralytics_version: str | None
    yolo12_checkpoint: Path | None = None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read YAML file {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"YAML file {path} must contain a mapping")
    return parsed


def _input_file(raw: Any, key: str, base: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{key} must be a non-empty file path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(f"{key} file does not exist: {path}")
    return path


def _output_path(raw: Any, key: str, base: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{key} must be a non-empty directory path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if path.exists() and not path.is_dir():
        raise ConfigError(f"{key} must name a directory: {path}")
    return path


def check_dataset(data_yaml: Path) -> dict[str, Any]:
    """Check the six-class YOLO dataset layout without modifying its files."""
    yaml_path = Path(data_yaml).expanduser().resolve()
    raw = _read_yaml(yaml_path)
    if type(raw.get("nc")) is not int or raw["nc"] != 6:
        raise ConfigError("Dataset must declare nc=6")
    if raw.get("names") != list(CLASS_NAMES):
        raise ConfigError("Dataset class order must match the six-class contract")

    splits: dict[str, Path] = {}
    for key in ("train", "val", "test"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"Dataset {key} must be an image-directory path")
        image_dir = Path(value).expanduser()
        if not image_dir.is_absolute():
            image_dir = yaml_path.parent / image_dir
        image_dir = image_dir.resolve()
        if not image_dir.is_dir():
            raise ConfigError(f"Missing {key} images directory: {image_dir}")
        if image_dir.name != "images":
            raise ConfigError(f"{key} must point to a directory named images: {image_dir}")
        labels_dir = image_dir.parent / "labels"
        if not labels_dir.is_dir():
            raise ConfigError(f"Missing {key} labels directory: {labels_dir}")
        try:
            images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
            if not images:
                raise ConfigError(f"No images found in {key} images directory: {image_dir}")
            for image in images:
                if not (labels_dir / f"{image.stem}.txt").is_file():
                    raise ConfigError(f"Missing {key} label for {image.name}: {labels_dir / (image.stem + '.txt')}")
        except OSError as exc:
            raise ConfigError(f"Cannot inspect {key} split {image_dir}: {exc}") from exc
        splits[key] = image_dir
    return {"nc": 6, "names": list(CLASS_NAMES), "splits": splits}


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load local paths and enforce the fixed main-comparison settings."""
    config_path = Path(path).expanduser().resolve()
    raw = _read_yaml(config_path)
    base = config_path.parent

    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or seeds != [0, 1, 2] or any(type(s) is not int for s in seeds):
        raise ConfigError("Main comparison seeds must be [0, 1, 2]")
    if type(raw.get("imgsz")) is not int or raw["imgsz"] != 640:
        raise ConfigError("Main comparison imgsz must be 640")

    train = raw.get("train")
    validation = raw.get("validation")
    if not isinstance(train, dict):
        raise ConfigError("train must be a mapping of Ultralytics training options")
    if not isinstance(validation, dict):
        raise ConfigError("validation must be a mapping")
    reserved = {"data", "model", "imgsz", "seed", "project", "name", "save_dir"}
    for section_name, options in (("train", train), ("validation", validation)):
        conflict = reserved.intersection(options)
        if conflict:
            key = sorted(conflict)[0]
            raise ConfigError(f"{section_name}.{key} must be set by the training entry point")
        if (raw.get("yolo12_checkpoint") is not None
                and "pretrained" in options and options["pretrained"] is not True):
            raise ConfigError(f"{section_name}.pretrained conflicts with yolo12_checkpoint; omit it or use true")
    if train.get("multi_scale") is not False:
        raise ConfigError("train.multi_scale must be false for fixed 640 inputs")
    if train.get("rect") is not False:
        raise ConfigError("train.rect must be false for square inputs")
    if validation.get("rect") is not False:
        raise ConfigError("validation.rect must be false for square inputs")
    for key in ("epochs", "batch"):
        if type(train.get(key)) is not int or train[key] <= 0:
            raise ConfigError(f"train.{key} must be a positive integer")

    version = raw.get("baseline_ultralytics_version")
    if version is not None and (not isinstance(version, str) or not version.strip()):
        raise ConfigError("baseline_ultralytics_version must be a version string or null")

    data_path = _input_file(raw.get("data"), "data", base)
    check_dataset(data_path)
    train_options = dict(train)
    if isinstance(train_options.get("resume"), str):
        train_options["resume"] = str(_input_file(train_options["resume"], "train.resume", base))

    return ExperimentConfig(
        data=data_path,
        model_yaml=_input_file(raw.get("model_yaml"), "model_yaml", base),
        vit5_checkpoint=_input_file(raw.get("vit5_checkpoint"), "vit5_checkpoint", base),
        output_dir=_output_path(raw.get("output_dir"), "output_dir", base),
        seeds=tuple(seeds),
        imgsz=640,
        train=train_options,
        validation=dict(validation),
        baseline_ultralytics_version=version,
        yolo12_checkpoint=(
            _input_file(raw["yolo12_checkpoint"], "yolo12_checkpoint", base)
            if raw.get("yolo12_checkpoint") is not None else None
        ),
    )


def build_run_metadata(
    config: ExperimentConfig,
    *,
    ultralytics_version: str,
    vit5_revision: str,
    effective_train_args: dict[str, Any],
    initialization_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture the exact inputs that a later training run must preserve."""
    if not ultralytics_version.strip() or not vit5_revision.strip():
        raise ConfigError("Ultralytics version and ViT-5 revision are required")
    return {
        "dataset_yaml_sha256": hashlib.sha256(config.data.read_bytes()).hexdigest(),
        "vit5_checkpoint_sha256": _sha256_file(config.vit5_checkpoint),
        "yolo12_checkpoint_sha256": _sha256_file(config.yolo12_checkpoint) if config.yolo12_checkpoint else None,
        "initialization": initialization_report,
        "ultralytics_version": ultralytics_version,
        "baseline_ultralytics_version": config.baseline_ultralytics_version,
        "vit5_revision": vit5_revision,
        "effective_train_args": dict(effective_train_args),
        "resolved_config": {
            "data": str(config.data),
            "model_yaml": str(config.model_yaml),
            "vit5_checkpoint": str(config.vit5_checkpoint),
            "yolo12_checkpoint": str(config.yolo12_checkpoint) if config.yolo12_checkpoint else None,
            "output_dir": str(config.output_dir),
            "seeds": list(config.seeds),
            "imgsz": config.imgsz,
            "train": dict(config.train),
            "validation": dict(config.validation),
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_metadata(run_dir: Path, metadata: dict[str, Any]) -> Path:
    """Save reproducibility metadata alongside a run's outputs."""
    output = Path(run_dir) / "metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
