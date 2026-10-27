"""Three-seed training through Ultralytics' native detection trainer."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import ultralytics
from ultralytics import YOLO

from .config import ConfigError, ExperimentConfig, build_run_metadata, check_dataset, write_run_metadata
from .ultralytics_integration import (
    HybridDetectionTrainer,
    SUPPORTED_ULTRALYTICS_VERSION,
    register_ultralytics_modules,
)


VIT5_REVISION = "b7b0796062f4b51ae00523cb77c7117ae2dff536"


def resume_source(resume: object) -> str | None:
    """Require an explicit path so Ultralytics cannot resume an unrelated latest run."""
    if resume is True:
        raise ConfigError("Resume requires an explicit checkpoint path")
    if resume is False or resume is None:
        return None
    if not isinstance(resume, (str, Path)) or not str(resume).strip():
        raise ConfigError("Resume requires an explicit checkpoint path")
    path = Path(resume).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"Resume checkpoint does not exist: {path}")
    return str(path)


def validate_effective_args(args: object, seed: int) -> None:
    """Check the args Ultralytics actually uses after applying resume state."""
    if args.imgsz != 640:
        raise ConfigError(f"Effective imgsz must be 640, found {args.imgsz}")
    if args.rect:
        raise ConfigError("Effective rect must be false")
    if args.multi_scale:
        raise ConfigError("Effective multi_scale must be false")
    if args.seed != seed:
        raise ConfigError(f"Effective seed {args.seed} does not match requested seed {seed}")


def train_one_seed(config: ExperimentConfig, seed: int) -> Path:
    """Train one seed and return its saved best (or last) checkpoint."""
    if seed not in config.seeds:
        raise ConfigError(f"Seed {seed} is outside the configured experiment seeds")
    check_dataset(config.data)
    if not config.vit5_checkpoint.is_file():
        raise ConfigError(f"ViT-5 checkpoint does not exist: {config.vit5_checkpoint}")
    if ultralytics.__version__ != SUPPORTED_ULTRALYTICS_VERSION:
        raise RuntimeError(f"Expected Ultralytics {SUPPORTED_ULTRALYTICS_VERSION}; found {ultralytics.__version__}")
    register_ultralytics_modules()

    resume = resume_source(config.train.get("resume", False))
    model_source = resume or str(config.model_yaml)
    model = YOLO(model_source, task="detect")

    def record_run(trainer: HybridDetectionTrainer) -> None:
        validate_effective_args(trainer.args, seed)
        effective_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(trainer.args).items()
        }
        metadata = build_run_metadata(
            config,
            ultralytics_version=ultralytics.__version__,
            vit5_revision=VIT5_REVISION,
            effective_train_args=effective_args,
            initialization_report=getattr(trainer, "initialization_report", {"mode": "pending"}),
        )
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None:
            keys = {"lr", "initial_lr", "momentum", "betas", "weight_decay", "param_group", "use_muon"}
            metadata["optimizer"] = {
                "name": type(optimizer).__name__,
                "groups": [{key: value for key, value in group.items() if key in keys}
                           for group in optimizer.param_groups],
            }
        write_run_metadata(trainer.save_dir, metadata)

    model.add_callback("on_pretrain_routine_start", record_run)
    model.add_callback("on_pretrain_routine_end", record_run)
    options = {
        **config.train,
        **config.validation,
        "data": str(config.data),
        "imgsz": config.imgsz,
        "seed": seed,
        "project": str(config.output_dir),
        "name": f"seed_{seed}",
    }
    model.train(trainer=partial(
        HybridDetectionTrainer, vit5_checkpoint=config.vit5_checkpoint,
        yolo12_checkpoint=config.yolo12_checkpoint,
    ), **options)
    trainer = model.trainer
    checkpoint = trainer.best if trainer.best.exists() else trainer.last
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training did not save a checkpoint in {trainer.save_dir}")
    return checkpoint
