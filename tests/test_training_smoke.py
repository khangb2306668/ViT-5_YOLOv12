"""Task 5 contracts at the Ultralytics training boundary."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from ultralytics import YOLO

from vit5_yolo12.ultralytics_integration import register_ultralytics_modules


MODEL_YAML = Path(__file__).resolve().parents[1] / "configs" / "yolo12-vit5n.yaml"


def make_dataset(tmp_path: Path) -> Path:
    from vit5_yolo12.config import CLASS_NAMES

    for split in ("train", "valid", "test"):
        images = tmp_path / split / "images"
        labels = tmp_path / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        Image.fromarray(np.full((96, 112, 3), 127, dtype=np.uint8)).save(images / "leaf.jpg")
        (labels / "leaf.txt").write_text(
            "0 0.5 0.5 0.7 0.8\n1 0.5 0.5 0.25 0.3\n", encoding="utf-8"
        )
    data = tmp_path / "data.yaml"
    data.write_text(
        "train: train/images\nval: valid/images\ntest: test/images\nnc: 6\n"
        + "names: " + repr(list(CLASS_NAMES)) + "\n",
        encoding="utf-8",
    )
    return data


def test_trainer_rebuild_loads_vit_then_resume_weights(tmp_path: Path) -> None:
    from vit5_yolo12.ultralytics_integration import HybridDetectionTrainer

    register_ultralytics_modules()
    data = make_dataset(tmp_path)
    original = YOLO(str(MODEL_YAML), task="detect").model
    vit_state = {key: value.cpu().clone() for key, value in original.model[0].vit5.state_dict().items()}
    vit_state["blocks.0.attn.qkv.weight"].fill_(0.125)
    vit_state["head.weight"] = torch.zeros(1000, 384)
    vit_state["head.bias"] = torch.zeros(1000)
    checkpoint = tmp_path / "vit.pth"
    torch.save({"model": vit_state}, checkpoint)
    trainer = HybridDetectionTrainer(
        vit5_checkpoint=checkpoint,
        overrides={"model": str(MODEL_YAML), "data": str(data), "imgsz": 640,
                   "workers": 0, "batch": 1, "project": str(tmp_path / "runs")},
    )
    model = trainer.get_model(cfg=str(MODEL_YAML), verbose=False)
    assert torch.all(model.model[0].vit5.blocks[0].attn.qkv.weight == 0.125)

    resumed = YOLO(str(MODEL_YAML), task="detect").model
    resumed.model[0].vit5.blocks[0].attn.qkv.weight.data.fill_(0.375)
    rebuilt = trainer.get_model(cfg=str(MODEL_YAML), weights=resumed, verbose=False)
    assert torch.all(rebuilt.model[0].vit5.blocks[0].attn.qkv.weight == 0.375)


def test_train_and_validation_dataset_keep_square_inputs_and_overlaps(tmp_path: Path) -> None:
    from vit5_yolo12.ultralytics_integration import HybridDetectionTrainer, HybridDetectionValidator

    register_ultralytics_modules()
    data = make_dataset(tmp_path)
    trainer = HybridDetectionTrainer(
        vit5_checkpoint=tmp_path / "unused.pth",
        overrides={"model": str(MODEL_YAML), "data": str(data), "imgsz": 640,
                   "workers": 0, "batch": 1, "project": str(tmp_path / "runs"),
                   "mosaic": 0.0, "mixup": 0.0, "scale": 0.0},
    )
    trainer.model = YOLO(str(MODEL_YAML), task="detect").model
    for mode, split in (("train", "train"), ("val", "valid")):
        dataset = trainer.build_dataset(str(tmp_path / split / "images"), mode=mode, batch=1)
        sample = dataset[0]
        assert sample["img"].shape == (3, 640, 640)
        assert len(sample["cls"]) == 2
        assert set(sample["cls"].flatten().tolist()) == {0.0, 1.0}
        assert not dataset.rect

    validator = HybridDetectionValidator(args={"model": str(MODEL_YAML), "data": str(data), "imgsz": 640})
    validator.data = trainer.data
    validator.stride = 32
    standalone = validator.build_dataset(str(tmp_path / "valid" / "images"), batch=1)
    assert standalone[0]["img"].shape == (3, 640, 640)
    assert not standalone.rect


def test_training_entry_requires_existing_dataset(tmp_path: Path) -> None:
    from vit5_yolo12.train import train_one_seed
    from vit5_yolo12.config import ExperimentConfig, ConfigError

    config = ExperimentConfig(
        data=tmp_path / "missing.yaml", model_yaml=MODEL_YAML,
        vit5_checkpoint=tmp_path / "missing.pth", output_dir=tmp_path / "runs",
        seeds=(0, 1, 2), imgsz=640,
        train={"epochs": 1, "batch": 1, "multi_scale": False, "rect": False},
        validation={"rect": False}, baseline_ultralytics_version="8.4.185",
    )
    with pytest.raises(ConfigError, match="Cannot read YAML"):
        train_one_seed(config, 0)


def test_preprocess_rejects_multiscale_resize_after_upstream_step(tmp_path: Path, monkeypatch) -> None:
    from vit5_yolo12.ultralytics_integration import HybridDetectionTrainer
    import ultralytics.models.yolo.detect.train as detect_train

    data = make_dataset(tmp_path)
    trainer = HybridDetectionTrainer(
        vit5_checkpoint=tmp_path / "unused.pth",
        overrides={"model": str(MODEL_YAML), "data": str(data), "imgsz": 640,
                   "multi_scale": 0.5, "batch": 1, "workers": 0,
                   "project": str(tmp_path / "runs")},
    )
    trainer.stride = 32
    monkeypatch.setattr(detect_train.random, "randrange", lambda *args: 512)
    with pytest.raises(ValueError, match="640x640"):
        trainer.preprocess_batch({"img": torch.zeros(1, 3, 640, 640, dtype=torch.uint8)})


def test_resume_requires_explicit_checkpoint_and_matching_seed(tmp_path: Path) -> None:
    from vit5_yolo12.train import resume_source, validate_effective_args
    from vit5_yolo12.config import ConfigError

    with pytest.raises(ConfigError, match="explicit checkpoint"):
        resume_source(True)
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"fixture")
    assert resume_source(checkpoint) == str(checkpoint)
    with pytest.raises(ConfigError, match="seed"):
        validate_effective_args(SimpleNamespace(imgsz=640, rect=False, multi_scale=0.0, seed=0), 1)
    with pytest.raises(ConfigError, match="multi_scale"):
        validate_effective_args(SimpleNamespace(imgsz=640, rect=False, multi_scale=0.5, seed=1), 1)
    with pytest.raises(ConfigError, match="rect"):
        validate_effective_args(SimpleNamespace(imgsz=640, rect=True, multi_scale=0.0, seed=1), 1)


def test_metadata_exists_when_training_is_interrupted(tmp_path: Path, monkeypatch) -> None:
    import vit5_yolo12.train as training
    from vit5_yolo12.config import ExperimentConfig

    data = make_dataset(tmp_path)
    checkpoint = tmp_path / "vit.pth"
    checkpoint.write_bytes(b"fixture")
    config = ExperimentConfig(
        data=data, model_yaml=MODEL_YAML, vit5_checkpoint=checkpoint,
        output_dir=tmp_path / "runs", seeds=(0, 1, 2), imgsz=640,
        train={"epochs": 1, "batch": 1, "multi_scale": False, "rect": False},
        validation={"rect": False}, baseline_ultralytics_version="8.4.185",
    )

    class InterruptedYOLO:
        def __init__(self, *args, **kwargs):
            self.callbacks = {}
            self.trainer = SimpleNamespace(
                args=SimpleNamespace(imgsz=640, rect=False, multi_scale=0.0, seed=0),
                save_dir=tmp_path / "runs" / "seed_0",
            )

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

        def train(self, **kwargs):
            self.callbacks["on_pretrain_routine_start"](self.trainer)
            raise RuntimeError("interrupted")

    monkeypatch.setattr(training, "YOLO", InterruptedYOLO)
    with pytest.raises(RuntimeError, match="interrupted"):
        training.train_one_seed(config, 0)
    assert (tmp_path / "runs" / "seed_0" / "metadata.json").is_file()
