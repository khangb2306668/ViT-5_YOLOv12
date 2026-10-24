from pathlib import Path

import pytest
import yaml

from vit5_yolo12.config import (
    ConfigError,
    build_run_metadata,
    load_experiment_config,
    write_run_metadata,
)


def write_config(tmp_path: Path, **changes) -> Path:
    data = tmp_path / "data.yaml"
    model = tmp_path / "model.yaml"
    checkpoint = tmp_path / "vit5.pth"
    for split in ("train", "valid", "test"):
        image_dir = tmp_path / split / "images"
        label_dir = tmp_path / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / "sample.jpg").write_bytes(b"image placeholder")
        (label_dir / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    data.write_text(
        yaml.safe_dump(
            {
                "nc": 6,
                "names": [
                    "alternaria_leaf_spot",
                    "colletotrichum_anthracnose",
                    "glomerella_anthracnose",
                    "healthy",
                    "pestalotia_leaf_spot",
                    "pink_disease",
                ],
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
            }
        ),
        encoding="utf-8",
    )
    for path in (model, checkpoint):
        path.write_bytes(b"placeholder")
    raw = {
        "data": "data.yaml",
        "model_yaml": "model.yaml",
        "vit5_checkpoint": "vit5.pth",
        "output_dir": "runs",
        "seeds": [0, 1, 2],
        "imgsz": 640,
        "train": {"epochs": 100, "batch": 16, "multi_scale": False, "rect": False},
        "validation": {"rect": False},
    }
    raw.update(changes)
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config


def test_config_resolves_paths_and_keeps_training_options(tmp_path: Path) -> None:
    config = load_experiment_config(write_config(tmp_path))

    assert config.data == tmp_path / "data.yaml"
    assert config.model_yaml == tmp_path / "model.yaml"
    assert config.vit5_checkpoint == tmp_path / "vit5.pth"
    assert config.output_dir == tmp_path / "runs"
    assert config.seeds == (0, 1, 2)
    assert config.imgsz == 640
    assert config.train["epochs"] == 100


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"imgsz": 512}, "640"),
        ({"seeds": [0, 1]}, "0, 1, 2"),
        ({"train": {"multi_scale": True, "rect": False}}, "multi_scale"),
        ({"train": {"multi_scale": False, "rect": True}}, "rect"),
        ({"validation": {"rect": True}}, "validation.*rect"),
        ({"vit5_checkpoint": "missing.pth"}, "vit5_checkpoint"),
    ],
)
def test_config_rejects_incompatible_main_experiment(
    tmp_path: Path, changes: dict, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_experiment_config(write_config(tmp_path, **changes))


def test_template_exposes_local_dataset_path() -> None:
    template = Path(__file__).resolve().parents[1] / "configs" / "experiment.example.yaml"
    raw = yaml.safe_load(template.read_text(encoding="utf-8"))

    assert raw["data"] == "../dataset/sample_v1_maivang/data.yaml"
    assert raw["vit5_checkpoint"] == "../checkpoints/vit5_small_patch16_224.pth"
    assert raw["seeds"] == [0, 1, 2]
    assert raw["baseline_ultralytics_version"] == "8.4.185"


def test_run_metadata_records_resolved_inputs_and_dataset_hash(tmp_path: Path) -> None:
    config = load_experiment_config(write_config(tmp_path))

    metadata = build_run_metadata(
        config,
        ultralytics_version="8.3.0",
        vit5_revision="abc123",
        effective_train_args={"imgsz": 640, "batch": 16},
    )

    assert metadata["dataset_yaml_sha256"] == (
        "adc0221ac71c8f3a90d1be5a63e8195cb2cca99b32f139014101bde030bd9f64"
    )
    assert metadata["resolved_config"]["data"] == str(tmp_path / "data.yaml")
    assert metadata["ultralytics_version"] == "8.3.0"
    assert metadata["vit5_revision"] == "abc123"
    assert metadata["vit5_checkpoint_sha256"] == (
        "4097889236a2af26c293033feb964c4cf118c0224e0d063fec0a89e9d0569ef2"
    )
    assert metadata["effective_train_args"] == {"imgsz": 640, "batch": 16}


def test_run_metadata_can_be_saved_as_json(tmp_path: Path) -> None:
    import json

    config = load_experiment_config(write_config(tmp_path))
    metadata = build_run_metadata(
        config,
        ultralytics_version="8.3.0",
        vit5_revision="abc123",
        effective_train_args={"imgsz": 640},
    )

    saved = write_run_metadata(tmp_path / "run", metadata)

    assert saved == tmp_path / "run" / "metadata.json"
    assert json.loads(saved.read_text(encoding="utf-8")) == metadata


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("train", "imgsz", 512),
        ("train", "seed", 7),
        ("train", "data", "other.yaml"),
        ("validation", "imgsz", 512),
        ("validation", "data", "other.yaml"),
    ],
)
def test_config_rejects_nested_core_setting_override(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    options = {"epochs": 100, "batch": 16, "multi_scale": False, "rect": False}
    validation = {"rect": False}
    target = options if section == "train" else validation
    target[key] = value

    with pytest.raises(ConfigError, match=key):
        load_experiment_config(write_config(tmp_path, train=options, validation=validation))


def test_config_rejects_invalid_dataset_contents(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(data_yaml.read_text(encoding="utf-8").replace("nc: 6", "nc: 5"), encoding="utf-8")

    with pytest.raises(ConfigError, match="nc=6"):
        load_experiment_config(config_path)
