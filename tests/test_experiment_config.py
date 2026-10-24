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
    for path in (data, model, checkpoint):
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
    assert raw["seeds"] == [0, 1, 2]


def test_run_metadata_records_resolved_inputs_and_dataset_hash(tmp_path: Path) -> None:
    config = load_experiment_config(write_config(tmp_path))

    metadata = build_run_metadata(
        config,
        ultralytics_version="8.3.0",
        vit5_revision="abc123",
        effective_train_args={"imgsz": 640, "batch": 16},
    )

    assert metadata["dataset_yaml_sha256"] == (
        "4097889236a2af26c293033feb964c4cf118c0224e0d063fec0a89e9d0569ef2"
    )
    assert metadata["resolved_config"]["data"] == str(tmp_path / "data.yaml")
    assert metadata["ultralytics_version"] == "8.3.0"
    assert metadata["vit5_revision"] == "abc123"
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
