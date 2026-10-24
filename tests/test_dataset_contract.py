from pathlib import Path

import pytest
import yaml

from vit5_yolo12.config import ConfigError, check_dataset


NAMES = [
    "alternaria_leaf_spot",
    "colletotrichum_anthracnose",
    "glomerella_anthracnose",
    "healthy",
    "pestalotia_leaf_spot",
    "pink_disease",
]


def make_dataset(root: Path, *, names=NAMES, nc=6) -> Path:
    for split in ("train", "valid", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / "sample.jpg").write_bytes(b"image placeholder")
        (label_dir / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "nc": nc,
                "names": names,
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
            }
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_dataset_resolves_three_splits_relative_to_yaml(tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "sample_v1_maivang")

    result = check_dataset(data_yaml)

    assert result["nc"] == 6
    assert result["names"] == NAMES
    assert result["splits"] == {
        "train": data_yaml.parent / "train" / "images",
        "val": data_yaml.parent / "valid" / "images",
        "test": data_yaml.parent / "test" / "images",
    }


def test_dataset_rejects_swapped_class_order(tmp_path: Path) -> None:
    swapped = NAMES.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    data_yaml = make_dataset(tmp_path / "dataset", names=swapped)

    with pytest.raises(ConfigError, match="class order"):
        check_dataset(data_yaml)


def test_dataset_rejects_wrong_class_count(tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "dataset", nc=5)

    with pytest.raises(ConfigError, match="nc=6"):
        check_dataset(data_yaml)


def test_dataset_reports_missing_split_folders(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "nc": 6,
                "names": NAMES,
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Missing train images directory"):
        check_dataset(data_yaml)


def test_dataset_reports_missing_labels_folder(tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "dataset")
    labels = data_yaml.parent / "valid" / "labels"
    (labels / "sample.txt").unlink()
    labels.rmdir()

    with pytest.raises(ConfigError, match="Missing val labels directory"):
        check_dataset(data_yaml)
