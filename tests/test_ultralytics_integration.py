"""YOLO12n neck/Detect integration with the custom three-scale backbone."""

from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Index

from vit5_yolo12.ultralytics_integration import register_ultralytics_modules


MODEL_YAML = Path(__file__).resolve().parents[1] / "configs" / "yolo12-vit5n.yaml"


def build_hybrid() -> YOLO:
    register_ultralytics_modules()
    return YOLO(str(MODEL_YAML), task="detect")


def test_parser_builds_custom_backbone_and_three_detection_strides() -> None:
    model = build_hybrid()
    layers = model.model.model

    assert len(layers) == 17
    assert [type(layer) for layer in layers[1:4]] == [Index, Index, Index]
    assert isinstance(layers[-1], Detect)
    assert layers[-1].nc == 6
    assert layers[-1].nl == 3
    assert layers[-1].stride.tolist() == [8.0, 16.0, 32.0]
    assert model.model.stride.tolist() == [8.0, 16.0, 32.0]


def test_full_model_reaches_detect_at_640() -> None:
    model = build_hybrid().model.eval()

    with torch.inference_mode():
        prediction, raw = model(torch.randn(1, 3, 640, 640))

    assert prediction.shape == (1, 10, 8400)
    assert [tuple(feature.shape[1:]) for feature in raw["feats"]] == [
        (64, 80, 80),
        (128, 40, 40),
        (256, 20, 20),
    ]
    assert torch.isfinite(prediction).all()


def test_neck_graph_matches_official_yolo12n_after_index_remapping() -> None:
    hybrid = build_hybrid().model.model
    original = YOLO("yolo12n.yaml").model.model
    backbone_refs = {4: 1, 6: 2, 8: 3}

    def remap(source: int) -> int:
        if source == -1:
            return -1
        return source - 5 if source >= 9 else backbone_refs[source]

    for old_index, new_index in zip(range(9, 22), range(4, 17)):
        original_layer = original[old_index]
        hybrid_layer = hybrid[new_index]
        assert type(hybrid_layer) is type(original_layer)
        old_from = original_layer.f
        expected_from = [remap(index) for index in old_from] if isinstance(old_from, list) else remap(old_from)
        assert hybrid_layer.f == expected_from
        if old_index != 21:  # Detect has six classes instead of eighty.
            assert sum(p.numel() for p in hybrid_layer.parameters()) == sum(
                p.numel() for p in original_layer.parameters()
            )


def test_model_info_reports_nonzero_compute() -> None:
    from ultralytics.utils.torch_utils import get_flops

    model = build_hybrid().model

    assert get_flops(model) > 0
