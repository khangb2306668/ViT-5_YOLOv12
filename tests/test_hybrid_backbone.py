import torch
from ultralytics import YOLO

from vit5_yolo12.hybrid import ViT5HybridBackbone
from vit5_yolo12.vit5 import ViT5SpatialEncoder


def test_p3_reuses_official_yolo12n_module_graph() -> None:
    hybrid = ViT5HybridBackbone(vit5=ViT5SpatialEncoder(depth=1))
    reference = YOLO("yolo12n.yaml").model.model[:5]

    assert [type(module) for module in hybrid.p3] == [type(module) for module in reference]
    assert [sum(p.numel() for p in module.parameters()) for module in hybrid.p3] == [
        sum(p.numel() for p in module.parameters()) for module in reference
    ]
    assert [type(module).__name__ for module in hybrid.p3] == [
        "Conv", "Conv", "C3k2", "Conv", "C3k2"
    ]


def test_backbone_produces_three_feature_maps_at_640() -> None:
    model = ViT5HybridBackbone(vit5=ViT5SpatialEncoder(depth=1)).eval()

    with torch.inference_mode():
        features = model(torch.randn(1, 3, 640, 640))

    assert [tuple(feature.shape) for feature in features] == [
        (1, 128, 80, 80),
        (1, 128, 40, 40),
        (1, 256, 20, 20),
    ]
    assert all(torch.isfinite(feature).all() for feature in features)


def test_backbone_gradients_reach_cnn_vit_and_adapters() -> None:
    model = ViT5HybridBackbone(vit5=ViT5SpatialEncoder(depth=1)).train()
    p3, p4, p5 = model(torch.randn(1, 3, 256, 256))

    (p3.square().mean() + p4.square().mean() + p5.square().mean()).backward()

    parameters = (
        model.p3[0].conv.weight,
        model.vit5.patch_embed.proj.weight,
        model.p4.conv.weight,
        model.p5.conv.weight,
    )
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in parameters)
