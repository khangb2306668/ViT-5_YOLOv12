"""Three-scale hybrid backbone using the official YOLO12n P3 module types."""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules import C3k2, Conv

from .vit5 import ViT5SpatialEncoder


class ViT5HybridBackbone(nn.Module):
    """Return P3/8 from YOLO12n topology and P4/16, P5/32 from ViT-5."""

    def __init__(self, vit5: ViT5SpatialEncoder | None = None) -> None:
        super().__init__()
        # YOLO12n YAML layers 0–4 after n-scale width/depth expansion.
        self.p3 = nn.Sequential(
            Conv(3, 16, k=3, s=2),
            Conv(16, 32, k=3, s=2),
            C3k2(32, 64, n=1, c3k=False, e=0.25),
            Conv(64, 64, k=3, s=2),
            C3k2(64, 128, n=1, c3k=False, e=0.25),
        )
        self.vit5 = vit5 if vit5 is not None else ViT5SpatialEncoder()
        self.p4 = Conv(384, 128, k=1, s=1)
        self.p5 = Conv(128, 256, k=3, s=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        p3 = self.p3(x)
        p4 = self.p4(self.vit5(x))
        p5 = self.p5(p4)
        return [p3, p4, p5]
