"""Behavior and checkpoint-shape checks for the spatial ViT-5-Small encoder."""

import pytest
import torch
from torch import nn

from vit5_yolo12.vit5 import ViT5SpatialEncoder


def test_default_model_matches_vit5_small_checkpoint_shapes() -> None:
    model = ViT5SpatialEncoder()
    state = model.state_dict()

    assert len(model.blocks) == 12
    assert model.blocks[0].attn.num_heads == 6
    assert state["patch_embed.proj.weight"].shape == (384, 3, 16, 16)
    assert state["cls_token"].shape == (1, 1, 384)
    assert state["reg_token"].shape == (1, 4, 384)
    assert state["pos_embed"].shape == (1, 196, 384)
    assert state["blocks.0.attn.q_norm.weight"].shape == (64,)
    assert state["blocks.0.attn.rope.freqs"].shape == (16,)
    assert state["blocks.0.attn.rope_reg.freqs"].shape == (16,)
    assert state["blocks.0.mlp.fc1.weight"].shape == (1536, 384)
    assert state["norm.weight"].shape == (384,)
    assert not any(key.startswith("head.") for key in state)


@pytest.mark.parametrize("size,grid", [(256, 16), (640, 40)])
def test_forward_returns_finite_spatial_feature_map(size: int, grid: int) -> None:
    model = ViT5SpatialEncoder(depth=1).eval()

    with torch.inference_mode():
        feature = model(torch.randn(1, 3, size, size))

    assert feature.shape == (1, 384, grid, grid)
    assert torch.isfinite(feature).all()


@pytest.mark.parametrize("height,width", [(256, 320), (250, 250)])
def test_forward_rejects_unsupported_image_shapes(height: int, width: int) -> None:
    model = ViT5SpatialEncoder(depth=1)

    with pytest.raises(ValueError, match="square.*divisible by 16"):
        model(torch.zeros(1, 3, height, width))


def test_sequence_order_keeps_cls_and_registers_out_of_spatial_map() -> None:
    class CaptureBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tokens = None

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            self.tokens = tokens.detach().clone()
            return tokens

    model = ViT5SpatialEncoder(depth=0)
    capture = CaptureBlock()
    model.blocks = nn.ModuleList([capture])
    model.norm = nn.Identity()
    with torch.no_grad():
        model.patch_embed.proj.weight.zero_()
        model.patch_embed.proj.bias.zero_()
        model.pos_embed.fill_(3)
        model.cls_token.fill_(1)
        model.reg_token.fill_(2)

    feature = model(torch.zeros(1, 3, 32, 32))

    assert capture.tokens.shape == (1, 9, 384)
    assert torch.all(capture.tokens[:, 0] == 1)
    assert torch.all(capture.tokens[:, 1:5] == 3)
    assert torch.all(capture.tokens[:, 5:] == 2)
    assert feature.shape == (1, 384, 2, 2)
    assert torch.all(feature == 3)


def test_cls_and_registers_influence_patch_features_through_attention() -> None:
    torch.manual_seed(4)
    model = ViT5SpatialEncoder(depth=1)

    feature = model(torch.randn(1, 3, 32, 32))
    feature[:, :1].sum().backward()

    assert model.cls_token.grad is not None
    assert model.cls_token.grad.abs().sum() > 0
    assert model.reg_token.grad is not None
    assert model.reg_token.grad.abs().sum() > 0


def test_rope_uses_pretraining_coordinate_bases() -> None:
    model = ViT5SpatialEncoder(depth=1)
    attention = model.blocks[0].attn
    assert attention.rope.pt_seq_len == 14
    assert attention.rope_reg.pt_seq_len == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable locally")
def test_cuda_forward_uses_input_device_for_position_encoding() -> None:
    model = ViT5SpatialEncoder(depth=1).cuda().eval()

    with torch.inference_mode():
        feature = model(torch.randn(1, 3, 256, 256, device="cuda"))

    assert feature.shape == (1, 384, 16, 16)
    assert feature.is_cuda
    assert torch.isfinite(feature).all()
