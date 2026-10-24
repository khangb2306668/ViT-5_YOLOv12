from pathlib import Path

import pytest
import torch

from vit5_yolo12.checkpoint import CheckpointError, load_vit5_checkpoint
from vit5_yolo12.hybrid import ViT5HybridBackbone
from vit5_yolo12.vit5 import ViT5SpatialEncoder


def small_backbone() -> ViT5HybridBackbone:
    return ViT5HybridBackbone(vit5=ViT5SpatialEncoder(depth=1))


def checkpoint_state(backbone: ViT5HybridBackbone) -> dict[str, torch.Tensor]:
    state = {key: value.clone() for key, value in backbone.vit5.state_dict().items()}
    state["blocks.0.attn.qkv.weight"].fill_(0.125)
    state["head.weight"] = torch.zeros(1000, 384)
    state["head.bias"] = torch.zeros(1000)
    return state


def test_loader_accepts_official_model_envelope_and_skips_classifier(tmp_path: Path) -> None:
    backbone = small_backbone()
    path = tmp_path / "vit5.pth"
    torch.save({"model": checkpoint_state(backbone)}, path)

    report = load_vit5_checkpoint(backbone, path)

    assert report.source == "model"
    assert report.prefix == ""
    assert len(report.loaded) == len(backbone.vit5.state_dict())
    assert set(report.skipped) == {"head.weight", "head.bias"}
    assert report.unexpected == ()
    assert torch.all(backbone.vit5.blocks[0].attn.qkv.weight == 0.125)
    assert backbone.vit5.pos_embed.shape == (1, 196, 384)
    assert backbone.vit5._interpolate_pos_embed(40).shape == (1, 1600, 384)


def test_loader_accepts_prefixed_state_dict(tmp_path: Path) -> None:
    backbone = small_backbone()
    state = {f"module.{key}": value for key, value in checkpoint_state(backbone).items()}
    path = tmp_path / "wrapped.pth"
    torch.save({"state_dict": state}, path)

    report = load_vit5_checkpoint(backbone, path)

    assert report.source == "state_dict"
    assert report.prefix == "module."
    assert len(report.loaded) == len(backbone.vit5.state_dict())


def test_loader_rejects_shape_mismatch_without_partial_update(tmp_path: Path) -> None:
    backbone = small_backbone()
    original = backbone.vit5.blocks[0].attn.qkv.weight.detach().clone()
    state = checkpoint_state(backbone)
    state["blocks.0.attn.qkv.weight"] = torch.zeros(1, 1)
    path = tmp_path / "wrong-shape.pth"
    torch.save({"model": state}, path)

    with pytest.raises(CheckpointError, match="shape"):
        load_vit5_checkpoint(backbone, path)

    assert torch.equal(backbone.vit5.blocks[0].attn.qkv.weight, original)


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_loader_rejects_incomplete_or_extra_backbone_weights(tmp_path: Path, change: str) -> None:
    backbone = small_backbone()
    state = checkpoint_state(backbone)
    if change == "missing":
        del state["blocks.0.mlp.fc1.weight"]
    else:
        state["blocks.0.unknown.weight"] = torch.zeros(1)
    path = tmp_path / "invalid.pth"
    torch.save({"model": state}, path)

    with pytest.raises(CheckpointError, match=f"(?i){change}"):
        load_vit5_checkpoint(backbone, path)


def test_loader_prefers_model_unless_ema_is_requested(tmp_path: Path) -> None:
    backbone = small_backbone()
    model_state = checkpoint_state(backbone)
    ema_state = checkpoint_state(backbone)
    ema_state["blocks.0.attn.qkv.weight"].fill_(0.25)
    path = tmp_path / "both.pth"
    torch.save({"model": model_state, "model_ema": ema_state}, path)

    report = load_vit5_checkpoint(backbone, path)
    assert report.source == "model"
    assert torch.all(backbone.vit5.blocks[0].attn.qkv.weight == 0.125)

    report = load_vit5_checkpoint(backbone, path, prefer_ema=True)
    assert report.source == "model_ema"
    assert torch.all(backbone.vit5.blocks[0].attn.qkv.weight == 0.25)


def test_loader_reports_corrupt_checkpoint_as_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pth"
    path.write_bytes(b"not a torch checkpoint")

    with pytest.raises(CheckpointError, match="Cannot read"):
        load_vit5_checkpoint(small_backbone(), path)


@pytest.mark.filterwarnings("ignore:Validating sparse tensor invariants")
def test_loader_rejects_sparse_tensor_without_changing_any_weight(tmp_path: Path) -> None:
    backbone = small_backbone()
    original = backbone.vit5.cls_token.detach().clone()
    state = checkpoint_state(backbone)
    state["cls_token"].fill_(0.375)
    state["norm.weight"] = torch.sparse_coo_tensor(
        torch.tensor([[0]]), torch.tensor([1.0]), size=(384,), check_invariants=True
    )
    path = tmp_path / "sparse.pth"
    torch.save({"model": state}, path)

    with pytest.raises(CheckpointError, match="layout"):
        load_vit5_checkpoint(backbone, path)

    assert torch.equal(backbone.vit5.cls_token, original)


def test_official_small_checkpoint_loads_when_available() -> None:
    path = Path(__file__).resolve().parents[1] / "checkpoints" / "vit5_small_patch16_224.pth"
    if not path.exists():
        pytest.skip("Official ViT-5-Small checkpoint is not downloaded")
    backbone = ViT5HybridBackbone()

    report = load_vit5_checkpoint(backbone, path)

    assert report.source == "model"
    assert len(report.loaded) == 186
    assert set(report.skipped) == {"head.weight", "head.bias"}
    assert report.unexpected == ()
