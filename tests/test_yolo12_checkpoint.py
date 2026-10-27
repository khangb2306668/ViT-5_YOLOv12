"""Initialization B copies an explicit graph mapping, never a global key intersection."""

import copy
from pathlib import Path

import pytest
import torch
from ultralytics import YOLO

from vit5_yolo12.ultralytics_integration import register_ultralytics_modules


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def models():
    register_ultralytics_modules()
    source = YOLO("yolo12n.yaml").model
    target = YOLO(str(ROOT / "configs/yolo12-vit5n.yaml"), task="detect").model
    return source, target


def save_source(tmp_path, source):
    path = tmp_path / "source.pt"
    torch.save({"model": source, "train_args": {"task": "detect"}}, path)
    return path


def test_transfer_maps_p3_neck_box_and_preserves_every_other_tensor(tmp_path, models):
    from vit5_yolo12.yolo12_checkpoint import load_yolo12_checkpoint

    source, target = (copy.deepcopy(model) for model in models)
    pairs = [(source.model[i], target.model[0].p3[i]) for i in range(5)]
    pairs += [(source.model[i], target.model[i - 5]) for i in range(9, 21)]
    pairs += [(source.model[21].cv2, target.model[16].cv2)]
    for i, (module, _) in enumerate(pairs):
        with torch.no_grad():
            for tensor in module.state_dict().values():
                tensor.fill_(i + 1)
    original = {key: value.clone() for key, value in target.state_dict().items()}
    rng = torch.random.get_rng_state().clone()

    report = load_yolo12_checkpoint(target, save_source(tmp_path, source))

    assert torch.equal(torch.random.get_rng_state(), rng)
    assert [group.name for group in report.groups] == ["p3", "neck", "box"]
    assert sum(len(group.mappings) for group in report.groups) == 336
    assert sum(group.parameters for group in report.groups) == 1_205_856
    mapped = set()
    source_state, target_state = source.state_dict(), target.state_dict()
    for group in report.groups:
        for source_key, target_key in group.mappings:
            mapped.add(target_key)
            assert torch.equal(target_state[target_key], source_state[source_key]), target_key
    for key in set(original) - mapped:
        assert torch.equal(target_state[key], original[key]), key
    assert set(report.untouched_target) == set(original) - mapped
    assert "model.21.cv3.0.2.bias" in report.skipped_source
    assert all(param.requires_grad for module in (target.model[0].p3, target.model[16].cv2)
               for param in module.parameters())


@pytest.mark.parametrize("damage", ["shape", "graph", "classes", "fused", "nonfinite", "shortcut"])
def test_invalid_source_is_rejected_before_any_target_update(tmp_path, models, damage):
    from vit5_yolo12.yolo12_checkpoint import Yolo12CheckpointError, load_yolo12_checkpoint

    source, target = (copy.deepcopy(model) for model in models)
    if damage == "shape":
        source.model[21].cv2[2][-1].weight = torch.nn.Parameter(torch.zeros(1, 1, 1, 1))
    elif damage == "graph":
        source.model[11].f = [-1, 5]
    elif damage == "classes":
        source.model[21].nc = 6
    elif damage == "fused":
        source.fuse(verbose=False)
    elif damage == "shortcut":
        source.model[2].m[0].add = not source.model[2].m[0].add
    else:
        with torch.no_grad():
            source.model[21].cv2[2][-1].weight.fill_(float("nan"))
    original = {key: value.clone() for key, value in target.state_dict().items()}

    with pytest.raises(Yolo12CheckpointError):
        load_yolo12_checkpoint(target, save_source(tmp_path, source))

    assert all(torch.equal(value, original[key]) for key, value in target.state_dict().items())


def test_missing_and_corrupt_local_checkpoint_do_not_trigger_download(tmp_path, models):
    from vit5_yolo12.yolo12_checkpoint import Yolo12CheckpointError, load_yolo12_checkpoint

    for path in (tmp_path / "yolo12n.pt", tmp_path / "corrupt.pt"):
        if path.name == "corrupt.pt":
            path.write_bytes(b"corrupt checkpoint")
        with pytest.raises(Yolo12CheckpointError):
            load_yolo12_checkpoint(models[1], path)


def test_legacy_official_positional_conv_biases_are_skipped_only_in_unused_backbone(tmp_path, models):
    from vit5_yolo12.yolo12_checkpoint import load_yolo12_checkpoint

    source, target = (copy.deepcopy(model) for model in models)
    added = set()
    for index in (6, 8):
        for name, module in source.model[index].named_modules():
            if name.endswith("attn.pe.conv"):
                module.bias = torch.nn.Parameter(torch.zeros(module.out_channels))
                added.add(f"model.{index}.{name}.bias")
    report = load_yolo12_checkpoint(target, save_source(tmp_path, source))
    assert len(added) == 8
    assert added <= set(report.skipped_source)


@pytest.mark.parametrize("damage", ["reference", "concat"])
def test_wrong_target_graph_is_rejected_without_mutation(tmp_path, models, damage):
    from vit5_yolo12.yolo12_checkpoint import Yolo12CheckpointError, load_yolo12_checkpoint

    source, target = (copy.deepcopy(model) for model in models)
    if damage == "reference":
        target.model[6].f = [-1, 1]
    else:
        target.model[5].d = 0
    before = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(Yolo12CheckpointError):
        load_yolo12_checkpoint(target, save_source(tmp_path, source))
    assert all(torch.equal(value, before[key]) for key, value in target.state_dict().items())


def test_official_checkpoint_loads_when_available(models):
    from vit5_yolo12.yolo12_checkpoint import load_yolo12_checkpoint

    path = ROOT / "checkpoints/yolo12n.pt"
    if not path.is_file():
        pytest.skip("Official YOLO12n checkpoint is not downloaded")
    target = copy.deepcopy(models[1])
    report = load_yolo12_checkpoint(target, path)
    assert sum(group.parameters for group in report.groups) == 1_205_856
    target.eval()
    with torch.inference_mode():
        prediction, _ = target(torch.zeros(1, 3, 640, 640))
    assert prediction.shape == (1, 10, 8400)
    assert torch.isfinite(prediction).all()
