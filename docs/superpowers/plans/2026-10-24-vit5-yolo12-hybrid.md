# ViT-5–YOLO12n Hybrid Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a six-class YOLO12n detector whose principal backbone is pretrained ViT-5-Small, with the original YOLO12n P3 topology as a high-resolution CNN branch.

**Architecture:** A custom backbone consumes square RGB images and returns `[P3,P4,P5]` at strides `[8,16,32]` and channels `[128,128,256]`. P3 comes from YOLO12n layers 0–4 topology with fresh weights; P4 comes from normalized ViT-5 patch tokens; P5 downsamples P4. Ultralytics `Index` nodes feed the unchanged YOLO12n neck and six-class Detect head.

**Tech Stack:** Python, PyTorch, Ultralytics, timm-compatible ViT-5 layers, pytest. Pin the exact Ultralytics version used by the existing YOLO12n baseline before implementing version-sensitive integration.

**Spec:** `docs/superpowers/specs/2026-10-23-vit5-yolo12-hybrid-design.md`

## Global Constraints

- Six class names and order are exactly those in the spec; `nc=6`.
- Training and validation model input is `[B,3,640,640]`; disable multiscale training and rectangular validation. Ultralytics' internal 256 × 256 stride probe must also work.
- Initialize ViT-5-Small from the authors' ImageNet checkpoint; omit its classifier. Initialize P3, adapters, neck, and Detect anew.
- Retain the original YOLO12n neck topology and Detect loss/postprocessing for the installed baseline version. The baseline uses `yolo12n.pt`.
- Use the existing annotation files and split, including overlapping whole-leaf and disease-region boxes. Run seeds 0, 1, 2. Report the result as a system comparison because the pretrained sources differ.
- Do not vendor the ViT-5 source without retaining its Apache-2.0 license and attribution; record the exact upstream revision/checkpoint identifier.
- The current `.gitignore` ignores `docs/`; force-add this spec and plan in the first documentation commit so the chosen design travels with the implementation.
- Keep architecture, dataset, and experiment settings in separate YAML files. Save the resolved experiment configuration with every run.
- Treat `dataset/sample_v1_maivang/data.yaml` as the local layout example. Its `train`, `val`, and `test` entries resolve to `train/images`, `valid/images`, and `test/images` beneath the dataset root. The split folders are currently empty, so data-contract checks must report missing `images`/`labels` clearly until the user adds the real dataset.
- Pin the Ultralytics training extra to `8.4.185`, the version named in the supplied pip metadata-download log. Confirm the actual installed version in the baseline Colab environment before claiming a version-matched comparison.

## File Map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Installable package, runtime dependencies, pytest configuration |
| `configs/experiment.example.yaml` | Tracked, editable template for paths, seeds, image size, and training settings |
| `src/vit5_yolo12/config.py` | Load and validate experiment settings; resolve paths and reject incompatible image/validation options |
| `src/vit5_yolo12/vit5.py` | ViT-5-Small implementation compatible with upstream checkpoint keys; spatial patch-token output and position handling |
| `src/vit5_yolo12/hybrid.py` | YOLO12n P3 branch, P4/P5 adapters, three-map backbone interface |
| `src/vit5_yolo12/checkpoint.py` | Strict ViT checkpoint loading and 14×14 to 40×40 APE adaptation |
| `src/vit5_yolo12/ultralytics_integration.py` | Register custom module before YAML parsing; square-input trainer/validator hooks |
| `configs/yolo12-vit5n.yaml` | Custom backbone, `Index` outputs, unchanged YOLO12n neck and Detect |
| `scripts/train.py` | Reproducible three-seed training entry point and saved run metadata |
| `scripts/evaluate.py` | Aggregate metrics and inspect overlapping-label examples |
| `tests/` | Focused shape, loading, integration, and single-step training checks |

## Review Focus

1. **Square build probe:** model construction at 256 × 256 must interpolate APE and return strides `[8,16,32]`; Task 4 tests this.
2. **Non-square input:** wrapper must reject it clearly until rectangular 2D RoPE is implemented; Task 2 tests this.
3. **Checkpoint mismatch:** loading must fail on unexpected non-head keys/shapes instead of silently dropping most weights; Task 3 tests this.
4. **Overlapping labels:** validation must retain both boxes and predictions for different disease classes; Task 5 tests data handling and Task 6 inspects examples.
5. **Unexpected resized validation input:** the training and standalone validation paths must supply 640 × 640 tensors, despite Ultralytics' rectangular validation default; Task 5 tests this.

---

### Task 1: Reproducible package and baseline contract

**Files:** Create `pyproject.toml`, `configs/experiment.example.yaml`, `src/vit5_yolo12/__init__.py`, `src/vit5_yolo12/config.py`, `tests/test_dataset_contract.py`, `tests/test_experiment_config.py`, `scripts/check_dataset.py`; modify `.gitignore` to exclude only the user's filled `configs/experiment.yaml`.

**Interfaces:** `check_dataset(data_yaml: Path) -> dict` returns `nc`, ordered `names`, and resolved split paths after validation. It resolves relative image paths from the directory containing `data.yaml`, checks each split's `images` and sibling `labels` directory, and reports a clear error for the current empty sample folders. `load_experiment_config(path: Path) -> ExperimentConfig` validates `data`, `model_yaml`, `vit5_checkpoint`, `output_dir`, `seeds`, `imgsz`, and Ultralytics training options. The tracked template points to `dataset/sample_v1_maivang/data.yaml` as an editable example; the user copies it to `configs/experiment.yaml` and edits that local file. Later scripts consume the validated config. Record the baseline Ultralytics version, effective training arguments, dataset YAML hash, resolved config, and upstream model revision in run metadata.

- [ ] **Step 1: Write failing dataset-contract tests.** Assert exactly six ordered class names, three readable split paths, and rejection of a swapped class order or `nc != 6`.
- [ ] **Step 2: Write failing experiment-config tests.** Assert the template loads after supplying local paths; reject an image size other than 640, multiscale training, rectangular validation, missing paths, or seeds other than `[0,1,2]` in the main comparison.
- [ ] **Step 3: Run `python -m pytest tests/test_dataset_contract.py tests/test_experiment_config.py -q`.** Expect the missing implementations to fail.
- [ ] **Step 4: Implement package metadata, `check_dataset`, and `load_experiment_config`.** Pin the installed baseline Ultralytics version; make `configs/experiment.example.yaml` the documented template and keep the filled local copy out of Git.
- [ ] **Step 5: Run the task tests and `python -m pip check`.** Expect both to pass after dependencies are installed.
- [ ] **Step 6: Commit the package, config template, validation code, spec, and plan.** Use `git add -f` for the two ignored documentation files; leave unrelated untracked files untouched.

### Task 2: Spatial ViT-5-Small encoder

**Files:** Create `src/vit5_yolo12/vit5.py`, `tests/test_vit5_encoder.py`; retain upstream attribution and license if adapting source.

**Interfaces:** `ViT5SpatialEncoder.forward(x: torch.Tensor) -> torch.Tensor` returns `[B,384,S/16,S/16]` for supported square inputs divisible by 16. `ViT5SpatialEncoder` keeps checkpoint-compatible names for patch embed, blocks, norm, CLS, registers, and APE. The pretrained coordinate base for patch RoPE remains 14; register RoPE uses its separate convention.

- [ ] **Step 1: Write failing tests.** On 256 and 640 square inputs, assert 16×16 and 40×40 outputs; assert a clear `ValueError` for rectangular input; verify patch-token order excludes CLS and all four registers while they still enter attention.
- [ ] **Step 2: Run `python -m pytest tests/test_vit5_encoder.py -q`.** Expect failure before the encoder exists.
- [ ] **Step 3: Implement the encoder.** Adapt upstream ViT-5-Small (12 blocks, width 384, six heads, patch 16, four registers), dynamically interpolate APE for the current square patch grid, and create RoPE tensors on the input device rather than calling `.cuda()`.
- [ ] **Step 4: Run the encoder tests on CPU; on L4, add one CUDA forward check.** Expect correct output shapes and finite values.
- [ ] **Step 5: Commit the encoder and its tests.**

### Task 3: Hybrid three-scale backbone and checkpoint loader

**Files:** Create `src/vit5_yolo12/hybrid.py`, `src/vit5_yolo12/checkpoint.py`, `tests/test_hybrid_backbone.py`, `tests/test_checkpoint.py`.

**Interfaces:** `ViT5HybridBackbone.forward(x: torch.Tensor) -> list[torch.Tensor]` returns `[P3,P4,P5]`. `load_vit5_checkpoint(backbone: ViT5HybridBackbone, path: Path) -> LoadReport` loads only `backbone.vit5`, returning loaded/skipped/unexpected key lists. P3 uses YOLO12n layers 0–4 topology at scale `n`; P4 is `Conv(384,128,k=1,s=1)`; P5 is `Conv(128,256,k=3,s=2)`.

- [ ] **Step 1: Write failing backbone tests.** Assert 640-input shapes `[B,128,80,80]`, `[B,128,40,40]`, `[B,256,20,20]`; assert P4/P5 gradients reach ViT and P3 gradients reach the CNN branch.
- [ ] **Step 2: Write failing loader tests.** With an upstream-shaped fixture, assert classifier keys are intentionally skipped, 14×14 APE becomes 40×40 at 640, a sentinel block weight is loaded unchanged, and an unrelated shape mismatch raises an error.
- [ ] **Step 3: Run `python -m pytest tests/test_hybrid_backbone.py tests/test_checkpoint.py -q`.** Expect failure before implementation.
- [ ] **Step 4: Implement the backbone and strict loader.** Match the actual official checkpoint's `model`/`state_dict` envelope, prefix, and EMA choice after inspecting its metadata; record these in `LoadReport` and refuse a nearly empty load.
- [ ] **Step 5: Rerun both test files; then run a real checkpoint load when the file is available.** Expect all assertions to pass and no unaccounted missing keys.
- [ ] **Step 6: Commit the backbone, loader, and tests.**

### Task 4: Ultralytics YAML and neck/head integration

**Files:** Create `configs/yolo12-vit5n.yaml`, `src/vit5_yolo12/ultralytics_integration.py`, `tests/test_ultralytics_integration.py`.

**Interfaces:** `register_ultralytics_modules() -> None` makes `ViT5HybridBackbone` visible to the pinned Ultralytics YAML parser before constructing `YOLO`. The YAML uses one custom backbone node plus three `Index` nodes with declared channels `[128,128,256]`, then copies the YOLO12n neck and Detect rows with new `from` indices.

- [ ] **Step 1: Write failing parser tests.** After registration, `YOLO(config, task="detect")` builds; its 256-square stride probe succeeds with strides `[8,16,32]`; a 640-square forward reaches Detect with three scales and `nc=6`.
- [ ] **Step 2: Run `python -m pytest tests/test_ultralytics_integration.py -q`.** Expect failure before the config and registration exist.
- [ ] **Step 3: Add registration and YAML.** Recompute every `from` reference; preserve original YOLO12n neck module types, repeats, arguments, and Detect behavior. Do not apply a second width multiplier to the custom backbone's already-scaled output channels.
- [ ] **Step 4: Run the parser test and call `register_ultralytics_modules(); YOLO(config).info()`.** Expect successful construction, three correct strides, and nonzero FLOPs. Compare the instantiated neck/head module graph to the official YOLO12n graph after remapping indices.
- [ ] **Step 5: Commit integration files and tests.**

### Task 5: Training and square validation path

**Files:** Extend `src/vit5_yolo12/ultralytics_integration.py`; create `scripts/train.py`, `tests/test_training_smoke.py`.

**Interfaces:** `HybridDetectionTrainer.get_model(cfg, weights=None, verbose=True)` builds the custom model and loads ViT-5 weights into the newly constructed model, including trainer rebuilds. `HybridDetectionTrainer.build_dataset(img_path, mode="train", batch=None)` forces square validation batches. `train_one_seed(config: ExperimentConfig, seed: int) -> Path` launches a run and returns its best checkpoint path.

- [ ] **Step 1: Write failing smoke tests.** Verify a trainer rebuild retains a known ViT checkpoint weight; training and validation batches entering the backbone are square 640; a tiny dataset with two overlapping, different-class boxes retains both labels.
- [ ] **Step 2: Run `python -m pytest tests/test_training_smoke.py -q`.** Expect failure before the trainer and entry point exist.
- [ ] **Step 3: Implement trainer and `python scripts/train.py --config configs/experiment.yaml`.** Read the validated config, disable multiscale/rectangular batches, use `imgsz=640`, pass through the configured optimizer/schedule/augmentations, save the resolved config and checkpoint IDs, and seed all three runs independently. On a resumed hybrid run, load the full resume weights after any base ViT initialization so fine-tuned ViT weights are preserved.
- [ ] **Step 4: Run a single real-batch forward/backward and one short training epoch on L4.** Expect finite loss, nonzero gradients in all five component groups, successful square validation, and a reloadable checkpoint. If the pinned Ultralytics validator changes input shape independently, add a matching square validator hook and test it.
- [ ] **Step 5: Commit trainer, script, and tests.**

### Task 6: Evaluation and three-seed result aggregation

**Files:** Create `scripts/evaluate.py`, `tests/test_evaluation.py`, and a run-protocol README under `docs/`.

**Interfaces:** `aggregate_runs(run_dirs: list[Path]) -> dict` checks seeds `{0,1,2}`, identical class order/split hash/input size, and returns per-seed plus mean and standard deviation for mAP50, mAP50–95, per-class AP, recall, latency, parameters, and peak VRAM. The script accepts the baseline run directories and hybrid run directories as inputs.

- [ ] **Step 1: Write failing aggregation tests.** Assert correct mean/standard deviation from three fixture runs and rejection of missing seeds, changed class order, or changed split hash.
- [ ] **Step 2: Run `python -m pytest tests/test_evaluation.py -q`.** Expect failure before the aggregator exists.
- [ ] **Step 3: Implement aggregation and a short qualitative review protocol.** Save example detections on coinfected leaves; verify that same-leaf predictions of distinct disease classes are not silently discarded. Report whole-leaf versus disease-region AP only when box-type metadata exists.
- [ ] **Step 4: Run all tests with `python -m pytest -q`; then execute the three hybrid seeds and evaluate the held-out split with the same preprocessing as the YOLO12n control.** Expect a comparison table and saved per-seed artifacts; do not call a metric gain causal evidence for the backbone alone.
- [ ] **Step 5: Commit evaluation files and tests.**

## Self-review notes

- The spec's six classes, overlapping boxes, topology, position adaptation, checkpoint policy, square-input contract, three seeds, and system-level interpretation each map to a task above.
- The most version-sensitive points are YAML parser registration, model stride probing, trainer rebuilds, and validation resize. Their exact behavior must be tested against the pinned baseline Ultralytics version before full training.
- Dataset paths, author checkpoint path, and baseline run directories are execution inputs; none are assumed to live in this repository.
