# ViT-5 + YOLO12n playground

## Local environment (Windows PowerShell)

From the repository root:

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -e '.[test]'
.\.venv\Scripts\python.exe -m pytest -q
```

To activate the environment in a PowerShell session, run
`.\.venv\Scripts\Activate.ps1`. Calling `.venv\Scripts\python.exe` directly
also works without activation. The `.venv` directory is local and ignored by Git.

When preparing the training environment, install the pinned Ultralytics extra:

```powershell
uv pip install --python .venv\Scripts\python.exe -e '.[train,test]'
```

The training extra currently pins Ultralytics 8.4.185 based on the supplied
package-download log. Confirm the version installed in the YOLO12n baseline
Colab run before treating this as a version-matched comparison.

The ViT-5-Small checkpoint comes from the authors' [official Hugging Face
repository](https://huggingface.co/FengWang3211/ViT-5/blob/main/vit5_small_patch16_224.pth).
Place it at `checkpoints/vit5_small_patch16_224.pth`. The downloaded file used
for Task 3 has SHA-256
`b71918ba918ac5c59475cb7088d528f87045a0731a2bba8b24f49d92c997a890`.
The `checkpoints` folder is ignored by Git.

To construct the hybrid detector, register its backbone before the Ultralytics
YAML parser runs:

```python
from ultralytics import YOLO
from vit5_yolo12.ultralytics_integration import register_ultralytics_modules

register_ultralytics_modules()
model = YOLO("configs/yolo12-vit5n.yaml", task="detect")
```

This constructs the official YOLO12n neck and Detect modules with a newly
initialized six-class head. Loading the ViT-5 checkpoint into trainer-created
models is handled in the training step.

The dataset folders under `dataset/sample_v1_maivang` are placeholders. Add
`images` and `labels` under each of `train`, `valid`, and `test`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py dataset\sample_v1_maivang\data.yaml
```

For training, copy `configs/experiment.example.yaml` to the ignored
`configs/experiment.yaml`, edit its paths and options, then run:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\experiment.yaml
```

This runs seeds 0, 1, and 2 with the pinned Ultralytics trainer and the
official ViT-5 initialization. Each run saves its effective arguments,
dataset/checkpoint hashes, and source revision in `metadata.json` before
training starts. The example starts at batch 1 for the GPU pilot; measure
peak memory before increasing it, and use the same final recipe for the
YOLO12n control run. To resume a specific run, set `train.resume` in the local
experiment YAML to its `last.pt` path and pass `--seed` for that run. Resume
checks the checkpoint's effective seed, image size, rectangular mode, and
multiscale setting before training.

## Initialization B: ViT-5 plus pretrained YOLO12n detection components

Prepare the official **unfused, 80-class COCO** YOLO12n checkpoint once, before
training. With the training extra installed, this command downloads the official
Ultralytics asset to the ignored checkpoints directory (it also works on Colab):

```bash
python -c "from ultralytics.utils.downloads import attempt_download_asset; attempt_download_asset('checkpoints/yolo12n.pt')"
```

The checkpoint verified during implementation has SHA-256
`419ff3dca37d69bacc93a50fa0c186a1c6f9fe62fae0f108b0872829689e9ca6`.
The loader checks graph compatibility; metadata records the hash of the actual
file used. It never downloads a missing source during trainer reconstruction.

Copy `configs/experiment.pretrained-b.example.yaml` to
`configs/experiment.pretrained-b.yaml`, edit `data` to the same dataset used by
the old hybrid run, and check both checkpoint paths. Start with seed 1:

```bash
python scripts/train.py --config configs/experiment.pretrained-b.yaml --seed 1
```

The example reproduces the previous A100 recipe: batch 64, MuSGD at lr0 0.01
with momentum 0.9 and warmup_bias_lr 0.0, at most 100 epochs and patience 15.
Those explicit values match what `optimizer=auto` selected in that run.
Use a GPU that fits this batch for the direct comparison. The new output
directory keeps the previous run available.

`yolo12_checkpoint` enables initialization B: an explicit layer mapping copies
P3, the YOLO12n neck, and Detect's three box-regression branches. This transfers
336 state tensors (including BatchNorm buffers) and 1,205,856 parameters.
ViT-5 keeps the author's initialization; the P4/P5 adapters and the entire
six-class classifier initialize anew. DFL remains the fixed projector.
All transferred learned parameters remain trainable. Missing keys, wrong
shapes/scales/classes, or fused sources fail before changing destination state.
Legacy official positional-convolution biases in the unused source backbone
stages 6 and 8 are explicitly skipped and reported.

The run log prints `Initialization B: 336 tensors, 1,205,856 parameters ...`.
After setup, `metadata.json` includes `initialization.mode: vit5_yolo12_b`,
per-group source/target mappings, intentionally skipped keys, both checkpoint
hashes and actual optimizer groups. An early metadata record is still written
if setup is interrupted, but its initialization mode remains `pending` until
the model has been initialized.

Omit `yolo12_checkpoint` or set it to `null` to retain the original ViT-only
initialization. Do not pass the YOLO12n file through `train.pretrained`: its
layer indices differ from the hybrid's. When initialization B is configured,
`train.pretrained` and `validation.pretrained` must be omitted or `true`.
Full hybrid resume/fine-tune weights take priority and skip the base YOLO12n
transfer; metadata records `hybrid_weights` in that case. A completed run whose
optimizer has been stripped can only start a new fine-tuning run, not restore
the old optimizer state. Initialization B itself starts from the hybrid YAML
and the two official pretrained files, not a previous hybrid `best.pt`.
