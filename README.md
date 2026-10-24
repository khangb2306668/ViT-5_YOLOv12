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

The dataset folders under `dataset/sample_v1_maivang` are placeholders. Add
`images` and `labels` under each of `train`, `valid`, and `test`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py dataset\sample_v1_maivang\data.yaml
```
