# ViT-5 backbone with a high-resolution P3 branch for YOLO12n

## Intent

Test whether ViT-5 features improve detection of six leaf-condition classes on the existing dataset. YOLO12n is the primary comparator because it has the best reported mean result across seeds 0, 1, and 2 among the evaluated YOLO models. The detector must retain evidence from small lesion patterns while localizing mostly whole leaves. This document specifies the architecture and comparison, not an implementation plan.

## Dataset and evaluation contract

- Class order is fixed: `alternaria_leaf_spot`, `colletotrichum_anthracnose`, `glomerella_anthracnose`, `healthy`, `pestalotia_leaf_spot`, `pink_disease`; `nc=6`.
- The local dataset layout is `dataset/sample_v1_maivang/{train,valid,test}/{images,labels}` with `data.yaml` at the dataset root. The current split folders are placeholders; before training, populate each split with the real images and matching YOLO label files. Resolve `train: train/images`, `val: valid/images`, and `test: test/images` relative to that root. The experiment config points to this `data.yaml` and can be changed when the real dataset is placed elsewhere.
- About 95% of boxes enclose whole leaves. Some coinfected leaves also have an overlapping box around a disease region, typically about 20–30% of the leaf. Both boxes and their class labels remain in the training and evaluation data. The model is not asked to detect every tiny lesion as an individual object.
- Preserve the existing train/validation/test split and annotation files. Use the same seeds 0, 1, 2 and 640-pixel square model input as the YOLO12n baseline. Record mAP50, mAP50–95, per-class AP, recall, model size, latency, and peak GPU memory; report mean and variation across seeds.
- Separate AP for whole-leaf versus region boxes only if annotation metadata identifies the box type. Do not infer the box type from box area alone. Inspect overlapping predictions, including whether postprocessing removes a legitimate second disease label.

## Approaches considered

1. **Selected: ViT-5 with a small CNN P3 branch.** ViT-5 produces P4, its adapted output produces P5, and a shallow CNN produces high-resolution P3. This keeps ViT-5 as the main backbone while preserving local lesion detail.
2. **ViT-5 only.** Upsample ViT-5's stride-16 map to P3 and downsample it to P5. This isolates ViT-5 most clearly, but upsampling cannot recover spatial detail discarded by patchification. Keep as a useful ablation.
3. **Full YOLO12 backbone plus ViT-5 fusion.** This retains strong CNN features but increases cost and makes the contribution of ViT-5 harder to isolate. Do not use as the main model.

## Selected architecture

The input is one square RGB image, `[B,3,640,640]`. Two branches see the same image:

| Output | Source and transformation | Required tensor shape |
| --- | --- | --- |
| P3/8 | Reuse the **topology** of YOLO12n backbone layers 0–4: `Conv(64,3,2) → Conv(128,3,2) → C3k2(256,False,0.25) → Conv(256,3,2) → C3k2(512,False,0.25)`, with YOLO12n width/depth scaling applied | `[B,128,80,80]` |
| P4/16 | ViT-5-Small patch tokens, reshaped to a 2D grid, followed by 1 × 1 `Conv-BN-SiLU`, `384→128` channels | `[B,128,40,40]` |
| P5/32 | 3 × 3, stride-2 `Conv-BN-SiLU` from ViT-derived P4, `128→256` channels | `[B,256,20,20]` |

ViT-5-Small uses patch size 16 and embedding width 384. With a 640-pixel input it creates 40 × 40 = 1,600 spatial tokens. CLS and four register tokens remain in the transformer sequence so they can participate in attention; they are excluded when constructing the spatial map. The returned map is `[B,384,40,40]` before channel projection. The author's classification `forward_features()` returns only the CLS token, so the detection wrapper must expose normalized patch tokens instead.

The P3 CNN branch stops at the original YOLO12n P3 topology. In the nano scale its stage outputs are 16, 32, 64, 64, and 128 channels. There are two stride-2 convolutions before the P3 transition and a third stride-2 convolution that creates P3; the two C3k2 blocks process features at P2 and P3. This branch does not produce P4 or P5, so ViT-5 remains the source of both coarser scales. Its topology matches YOLO12n, but its weights are freshly initialized as specified below. The YOLO12n neck and Detect architecture are unchanged: the original references to backbone layers 4, 6, and 8 are replaced by the new P3, P4, and P5 outputs, respectively. The neck still produces three scales for one Detect module. Recompute layer indices in the generated model configuration rather than copying the original indices.

## Pretrained weights and position handling

- Initialize ViT-5-Small from the authors' ImageNet checkpoint. Omit its 1,000-class classifier. Initialize the P3 branch, P4/P5 adapters, YOLO12n neck, and six-class Detect head anew. This is the first, interpretable configuration. The YOLO12n baseline was initialized from `yolo12n.pt`; document the difference in pretraining sources when comparing results.
- The checkpoint's learned absolute positional embedding has a 14 × 14 patch grid for 224 × 224 input. Keep that 14 × 14 tensor as the checkpoint-compatible parameter, then interpolate it to the input's patch grid at forward time (40 × 40 for 640 input, 16 × 16 for the internal 256 probe). CLS and register tokens are separate parameters and are not mixed into this interpolation.
- 2D RoPE is generated from positions during the forward pass; it is not interpolated as checkpoint weights. Preserve the pretrained coordinate convention when evaluating the larger 40 × 40 grid, and validate token ordering and register-token RoPE separately. The author's implementation assumes a square patch grid and uses a CUDA-specific device call; the wrapper must make device handling explicit. Initially support square 640 × 640 input only.
- Require a final `[B,3,640,640]` tensor at the hybrid backbone entry for both training and validation. Keep compatible image/content augmentations, but disable multi-scale training and rectangular batches unless dynamic position handling is implemented and verified. The Ultralytics detection trainer may enable rectangular validation batches independently of `rect=False`; the hybrid training and standalone validation paths must explicitly produce square 640 × 640 tensors. Apply the same input-size rules to the YOLO12n control run. The wrapper may also accept square 256 × 256 input for Ultralytics' model-construction stride probe; interpolate APE for that grid as well. This internal probe does not expand the supported training input contract.
- A load report must list loaded, intentionally skipped, and unexpected keys. Shape mismatches other than the documented classifier and positional-embedding adaptation are errors.

## Validation gates

1. For a synthetic `[1,3,640,640]` input, assert P3/P4/P5 shapes, successful neck concatenations, and three Detect scales.
2. Confirm that the ViT checkpoint loads with the expected key report and that positional interpolation produces 1,600 patch positions.
3. Run forward and backward on one real batch; confirm finite loss and nonzero gradients in ViT-5, P3 branch, adapters, neck, and Detect.
4. Train and evaluate seeds 0, 1, 2 with the same data split and documented recipe as the baseline. Compare both overall and per-class metrics; inspect co-infection examples with overlapping ground-truth boxes.

## Constraints and exclusions

- Keep model topology in a model YAML, dataset paths/class names in the existing dataset YAML, and editable training settings in a separate experiment YAML. Training reads the experiment YAML and saves its resolved values with each run; local dataset/checkpoint paths must not be hardcoded in Python.
- The main experiment changes the backbone while retaining the YOLO12n neck and Detect structure. Training schedule and augmentation differences must be recorded, and any changed recipe should also be applied to a YOLO12n control run if the comparison is claimed to isolate architecture.
- The primary result is a **system-level comparison** of the specified pretrained ViT-5 hybrid against a `yolo12n.pt`-initialized baseline. Because the pretraining sources and the amount of newly initialized detector code differ, do not claim that a metric difference is caused solely by the backbone architecture. A causal backbone-only claim would require an additional matched-init or controlled pretraining experiment.
- GPU capacity and wall-clock time are not design constraints at this stage. Measure both during the implementation pilot rather than assuming a viable batch size.
- No new tiny-lesion box annotation scheme, segmentation head, or change to the six-class ontology is included.
- Optional later ablations: ViT-only P3 via upsampling; partial transfer of compatible YOLO12n neck weights. These are not prerequisites for the main model.

## References

- ViT-5 paper supplied by the user: `../docs/ViT-5.pdf`.
- [ViT-5 implementation](https://github.com/wangf3014/ViT-5/blob/main/models_vit5.py) and [2D RoPE implementation](https://github.com/wangf3014/ViT-5/blob/main/rope.py).
- [Ultralytics YOLO12 model configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/12/yolo12.yaml).
