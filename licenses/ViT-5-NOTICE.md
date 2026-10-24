# ViT-5 source attribution

The implementation in `src/vit5_yolo12/vit5.py` adapts the `models_vit5.py`
and `rope.py` files from [Feng Wang et al.'s ViT-5 repository](https://github.com/wangf3014/ViT-5),
revision `b7b0796062f4b51ae00523cb77c7117ae2dff536`.

The original files are licensed under Apache License 2.0; a copy is in
`ViT-5-LICENSE.txt` in this directory. The local adaptation exposes spatial
patch features, interpolates absolute patch positions at forward time,
constructs RoPE positions on the tensor's device, and omits the classifier.
