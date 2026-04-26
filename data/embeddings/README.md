# G3 Corfu Butterflies — Image Embeddings

Two frozen feature extractors over the 165 images, shipped as a
derived convenience product so downstream studies (nearest-neighbour,
clustering, retrieval, weak supervision) can skip the GPU step.

| File | Model | Shape | Dtype | Notes |
|---|---|---|---|---|
| `clip_vitb32.npy` | OpenCLIP `ViT-B-32` (`laion2b_s34b_b79k`) | (165, 512) | float32 | L2-normalised; matches `g3.clip_baseline` |
| `vitb16_imagenet.npy` | torchvision `vit_b_16` (`IMAGENET1K_V1`) | (165, 768) | float32 | Raw pooled CLS; pre-fine-tune backbone of `g3.train` |
| `index.csv` | — | 165 × 2 | — | `filename,row` aligned with the two `.npy` files **and** with `data/metadata.csv` |
| `manifest.json` | — | — | — | Model identifiers, shapes, SHA-256 of each `.npy` |

Row order is identical across `clip_vitb32.npy`, `vitb16_imagenet.npy`,
`index.csv`, and `data/metadata.csv` — so multi-hot labels in
`metadata.csv["labels_multihot"]` align row-for-row with the embeddings.

Reproduce:

```bash
python scripts/extract_embeddings.py --device cpu
```

Image preprocessing for the ViT extractor: resize to 258, centre-crop
to 224, ImageNet mean/std. The CLIP extractor uses the OpenCLIP
default preprocess for `ViT-B-32`. Both runs are deterministic
given the shipped images and the pinned weight tags.
