"""Extract image embeddings for the G3 Corfu Butterflies benchmark.

Two frozen feature extractors are evaluated, each producing one
``(N_images, D)`` ``float32`` matrix aligned row-for-row with
``data/metadata.csv``:

1. **CLIP ViT-B/32** (open_clip, ``laion2b_s34b_b79k``, 512-d).
   L2-normalised. Matches the linear-probe baseline in
   ``g3.clip_baseline``.
2. **ViT-B/16** (torchvision, ``IMAGENET1K_V1``, 768-d pooled CLS
   token). Frozen, *not* fine-tuned: this is the pre-fine-tune
   representation of the same backbone used by ``g3.train``.
   Standard ImageNet normalisation, 224x224 centre-crop after
   resize-to-258.

The matrices are deterministic given the shipped images, the seed of
the model weights (fixed), and the eval-time transform. They are
shipped alongside the data as a derived convenience product so that
downstream studies (nearest-neighbour, clustering, retrieval, weak
supervision) can skip the GPU step.

Outputs (under ``--out-dir``, default ``data/embeddings``):

- ``clip_vitb32.npy``         (N, 512) float32, L2-normalised.
- ``vitb16_imagenet.npy``     (N, 768) float32, raw pooled features.
- ``index.csv``               filename, row, ordered to match the
                              two ``.npy`` matrices and ``metadata.csv``.
- ``manifest.json``           model identifiers, dimensions, SHA-256
                              of each ``.npy`` for integrity checks.

Author:
    Nikolaos Korfiatis, Ionian University. Contact: nkorf@ionio.gr

License:
    MIT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageList(Dataset):
    def __init__(self, image_dir: Path, files: list[str], transform) -> None:
        self.image_dir = image_dir
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_dir / self.files[idx]).convert("RGB")
        return self.transform(img)


@torch.no_grad()
def encode_clip(files: list[str], image_dir: Path, device: str, batch_size: int) -> tuple[np.ndarray, str]:
    import open_clip

    model_name = "ViT-B-32"
    pretrained_tag = "laion2b_s34b_b79k"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag)
    model.eval().to(device)

    loader = DataLoader(ImageList(image_dir, files, preprocess), batch_size=batch_size, num_workers=0)
    chunks: list[np.ndarray] = []
    for x in loader:
        x = x.to(device)
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        chunks.append(feats.float().cpu().numpy())
    return np.concatenate(chunks, axis=0), f"open_clip:{model_name}:{pretrained_tag}"


@torch.no_grad()
def encode_vit(files: list[str], image_dir: Path, device: str, batch_size: int) -> tuple[np.ndarray, str]:
    from torchvision.models import ViT_B_16_Weights, vit_b_16
    from torchvision.transforms import v2 as T

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.heads = torch.nn.Identity()
    model.eval().to(device)

    transform = T.Compose([
        T.ToImage(),
        T.Resize(258),
        T.CenterCrop(224),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    loader = DataLoader(ImageList(image_dir, files, transform), batch_size=batch_size, num_workers=0)
    chunks: list[np.ndarray] = []
    for x in loader:
        x = x.to(device)
        feats = model(x)
        chunks.append(feats.float().cpu().numpy())
    return np.concatenate(chunks, axis=0), "torchvision:vit_b_16:IMAGENET1K_V1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="data/embeddings")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--skip-vit", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(data_dir / "metadata.csv")
    files = meta["filename"].tolist()
    image_dir = data_dir / "images"
    missing = [f for f in files if not (image_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing: {missing[:3]}")

    pd.DataFrame({"filename": files, "row": range(len(files))}).to_csv(
        out_dir / "index.csv", index=False
    )

    manifest: dict = {"n_images": len(files), "models": {}}

    if not args.skip_clip:
        print(f"[CLIP] encoding {len(files)} images on {args.device} ...")
        X, ident = encode_clip(files, image_dir, args.device, args.batch_size)
        out_path = out_dir / "clip_vitb32.npy"
        np.save(out_path, X.astype(np.float32))
        manifest["models"]["clip_vitb32"] = {
            "identifier": ident, "shape": list(X.shape), "dtype": "float32",
            "l2_normalised": True, "sha256": sha256(out_path),
        }
        print(f"  -> {out_path} {X.shape}")

    if not args.skip_vit:
        print(f"[ViT-B/16] encoding {len(files)} images on {args.device} ...")
        X, ident = encode_vit(files, image_dir, args.device, args.batch_size)
        out_path = out_dir / "vitb16_imagenet.npy"
        np.save(out_path, X.astype(np.float32))
        manifest["models"]["vitb16_imagenet"] = {
            "identifier": ident, "shape": list(X.shape), "dtype": "float32",
            "l2_normalised": False, "sha256": sha256(out_path),
        }
        print(f"  -> {out_path} {X.shape}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
