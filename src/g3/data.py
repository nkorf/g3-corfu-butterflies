"""Dataset and augmentation utilities for G3 Corfu Butterflies.

This module exposes the :class:`CorfuButterfliesDataset` used by the
benchmark training and evaluation loops, the canonical train / eval
torchvision transforms, and the :class:`VocabInfo` helper that reads
``label_vocab.json``. The code deliberately keeps the dependency
footprint small (torch, torchvision, Pillow, pandas, numpy) so that
it runs unchanged on laptops, Colab, and Kaggle GPU notebooks.

Author:
    Nikolaos Korfiatis, Ionian University. Contact: nkorf@ionio.gr

License:
    MIT.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.uint8, scale=True),
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandAugment(num_ops=2, magnitude=7),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@dataclass
class VocabInfo:
    labels: list[str]
    label_to_idx: dict[str, int]

    @classmethod
    def load(cls, path: Path | str) -> "VocabInfo":
        data = json.loads(Path(path).read_text())
        return cls(labels=data["labels"], label_to_idx=data["label_to_idx"])

    @property
    def num_labels(self) -> int:
        return len(self.labels)


class CorfuButterfliesDataset(Dataset):
    """Multi-label Corfu Butterflies dataset.

    Args:
        data_dir: directory containing ``images/`` and ``metadata.csv``.
        split_csv: CSV with columns ``filename, split`` (for train/val folds)
            or just ``filename`` (for the holdout set).
        split: "train", "val", or None (uses all rows from split_csv).
        transform: optional torchvision transform. Defaults chosen by ``split``.
    """

    def __init__(
        self,
        data_dir: Path | str,
        split_csv: Path | str,
        split: str | None = None,
        transform=None,
        img_size: int = 224,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        metadata = pd.read_csv(self.data_dir / "metadata.csv")
        metadata = metadata.set_index("filename")

        split_df = pd.read_csv(split_csv)
        if "split" in split_df.columns and split is not None:
            split_df = split_df[split_df["split"] == split]
        self.files = split_df["filename"].tolist()

        self.labels = np.stack([
            np.fromstring(metadata.loc[f, "labels_multihot"], sep=" ", dtype=np.int8)
            for f in self.files
        ])

        if transform is None:
            transform = train_transform(img_size) if split == "train" else eval_transform(img_size)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]
        img = Image.open(self.images_dir / fname).convert("RGB")
        x = self.transform(img)
        y = torch.from_numpy(self.labels[idx]).float()
        return x, y

    def pos_weights(self) -> torch.Tensor:
        """Per-label pos_weight for BCEWithLogitsLoss (clipped to avoid explosions)."""
        counts = self.labels.sum(axis=0).astype(np.float32)
        neg = len(self.labels) - counts
        pw = np.where(counts > 0, neg / np.maximum(counts, 1), 1.0)
        return torch.tensor(np.clip(pw, 0.5, 20.0), dtype=torch.float32)
