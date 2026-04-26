"""CLIP-ViT-B/32 linear-probe baseline for the G3 Corfu Butterflies benchmark.

Provides a cheap, well-known CV reference point against which
fine-tuning methods can be compared. The pipeline is:

1. Encode every image with a frozen open_clip ViT-B/32 (laion2b_s34b_b79k)
   visual tower (512-d features).
2. Standardise the features and fit one-vs-rest logistic regression
   heads on the training folds of the shipped splits.
3. Score on the validation fold / locked holdout using the same
   metrics as the ViT-B/16 baseline.

The intent is not to compete with fine-tuning but to sanity-check
the benchmark: a frozen-feature linear probe should be a strictly
weaker baseline than a fine-tuned transformer on this scale of
data.

Author:
    Nikolaos Korfiatis, Ionian University. Contact: nkorf@ionio.gr

License:
    MIT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from g3.data import VocabInfo
from g3.evaluate import compute_metrics


class JustImages(Dataset):
    def __init__(self, data_dir: Path, filenames: list[str], transform) -> None:
        self.data_dir = data_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.data_dir / "images" / self.filenames[idx]).convert("RGB")
        return self.transform(img)


@torch.no_grad()
def encode(files: list[str], data_dir: Path, model, preprocess, device: str, batch_size: int = 32) -> np.ndarray:
    ds = JustImages(data_dir, files, preprocess)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0)
    chunks = []
    for x in loader:
        x = x.to(device)
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        chunks.append(feats.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def fit_predict(Xtr, Ytr, Xte, C: float = 1.0) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    P = np.zeros((Xte_s.shape[0], Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        y = Ytr[:, j]
        if y.sum() == 0 or y.sum() == len(y):
            P[:, j] = y.mean()
            continue
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr_s, y)
        P[:, j] = clf.predict_proba(Xte_s)[:, 1]
    return P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/clip_linear_probe.json")
    args = ap.parse_args()

    import open_clip

    data_dir = Path(args.data_dir)
    vocab = VocabInfo.load(data_dir / "label_vocab.json")
    meta = pd.read_csv(data_dir / "metadata.csv").set_index("filename")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval().to(args.device)

    holdout = pd.read_csv(data_dir / "splits" / "holdout_test.csv")["filename"].tolist()
    Yho = np.stack([np.fromstring(meta.loc[f, "labels_multihot"], sep=" ", dtype=np.int8) for f in holdout])
    Xho = encode(holdout, data_dir, model, preprocess, args.device)

    fold_results = []
    for fold_i in range(5):
        split = pd.read_csv(data_dir / "splits" / f"fold_{fold_i}.csv")
        tr = split[split["split"] == "train"]["filename"].tolist()
        va = split[split["split"] == "val"]["filename"].tolist()
        Ytr = np.stack([np.fromstring(meta.loc[f, "labels_multihot"], sep=" ", dtype=np.int8) for f in tr])
        Yva = np.stack([np.fromstring(meta.loc[f, "labels_multihot"], sep=" ", dtype=np.int8) for f in va])
        Xtr = encode(tr, data_dir, model, preprocess, args.device)
        Xva = encode(va, data_dir, model, preprocess, args.device)
        Pva = fit_predict(Xtr, Ytr, Xva)
        m = compute_metrics(Yva, Pva)
        m["fold"] = fold_i
        fold_results.append(m)
        print(f"fold {fold_i}: macro_f1={m['macro_f1']:.3f}")

    # Combine all training folds for the holdout probe.
    train_files = pd.read_csv(data_dir / "splits" / "fold_0.csv")
    all_trainval = train_files["filename"].tolist()
    Y_all = np.stack([np.fromstring(meta.loc[f, "labels_multihot"], sep=" ", dtype=np.int8) for f in all_trainval])
    X_all = encode(all_trainval, data_dir, model, preprocess, args.device)
    Pho = fit_predict(X_all, Y_all, Xho)
    holdout_metrics = compute_metrics(Yho, Pho)
    print(f"holdout: macro_f1={holdout_metrics['macro_f1']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cv": fold_results, "holdout": holdout_metrics}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
