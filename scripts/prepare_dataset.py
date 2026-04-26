"""Prepare the G3 Corfu Butterflies benchmark data.

Tokenises the raw colour / pattern / extra-features strings into a
shared multi-label vocabulary, writes a cleaned ``metadata.csv`` and
``label_vocab.json``, draws a 15 % stratified holdout test set, and
builds five iteratively-stratified cross-validation folds over the
remainder. The procedure is deterministic (seed 42), so every user
produces bit-identical splits.

Author:
    Nikolaos Korfiatis, Ionian University. Contact: nkorf@ionio.gr

License:
    MIT.

Usage::

    python scripts/prepare_dataset.py \\
        --source-csv data/butterflies_source.csv \\
        --images-dir data/images \\
        --out-dir data
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_FOLDS = 5
HOLDOUT_FRAC = 0.15


def tokenize(cell: str) -> list[str]:
    if pd.isna(cell):
        return []
    parts = re.split(r"[,/]", str(cell))
    return [p.strip().lower() for p in parts if p.strip()]


def build_vocab(df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    vocab: set[str] = set()
    for col, prefix in (("color", "color"), ("pattern", "pattern"), ("extra_features", "extra")):
        for cell in df[col]:
            for tok in tokenize(cell):
                vocab.add(f"{prefix}:{tok}")
    ordered = sorted(vocab)
    return ordered, {label: i for i, label in enumerate(ordered)}


def multihot(row: pd.Series, label_to_idx: dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(label_to_idx), dtype=np.int8)
    for col, prefix in (("color", "color"), ("pattern", "pattern"), ("extra_features", "extra")):
        for tok in tokenize(row[col]):
            key = f"{prefix}:{tok}"
            if key in label_to_idx:
                vec[label_to_idx[key]] = 1
    return vec


def stratified_holdout(Y: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(Y)), Y))
    return train_idx, test_idx


def kfold_indices(Y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    kf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.zeros(len(Y)), Y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", default="data/butterflies_source.csv")
    ap.add_argument("--images-dir", default="data/images")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.source_csv)
    df["filename"] = df["image"].apply(lambda p: Path(p).name)
    missing = [f for f in df["filename"] if not (Path(args.images_dir) / f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images referenced in CSV are missing: {missing[:3]}...")

    labels, label_to_idx = build_vocab(df)
    Y = np.stack([multihot(row, label_to_idx) for _, row in df.iterrows()])
    df["labels_multihot"] = [" ".join(map(str, row)) for row in Y]

    metadata_path = out_dir / "metadata.csv"
    df[["filename", "color", "pattern", "extra_features", "labels_multihot"]].to_csv(metadata_path, index=False)

    vocab_path = out_dir / "label_vocab.json"
    vocab_path.write_text(json.dumps({"labels": labels, "label_to_idx": label_to_idx}, indent=2))

    train_idx, test_idx = stratified_holdout(Y, HOLDOUT_FRAC, SEED)
    holdout_df = df.iloc[test_idx][["filename"]].reset_index(drop=True)
    holdout_df.to_csv(splits_dir / "holdout_test.csv", index=False)

    trainval_df = df.iloc[train_idx].reset_index(drop=True)
    trainval_Y = Y[train_idx]
    for fold_i, (tr, va) in enumerate(kfold_indices(trainval_Y, N_FOLDS, SEED)):
        rows = []
        for i in tr:
            rows.append({"filename": trainval_df.iloc[i]["filename"], "split": "train"})
        for i in va:
            rows.append({"filename": trainval_df.iloc[i]["filename"], "split": "val"})
        pd.DataFrame(rows).to_csv(splits_dir / f"fold_{fold_i}.csv", index=False)

    print(f"Wrote {metadata_path} ({len(df)} rows, {len(labels)} labels)")
    print(f"Wrote {vocab_path}")
    print(f"Wrote {splits_dir}/holdout_test.csv ({len(test_idx)} samples)")
    print(f"Wrote {splits_dir}/fold_[0-{N_FOLDS-1}].csv from {len(train_idx)} trainval samples")
    print(f"Labels: {labels[:8]}{'...' if len(labels) > 8 else ''}")


if __name__ == "__main__":
    main()
