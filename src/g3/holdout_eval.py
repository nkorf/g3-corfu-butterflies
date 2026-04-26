"""Locked-holdout evaluator for the G3 Corfu Butterflies benchmark.

Loads every per-fold ``best.pt`` checkpoint, scores each against the
28-image locked holdout set, and reports both per-fold metrics and
the logit-averaged ensemble. The final ensemble macro-F1 is the
primary number reported in Table 2 of the accompanying paper.

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
import torch
from torch.utils.data import DataLoader

from g3.data import CorfuButterfliesDataset, VocabInfo
from g3.evaluate import compute_metrics, predict
from g3.model import build_vit_multilabel
from g3.train import pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    vocab = VocabInfo.load(data_dir / "label_vocab.json")
    device = pick_device(args.device)

    ds = CorfuButterfliesDataset(data_dir, data_dir / "splits" / "holdout_test.csv", split=None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    fold_dirs = sorted(results_dir.glob("fold_*"))
    if not fold_dirs:
        raise SystemExit(f"No fold_* checkpoints found under {results_dir}")

    ensemble_probs = []
    per_fold = []
    for fold_dir in fold_dirs:
        ckpt = fold_dir / "best.pt"
        if not ckpt.exists():
            continue
        model = build_vit_multilabel(vocab.num_labels).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        y, p = predict(model, loader, device)
        per_fold.append({"fold": fold_dir.name, **compute_metrics(y, p)})
        ensemble_probs.append(p)
    y_true = y  # same for each fold

    ens = np.mean(ensemble_probs, axis=0)
    ensemble_metrics = compute_metrics(y_true, ens)

    out = {"per_fold": per_fold, "ensemble": ensemble_metrics}
    (results_dir / "holdout_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
