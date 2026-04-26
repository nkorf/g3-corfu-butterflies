"""Reference training loop for the G3 Corfu Butterflies benchmark.

Given a prepared dataset directory (see
``scripts/prepare_dataset.py``) this module trains the ViT-B/16
multi-label classifier with 5-fold stratified cross-validation, saves
the best-epoch checkpoint of each fold, and writes a JSON summary
that later drives ``scripts/export_results.py``. The loop is written
so that the same code runs on CPU, CUDA, and Apple Silicon MPS (see
the *Future Work* section of the accompanying paper for the current
caveats on the MPS backend).

Author:
    Nikolaos Korfiatis, Ionian University. Contact: nkorf@ionio.gr

License:
    MIT.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from g3.data import CorfuButterfliesDataset, VocabInfo
from g3.evaluate import compute_metrics, predict, save_metrics
from g3.model import build_vit_multilabel, param_groups

N_FOLDS = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_one_fold(
    fold: int,
    data_dir: Path,
    results_dir: Path,
    vocab: VocabInfo,
    epochs: int,
    batch_size: int,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    device: str,
    amp: bool,
    seed: int,
) -> dict:
    set_seed(seed + fold)
    split_csv = data_dir / "splits" / f"fold_{fold}.csv"
    train_ds = CorfuButterfliesDataset(data_dir, split_csv, split="train")
    val_ds = CorfuButterfliesDataset(data_dir, split_csv, split="val")

    dl_workers = 2 if device != "mps" else 0
    persistent = dl_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=dl_workers, persistent_workers=persistent, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=dl_workers, persistent_workers=persistent,
    )

    model = build_vit_multilabel(vocab.num_labels).to(device)
    opt = AdamW(param_groups(model, backbone_lr, head_lr), weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    pos_weight = train_ds.pos_weights().to(device=device, dtype=torch.float32)

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = pos_weight * targets * torch.nn.functional.softplus(-logits)
        neg = (1.0 - targets) * torch.nn.functional.softplus(logits)
        out = (pos + neg).mean()
        if not torch.isfinite(out):
            # Fall back to CPU: some combinations of tensor sizes
            # occasionally trip the MPS reduction kernel.
            out = (pos.cpu() + neg.cpu()).mean().to(logits.device)
        return out

    use_amp = amp and device == "cuda"
    scaler: torch.amp.GradScaler | None = (
        torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    )


    best_f1 = -1.0
    best_epoch = -1
    history = []
    fold_dir = results_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0
        skipped = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", enabled=True):
                    logits = model(x)
                    loss = loss_fn(logits, y)
            else:
                logits = model(x)
                loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            running += loss.item() * x.size(0)
            n_seen += x.size(0)
        sched.step()
        train_loss = running / max(n_seen, 1)

        y_true, y_prob = predict(model, val_loader, device)
        metrics = compute_metrics(y_true, y_prob)
        metrics.update({"epoch": epoch, "train_loss": train_loss, "lr": sched.get_last_lr()[0], "time_s": time.time() - t0})
        history.append(metrics)
        tag = f" (skipped {skipped} non-finite batches)" if skipped else ""
        print(f"[fold {fold}] epoch {epoch}: loss={train_loss:.4f} val_macro_f1={metrics['macro_f1']:.4f} mAP={metrics['mean_ap']:.4f}{tag}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), fold_dir / "best.pt")

    save_metrics({"history": history, "best_epoch": best_epoch, "best_macro_f1": best_f1}, fold_dir / "metrics.json")
    return {"fold": fold, "best_epoch": best_epoch, "best_macro_f1": best_f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--backbone-lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true", help="2 epochs on fold 0 only")
    args = ap.parse_args()

    if args.smoke_test:
        args.folds = [0]
        args.epochs = 2

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    vocab = VocabInfo.load(data_dir / "label_vocab.json")
    device = pick_device(args.device)
    print(f"Device: {device}, num_labels: {vocab.num_labels}")

    summary = []
    for fold in args.folds:
        summary.append(train_one_fold(
            fold=fold,
            data_dir=data_dir,
            results_dir=results_dir,
            vocab=vocab,
            epochs=args.epochs,
            batch_size=args.batch_size,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            device=device,
            amp=args.amp,
            seed=args.seed,
        ))

    mean_f1 = float(np.mean([s["best_macro_f1"] for s in summary]))
    std_f1 = float(np.std([s["best_macro_f1"] for s in summary]))
    (results_dir / "summary.json").write_text(json.dumps({
        "folds": summary,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "args": vars(args),
    }, indent=2))
    print(f"\nMean macro-F1: {mean_f1:.4f} +/- {std_f1:.4f}")


if __name__ == "__main__":
    main()
