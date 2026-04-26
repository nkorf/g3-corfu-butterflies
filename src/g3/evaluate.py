"""Metrics and evaluation helpers for multi-label classification.

All metrics are computed in pure numpy so the module can be reused in
notebooks without a torch-metrics dependency. ``macro_f1`` is the
primary benchmark score. Labels with no positive example in the
evaluation set are excluded from the macro average (they still
contribute to the micro-averaged count).

Author:
    Nikolaos Korfiatis, Ionian University. Contact: nkorf@ionio.gr

License:
    MIT.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute macro/micro F1, exact-match, per-label AUROC, mAP.

    Pure numpy implementation keeps the dependency footprint tight.
    """
    y_pred = (y_prob >= threshold).astype(np.int8)
    eps = 1e-9

    tp = ((y_pred == 1) & (y_true == 1)).sum(axis=0)
    fp = ((y_pred == 1) & (y_true == 0)).sum(axis=0)
    fn = ((y_pred == 0) & (y_true == 1)).sum(axis=0)

    precision = tp / np.maximum(tp + fp, eps)
    recall = tp / np.maximum(tp + fn, eps)
    f1_per_label = 2 * precision * recall / np.maximum(precision + recall, eps)

    # Only average across labels that have any positives in the eval set.
    support_mask = (y_true.sum(axis=0) > 0)
    macro_f1 = float(f1_per_label[support_mask].mean()) if support_mask.any() else 0.0

    tp_all, fp_all, fn_all = tp.sum(), fp.sum(), fn.sum()
    micro_p = tp_all / max(tp_all + fp_all, eps)
    micro_r = tp_all / max(tp_all + fn_all, eps)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, eps)

    exact_match = float((y_pred == y_true).all(axis=1).mean())

    auroc_per_label = []
    ap_per_label = []
    for j in range(y_true.shape[1]):
        if not support_mask[j] or y_true[:, j].sum() == len(y_true):
            continue
        auroc_per_label.append(_auroc(y_true[:, j], y_prob[:, j]))
        ap_per_label.append(_average_precision(y_true[:, j], y_prob[:, j]))

    return {
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
        "exact_match": exact_match,
        "mean_auroc": float(np.mean(auroc_per_label)) if auroc_per_label else 0.0,
        "mean_ap": float(np.mean(ap_per_label)) if ap_per_label else 0.0,
        "labels_evaluated": int(support_mask.sum()),
    }


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    y_sorted = y[order]
    n_pos = y_sorted.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y) + 1)
    sum_ranks_pos = ranks[y_sorted == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_precision(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted)
    precision_at_k = tp_cum / np.arange(1, len(y) + 1)
    n_pos = y_sorted.sum()
    if n_pos == 0:
        return 0.0
    return float((precision_at_k * y_sorted).sum() / n_pos)


@torch.no_grad()
def predict(model, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def save_metrics(metrics: dict, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2))
