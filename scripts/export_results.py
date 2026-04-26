"""Assemble the Table 2 LaTeX input from the training / holdout JSON.

Aggregates the per-fold best-epoch metrics of each baseline into
5-fold mean :math:`\\pm` standard deviation, pairs them with the
locked-holdout numbers, and emits a two-baseline comparison table
(CLIP linear probe and fine-tuned ViT-B/16) written to
``paper/figures/baseline_table.tex``.

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


METRICS = ("macro_f1", "micro_f1", "exact_match", "mean_auroc", "mean_ap")
METRIC_LABELS = {
    "macro_f1": "Macro-F1",
    "micro_f1": "Micro-F1",
    "exact_match": "Exact match",
    "mean_auroc": "Mean AUROC",
    "mean_ap": "Mean AP",
}


def fmt(mean: float, std: float | None = None) -> str:
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def vit_stats(results: Path) -> tuple[dict, dict, dict]:
    fold_dirs = sorted(p for p in results.glob("fold_*") if p.is_dir())
    per_fold = []
    for d in fold_dirs:
        payload = json.loads((d / "metrics.json").read_text())
        per_fold.append(payload["history"][payload["best_epoch"]])
    arr = np.array([[f[m] for m in METRICS] for f in per_fold])
    cv = {m: (arr[:, i].mean(), arr[:, i].std()) for i, m in enumerate(METRICS)}
    ho = json.loads((results / "holdout_metrics.json").read_text())["ensemble"]
    return cv, ho, {"folds": len(per_fold)}


def clip_stats(results: Path) -> tuple[dict, dict]:
    data = json.loads((results / "clip_linear_probe.json").read_text())
    arr = np.array([[f[m] for m in METRICS] for f in data["cv"]])
    cv = {m: (arr[:, i].mean(), arr[:, i].std()) for i, m in enumerate(METRICS)}
    ho = data["holdout"]
    return cv, ho


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="paper/figures/baseline_table.tex")
    args = ap.parse_args()

    results = Path(args.results_dir)
    vit_cv, vit_ho, _ = vit_stats(results)
    clip_cv, clip_ho = clip_stats(results)

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{5-fold CV} & \\multicolumn{2}{c}{Locked holdout} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Metric & CLIP probe & ViT-B/16 FT & CLIP probe & ViT-B/16 FT \\\\",
        "\\midrule",
    ]
    for m in METRICS:
        label = METRIC_LABELS[m]
        c_cv = fmt(*clip_cv[m])
        v_cv = fmt(*vit_cv[m])
        c_ho = fmt(clip_ho[m])
        v_ho = fmt(vit_ho[m])
        lines.append(f"{label} & {c_cv} & {v_cv} & {c_ho} & {v_ho} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
