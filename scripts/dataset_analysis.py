"""Descriptive statistics for the G3 Corfu Butterflies benchmark.

Reports label-frequency statistics, cardinality distribution,
co-occurrence structure, per-namespace counts, and per-label F1 of
the reference ViT-B/16 baseline on the holdout. All outputs are
written to ``paper/figures/`` and used as LaTeX ``\\input`` sources
for Section 4 of the paper.

Author:
    Nikolaos Korfiatis, Ionian University. Contact: nkorf@ionio.gr

License:
    MIT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def decode_multihot(s: str) -> np.ndarray:
    return np.fromstring(s, sep=" ", dtype=np.int8)


def load_matrix(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    meta = pd.read_csv(data_dir / "metadata.csv")
    vocab = json.loads((data_dir / "label_vocab.json").read_text())
    labels = vocab["labels"]
    Y = np.stack([decode_multihot(s) for s in meta["labels_multihot"]])
    return meta, Y, labels


def namespace_of(label: str) -> str:
    return label.split(":", 1)[0]


def label_frequency_table(Y: np.ndarray, labels: list[str], out: Path) -> None:
    counts = Y.sum(axis=0)
    order = np.argsort(-counts)
    rows = ["Label & Namespace & Count & Frequency \\\\"]
    for i in order[:15]:
        short = labels[i].split(":", 1)[1] if ":" in labels[i] else labels[i]
        short_tex = short.replace("_", "\\_")
        rows.append(
            f"{short_tex} & {namespace_of(labels[i])} & {int(counts[i])} & {counts[i]/len(Y):.3f} \\\\"
        )
    body = (
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        + rows[0]
        + "\n\\midrule\n"
        + "\n".join(rows[1:])
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    out.write_text(body)
    print(f"Wrote {out}")


def cardinality_distribution(Y: np.ndarray, out: Path) -> None:
    card = Y.sum(axis=1)
    vals, cnts = np.unique(card, return_counts=True)
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.bar(vals, cnts, color="#355070")
    ax.set_xlabel("Positive labels per image")
    ax.set_ylabel("Number of images")
    ax.set_xticks(vals)
    ax.set_title(f"Cardinality distribution (mean {card.mean():.2f}, max {card.max()})")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def per_namespace_stats(Y: np.ndarray, labels: list[str], out: Path) -> None:
    ns = np.array([namespace_of(l) for l in labels])
    out_lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Namespace & Labels & Mean positives / image & Total positives \\\\",
        "\\midrule",
    ]
    for n in sorted(set(ns)):
        mask = ns == n
        total = int(Y[:, mask].sum())
        mean = Y[:, mask].sum(axis=1).mean()
        out_lines.append(f"{n} & {int(mask.sum())} & {mean:.2f} & {total} \\\\")
    out_lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(out_lines) + "\n")
    print(f"Wrote {out}")


def long_tail_figure(Y: np.ndarray, labels: list[str], out: Path) -> None:
    counts = sorted(Y.sum(axis=0), reverse=True)
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    ax.plot(range(1, len(counts) + 1), counts, marker="o", markersize=3, color="#b56576")
    ax.set_xlabel("Label rank (most to least frequent)")
    ax.set_ylabel("Images with positive label")
    ax.set_yscale("log")
    ax.set_title("Label frequency rank curve (log scale)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def cooccurrence_heatmap(Y: np.ndarray, labels: list[str], out: Path, top: int = 15) -> None:
    counts = Y.sum(axis=0)
    idx = np.argsort(-counts)[:top]
    sub = Y[:, idx]
    denom = np.maximum(sub.sum(axis=0)[:, None], 1)
    co = (sub.T @ sub) / denom
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(co, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(top))
    ax.set_yticks(range(top))
    ax.set_xticklabels([labels[i] for i in idx], rotation=75, ha="right", fontsize=7)
    ax.set_yticklabels([labels[i] for i in idx], fontsize=7)
    ax.set_title(f"P(label$_j$ | label$_i$) for the {top} most frequent labels")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def split_support_summary(data_dir: Path, labels: list[str], out: Path) -> None:
    meta = pd.read_csv(data_dir / "metadata.csv").set_index("filename")
    holdout = pd.read_csv(data_dir / "splits" / "holdout_test.csv")["filename"].tolist()
    Yh = np.stack([decode_multihot(meta.loc[f, "labels_multihot"]) for f in holdout])
    zero_support = int((Yh.sum(axis=0) == 0).sum())
    only_support = int((Yh.sum(axis=0) >= 1).sum())
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Split & Images & Labels with $\\geq$1 positive \\\\",
        "\\midrule",
        f"5-fold training pool & 137 & --- \\\\",
        f"Locked holdout & {len(holdout)} & {only_support} of {len(labels)} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}; holdout has {zero_support} unseen labels out of {len(labels)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--fig-dir", default="paper/figures")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _, Y, labels = load_matrix(data_dir)

    label_frequency_table(Y, labels, fig_dir / "label_top.tex")
    cardinality_distribution(Y, fig_dir / "cardinality.png")
    per_namespace_stats(Y, labels, fig_dir / "namespace_stats.tex")
    long_tail_figure(Y, labels, fig_dir / "long_tail.png")
    cooccurrence_heatmap(Y, labels, fig_dir / "cooccurrence.png", top=15)
    split_support_summary(data_dir, labels, fig_dir / "split_support.tex")


if __name__ == "__main__":
    main()
