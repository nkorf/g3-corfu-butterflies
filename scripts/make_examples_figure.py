"""Generate a 2x3 panel of example photographs with attribute labels.

Produces ``paper/figures/examples.png``, the six-image figure in
Section 2 of the accompanying paper. The dataset carries
image-level attribute annotations only; this figure visualises how
those annotations appear alongside each photograph.

Author:
    Nikolaos Korfiatis, Ionian University. Contact: nkorf@ionio.gr

License:
    MIT.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="paper/figures/examples.png")
    ap.add_argument("--picks", nargs="+", default=[
        "image(1).jpg", "image(10).jpg", "image(30).jpg",
        "image(60).jpg", "image(90).jpg", "image(120).jpg",
    ])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    meta = pd.read_csv(data_dir / "metadata.csv").set_index("filename")

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, fname in zip(axes.flat, args.picks):
        row = meta.loc[fname]
        img = Image.open(data_dir / "images" / fname).convert("RGB")
        ax.imshow(img)
        ax.set_axis_off()
        caption = (
            f"color: {row['color']}\n"
            f"pattern: {row['pattern']}\n"
            f"extra: {row['extra_features']}"
        )
        ax.set_title(caption, fontsize=8, loc="left", pad=4)

    fig.suptitle(
        "G3 Corfu Butterflies: example photographs with attribute labels",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
