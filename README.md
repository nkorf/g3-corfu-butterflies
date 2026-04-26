# G3 Corfu Butterflies

A benchmark dataset and ViT-B/16 baseline for multi-label visual-attribute
classification of butterfly images photographed in Corfu, Greece.

- **Dataset DOI**: [10.7910/DVN/W6UMIR](https://doi.org/10.7910/DVN/W6UMIR) (Harvard Dataverse)
- **Images**: 165 JPEG photographs
- **Task**: multi-label classification over color, pattern, and extra-feature
  tokens (no species labels — see *Limitations*)
- **Baseline**: ViT-B/16 (ImageNet pretrained) fine-tuned with 5-fold CV

## Quickstart

```bash
cd /path/to/g3
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/prepare_dataset.py
python -m g3.train --smoke-test
python -m g3.train                  # full 5-fold run
python -m g3.holdout_eval
```

See [BENCHMARK.md](BENCHMARK.md) for full instructions.

## Data layout

```
data/
├── butterflies_source.csv    # raw CSV shipped with the images
├── images/                   # 165 *.jpg
├── metadata.csv              # cleaned + multi-hot labels
├── label_vocab.json          # label ordering used by the models
├── splits/
│   ├── holdout_test.csv
│   └── fold_0.csv … fold_4.csv
└── embeddings/               # frozen feature matrices (derived)
    ├── clip_vitb32.npy       # (165, 512) float32, L2-normalised
    ├── vitb16_imagenet.npy   # (165, 768) float32, ImageNet-pretrained CLS
    ├── index.csv             # filename → row, aligned with metadata.csv
    └── manifest.json         # model identifiers + SHA-256 integrity
```

## Citation

If you use this dataset or its baselines in academic work, please cite:

> Korfiatis, N., Avlonitis, M., Kourouthanassis, P., Karyotis, V.,
> Ghinis, S., Gjoni, V., Gkinis, S. (2026). *G3 Corfu Butterflies:
> A Small-Scale Benchmark for Long-Tailed Multi-Label Attribute
> Recognition* \[Data set]. Harvard Dataverse.
> https://doi.org/10.7910/DVN/W6UMIR

BibTeX:

```bibtex
@dataset{korfiatis2026g3,
  title     = {G3 Corfu Butterflies: A Small-Scale Benchmark for
               Long-Tailed Multi-Label Attribute Recognition},
  author    = {Korfiatis, Nikolaos and Avlonitis, Markos and
               Kourouthanassis, Panos and Karyotis, Vasileios and
               Ghinis, Stamatis and Gjoni, Vojsava and Gkinis, Spiros},
  year      = {2026},
  publisher = {Harvard Dataverse},
  version   = {V1},
  doi       = {10.7910/DVN/W6UMIR},
  url       = {https://doi.org/10.7910/DVN/W6UMIR}
}
```

The accompanying data-descriptor paper (`paper/main.pdf` in this
repository) provides the full dataset description, evaluation protocol,
and reference baseline numbers.

## License

- **Dataset**: CC BY 4.0
- **Code**: MIT
