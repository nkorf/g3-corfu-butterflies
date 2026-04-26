# Benchmark Reproduction

## 1. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Recommended hardware: any machine with >= 8 GB RAM. With `--device cpu` a
full 5-fold run (30 epochs × 5 folds) takes ~45 minutes; on Apple MPS or
a CUDA GPU it finishes in under 10 minutes.

## 2. Prepare the dataset

```bash
python scripts/prepare_dataset.py
```

This writes:

- `data/metadata.csv` — per-image multi-hot labels
- `data/label_vocab.json` — label order (stable, reproducible)
- `data/splits/holdout_test.csv` — 15 % stratified holdout
- `data/splits/fold_{0..4}.csv` — 5 stratified CV folds over the remainder

Seed is fixed to 42; rerunning produces identical splits.

## 3. Smoke test

```bash
python -m g3.train --smoke-test
```

2-epoch run on fold 0. Verifies the pipeline end-to-end in ~2 minutes.

## 4. Full 5-fold training

```bash
python -m g3.train                      # all defaults
# or tune:
python -m g3.train --epochs 30 --batch-size 16 \
                   --backbone-lr 3e-5 --head-lr 1e-3 \
                   --device auto --amp
```

Outputs:
- `results/fold_{i}/best.pt` — best checkpoint by val macro-F1
- `results/fold_{i}/metrics.json` — per-epoch history
- `results/summary.json` — mean / std macro-F1 across folds

## 5. Evaluate on the locked holdout

```bash
python -m g3.holdout_eval
```

Writes `results/holdout_metrics.json` with per-fold and ensemble-mean
metrics (macro-F1, micro-F1, exact-match, mean AUROC, mean AP).

## 6. Faster runs on a free GPU

### Google Colab (recommended)

The fastest path to reproduce the benchmark is a free Colab T4 GPU:
a full 5-fold run finishes in about 15 minutes. Open
`notebooks/colab_quickstart.ipynb` in Colab
(`Runtime → Change runtime type → T4 GPU`) and run all cells. Mixed
precision is enabled by default (`--amp`) and approximately halves
runtime on CUDA.

### Kaggle Notebooks

Kaggle offers a free P100 or T4 GPU with a 30-hour weekly quota. Create
a new notebook, attach this repository as a dataset (or `!git clone`
it), and run:

```bash
!pip install --quiet iterative-stratification
!pip install --quiet -e .
!python scripts/prepare_dataset.py
!python -m g3.train --device cuda --amp --batch-size 32
!python -m g3.holdout_eval --device cuda
```

### Apple Silicon (MPS)

`--device mps` targets Apple Silicon GPUs. PyTorch 2.11 exhibits
numerical instabilities in BCE and attention kernels on some
fold seeds, so the CPU path is the recommended local default. Stable
MPS support is tracked as future work (see the paper's *Future Work*
section).

## 7. Reporting

All numbers in the paper come from `results/summary.json` and
`results/holdout_metrics.json`. Rerunning with the same seed reproduces
them bit-for-bit on the same hardware.

## 8. Optional: regenerate figures and embeddings

The paper's descriptive-statistics figures are rendered with R/ggplot2:

```bash
Rscript scripts/figures.R       # writes paper/figures/{cardinality,long_tail,cooccurrence}.png
```

The shipped image embeddings (`data/embeddings/`) can be recomputed
from the source images with:

```bash
python scripts/extract_embeddings.py --device cpu   # ~1 min on CPU
```

Both scripts are deterministic given the pinned weight tags
(`open_clip:ViT-B-32:laion2b_s34b_b79k`,
`torchvision:vit_b_16:IMAGENET1K_V1`) and the shipped images.
