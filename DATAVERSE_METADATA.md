# Dataverse Upload Package (DOI 10.7910/DVN/W6UMIR)

This file drives the Dataverse draft edit pass before publishing
v1.0. Everything below is verified against the shipped files and the
accompanying paper.

## Title

> G3 Corfu Butterflies: A Small-Scale Benchmark for Long-Tailed
> Multi-Label Attribute Recognition

(Matches the paper title verbatim; replaces the earlier
"Multi-Label Visual-Attribute Classification" wording for
consistency.)

## Description (paste into the Dataverse description field)

G3 Corfu Butterflies is a public benchmark for small-scale,
long-tailed, multi-label attribute recognition. It is designed as a
tight, reproducible test-bed for computer-vision methods that target
data-efficient learning, multi-label classification with strong class
imbalance, and compositional attribute transfer. The subject matter
(butterfly photographs collected in Corfu, Greece, by Stamatis Ghinis,
Vojsava Gjoni, and Spiros Gkinis) is the data substrate; the
contribution is the benchmark layer built on top of it.

Contents of v1.0:
- 165 high-resolution JPEG images (~73 MB total);
- image-level multi-label annotations across three attribute families
  (colour, pattern, additional visual features), tokenised into a
  45-entry vocabulary in a fixed order;
- canonical evaluation splits: a locked 15% holdout test set
  (28 images) and five iteratively-stratified cross-validation folds
  over the remaining 137 images, all at seed 42;
- two reference baselines with fully documented hyper-parameters:
  a frozen CLIP ViT-B/32 linear probe and a fine-tuned ViT-B/16;
- two precomputed image-embedding matrices aligned row-for-row with
  `metadata.csv` (CLIP ViT-B/32, 512-d L2-normalised; torchvision
  ViT-B/16 ImageNet1k, 768-d pooled CLS) for downstream
  retrieval / clustering / weak-supervision studies; and
- a data-descriptor paper (main.pdf) with full reproduction
  instructions, a descriptive-statistics section, and a Colab/Kaggle
  quickstart notebook.

Reference numbers on the shipped splits:
- CLIP linear probe: 5-fold macro-F1 0.291 ± 0.046; holdout 0.256.
- Fine-tuned ViT-B/16: 5-fold macro-F1 0.395 ± 0.040; holdout
  ensemble macro-F1 0.342, micro-F1 0.563, mAP 0.443.

Evaluation-active labels. Because the benchmark is deliberately
long-tailed and the holdout is small, 24 of 45 labels have at least
one positive example in the locked holdout. Macro-averaged metrics
are reported over those 24 labels; the remaining labels contribute
only to the micro-averaged counts. This is fully documented in
Section 3 of the paper.

Primary uses (computer-vision research):
- data-efficient learning on O(10^2)-image budgets;
- long-tail multi-label classification and loss-reweighting studies;
- representation comparison of frozen foundation features against
  full fine-tuning;
- label-correlation modelling on a corpus small enough to iterate on;
- compositional transfer across attribute families.

Not included in v1.0: species labels, region-level masks/bounding
boxes, GPS coordinates, and per-image timestamps. These extensions
are listed under *Future Work* in the accompanying paper.

Dedication. The dataset is dedicated to the memory of
Professor Markos Avlonitis, whose sustained contributions to the
computational study of biodiversity in the Ionian Islands shaped
both the scientific culture in which this work was carried out and
the motivation to build open, reusable resources of this kind.

Please cite the dataset when using it in academic or applied
research, and contact the corresponding author for collaboration
or feedback.

## Authors (for the Dataverse citation block)

Ordered so that the Dataverse citation (APA-style by default) reads
naturally. The same order appears in the README citation block and
the paper.

- Korfiatis, Nikolaos; Ionian University; nkorf@ionio.gr
  (benchmark design, code, reference baselines, corresponding author)
- Avlonitis, Markos; Ionian University
  (in memoriam; senior co-author, computational biodiversity)
- Kourouthanassis, Panos; Ionian University
- Karyotis, Vasileios; Ionian University
- Ghinis, Stamatis; Corfu Butterfly Conservation
- Gjoni, Vojsava; IRBIM-CNR, Mazara del Vallo, Italy (vojsava.gjoni@irbim.cnr.it)
- Gkinis, Spiros; Corfu Butterfly Conservation

## Contact

Nikolaos Korfiatis — nkorf@ionio.gr (Ionian University).

## Subject

Primary: Computer and Information Science.
Secondary: Environmental Sciences (for biodiversity discoverability).

## Keywords

Multi-label classification; Visual attributes; Small-data benchmark;
Long-tail recognition; Vision Transformer; CLIP linear probe;
Data-efficient learning; Butterfly; Biodiversity; Corfu.

## Temporal coverage

2023-2025 (field-work window during which the images were captured).

## Spatial coverage

Corfu, Greece (approximate centre 39.62° N, 19.92° E).

## Related materials

- Paper (main.pdf, included in this release).
- Reference implementation (shipped as g3-code-v1.0.zip).
- Colab quickstart notebook (included in g3-code-v1.0.zip).

## License

- Data (images + annotations + splits): Creative Commons Attribution
  4.0 International (CC BY 4.0).
- Code (src/g3/, scripts/, notebooks/): MIT License.

## File-level descriptions

Apply these one-line descriptions to the uploaded files. Dataverse
displays them on the landing page and uses them for search.

| File | Description |
|------|-------------|
| `README.md` | Top-level overview; Dataverse renders this inline. |
| `paper/main.pdf` | Data-descriptor paper, ~8 pages, reproducible numbers. |
| `data/images.zip` | 165 JPEG photographs (~73 MB uncompressed). |
| `data/metadata.csv` | Per-image multi-label annotations and raw attribute strings. |
| `data/label_vocab.json` | The 45-entry multi-label vocabulary in canonical order. |
| `data/splits/holdout_test.csv` | 28 locked holdout filenames (seed 42). |
| `data/splits/fold_0.csv` … `fold_4.csv` | Five iteratively-stratified CV folds over the remaining 137 images. |
| `data/embeddings/clip_vitb32.npy` | Frozen OpenCLIP ViT-B/32 features, (165, 512) float32, L2-normalised. |
| `data/embeddings/vitb16_imagenet.npy` | Frozen torchvision ViT-B/16 (IMAGENET1K_V1) pooled CLS features, (165, 768) float32. |
| `data/embeddings/index.csv` | `filename,row` mapping aligning the two `.npy` files with `metadata.csv`. |
| `data/embeddings/manifest.json` | Model identifiers, shapes, dtypes, and SHA-256 of each `.npy`. |
| `g3-code-v1.0.zip` | Reference implementation (src/, scripts/, notebooks/, requirements.txt, LICENSE). |
| `results/summary.json` | Per-fold ViT-B/16 best-epoch metrics + 5-fold mean/std. |
| `results/holdout_metrics.json` | Per-fold and ensemble holdout metrics for ViT-B/16. |
| `results/clip_linear_probe.json` | CLIP ViT-B/32 linear probe CV + holdout metrics. |
| `LICENSE` | CC BY 4.0 (data) + MIT (code). |
| `BENCHMARK.md` | Step-by-step reproduction recipe. |

## Pre-publish checklist

- [ ] Draft title updated to the verbatim paper title.
- [ ] Description pasted from the block above.
- [ ] Author list ordered and affiliated as above; Markos Avlonitis
      marked as deceased (in memoriam) in the description.
- [ ] Subject set to *Computer and Information Science* (primary)
      and *Environmental Sciences* (secondary).
- [ ] Keywords block pasted verbatim.
- [ ] Temporal and spatial coverage fields populated.
- [ ] License changed to CC BY 4.0 (and MIT mentioned in description).
- [ ] File-level descriptions applied.
- [ ] Files uploaded exactly once each; no duplicates.
- [ ] Preview the public view; confirm `README.md` renders; confirm
      the Markos Avlonitis dedication is visible.
- [ ] Click *Publish → v1.0*.
- [ ] Copy the resulting DOI back into the paper (already done) and
      into README.md (already done).
- [ ] Post the paper PDF link on the Dataverse *Related Publication*
      field once hosted externally.
