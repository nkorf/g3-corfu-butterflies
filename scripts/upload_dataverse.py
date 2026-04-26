#!/usr/bin/env python3
"""Upload the v1.0 release files to the Harvard Dataverse draft.

Reads the API token from the DV_TOKEN env var so it never lands in shell
history. Walks a static manifest derived from DATAVERSE_METADATA.md and
posts each file via the native Dataverse add-file endpoint.

Idempotent: if a file with the same name+directoryLabel already exists,
the API will refuse with a 400 — we surface that and skip rather than
duplicate. (To replace an existing file use /api/files/{id}/replace.)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = "https://dataverse.harvard.edu"
PID  = "doi:10.7910/DVN/W6UMIR"
ROOT = Path("/Users/nkorf/Documents/coding_projects/g3/upload")

# (local_path_relative_to_upload, directoryLabel, description)
MANIFEST: list[tuple[str, str | None, str]] = [
    ("README.md",                              None,              "Top-level overview; Dataverse renders this inline."),
    ("main.pdf",                               None,              "Data-descriptor paper, ~10 pages, reproducible numbers."),
    ("LICENSE",                                None,              "CC BY 4.0 (data) + MIT (code)."),
    ("BENCHMARK.md",                           None,              "Step-by-step reproduction recipe."),
    ("g3-code-v1.0.zip",                       None,              "Reference implementation (src/, scripts/, notebooks/, requirements.txt, LICENSE)."),

    ("data/images.zip",                        "data",            "165 JPEG photographs (~73 MB uncompressed)."),
    ("data/metadata.csv",                      "data",            "Per-image multi-label annotations and raw attribute strings."),
    ("data/label_vocab.json",                  "data",            "The 45-entry multi-label vocabulary in canonical order."),

    ("data/splits/holdout_test.csv",           "data/splits",     "28 locked holdout filenames (seed 42)."),
    ("data/splits/fold_0.csv",                 "data/splits",     "Iteratively-stratified CV fold 0 over the 137 non-holdout images."),
    ("data/splits/fold_1.csv",                 "data/splits",     "Iteratively-stratified CV fold 1 over the 137 non-holdout images."),
    ("data/splits/fold_2.csv",                 "data/splits",     "Iteratively-stratified CV fold 2 over the 137 non-holdout images."),
    ("data/splits/fold_3.csv",                 "data/splits",     "Iteratively-stratified CV fold 3 over the 137 non-holdout images."),
    ("data/splits/fold_4.csv",                 "data/splits",     "Iteratively-stratified CV fold 4 over the 137 non-holdout images."),

    ("data/embeddings/clip_vitb32.npy",        "data/embeddings", "Frozen OpenCLIP ViT-B/32 features, (165, 512) float32, L2-normalised."),
    ("data/embeddings/vitb16_imagenet.npy",    "data/embeddings", "Frozen torchvision ViT-B/16 (IMAGENET1K_V1) pooled CLS features, (165, 768) float32."),
    ("data/embeddings/index.csv",              "data/embeddings", "filename,row mapping aligning the two .npy files with metadata.csv."),
    ("data/embeddings/manifest.json",          "data/embeddings", "Model identifiers, shapes, dtypes, and SHA-256 of each .npy."),

    ("results/summary.json",                   "results",         "Per-fold ViT-B/16 best-epoch metrics + 5-fold mean/std."),
    ("results/holdout_metrics.json",           "results",         "Per-fold and ensemble holdout metrics for ViT-B/16."),
    ("results/clip_linear_probe.json",         "results",         "CLIP ViT-B/32 linear probe CV + holdout metrics."),
]


def upload_one(token: str, local_path: Path, directory_label: str | None, description: str) -> dict:
    url = f"{BASE}/api/datasets/:persistentId/add"
    json_data: dict[str, object] = {"description": description, "categories": ["Data"]}
    if directory_label:
        json_data["directoryLabel"] = directory_label
    with local_path.open("rb") as fh:
        resp = requests.post(
            url,
            params={"persistentId": PID},
            headers={"X-Dataverse-key": token},
            files={"file": (local_path.name, fh)},
            data={"jsonData": json.dumps(json_data)},
            timeout=600,
        )
    try:
        body = resp.json()
    except json.JSONDecodeError:
        body = {"raw": resp.text[:400]}
    return {"status_code": resp.status_code, "body": body}


def main() -> int:
    token = os.environ.get("DV_TOKEN")
    if not token:
        print("ERROR: DV_TOKEN env var not set", file=sys.stderr)
        return 2

    total = len(MANIFEST)
    ok = 0
    fail = 0
    print(f"Uploading {total} files to {PID}\n")
    for i, (rel, dirlabel, desc) in enumerate(MANIFEST, 1):
        path = ROOT / rel
        if not path.exists():
            print(f"[{i:>2}/{total}] MISSING {rel}")
            fail += 1
            continue
        size_mb = path.stat().st_size / 1_048_576
        loc = f"{(dirlabel + '/') if dirlabel else ''}{path.name}"
        print(f"[{i:>2}/{total}] {loc:<45} {size_mb:>7.2f} MB  ", end="", flush=True)
        t0 = time.time()
        try:
            result = upload_one(token, path, dirlabel, desc)
        except requests.RequestException as e:
            print(f"NETWORK ERROR: {e}")
            fail += 1
            continue
        dt = time.time() - t0
        sc = result["status_code"]
        if sc in (200, 201):
            ok += 1
            print(f"OK ({dt:.1f}s)")
        else:
            fail += 1
            msg = ""
            body = result["body"]
            if isinstance(body, dict):
                msg = body.get("message") or body.get("status") or json.dumps(body)[:200]
            print(f"FAIL [{sc}] {msg}")

    print(f"\nUploaded: {ok}/{total}   Failed: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
