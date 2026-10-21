# LCA-PHC: Label-Conflict-Aware Pixel-Hash Curation for a Bangladeshi Chest X-ray Dataset

A reproducible Kaggle notebook that audits, cleans and benchmarks a public Bangladeshi chest X-ray dataset (normal vs. pneumonia). It implements the **Label-Conflict-Aware Pixel-Hash Curation protocol (LCA-PHC)** described in the accompanying MethodsX manuscript.

> **Paper:** *A Label-Conflict-Aware, Pixel-Hash Curation and Reproducible Benchmarking Protocol for a Bangladeshi Chest X-ray Dataset* — Md Irfanul Kabir Hira, Mst Moriom Akter Bithee (NITER, Dhaka University).
> **Source dataset:** [Mendeley Data V5, DOI 10.17632/wndbd5r26y.5](https://data.mendeley.com/datasets/wndbd5r26y/5) (CC BY 4.0)
> **Cleaned release:** _[PENDING: DOI of the cleaned version on Mendeley Data]_
> **Kaggle notebook:** _[PENDING: public notebook link]_

---

## Why this exists

Public chest radiograph collections are often released without an executable audit of repeated images and conflicting labels, which can silently distort benchmark results. Hashing file bytes (e.g. MD5) misses re-encoded copies, and keeping "one copy" of an image that appears under both class labels turns an unresolved disagreement into a confident label. LCA-PHC addresses this with explicit, auditable rules.

## What the protocol does

| Step | Description |
|---|---|
| 1. Checksum-pinned acquisition | Downloads the v5 archive and accepts it only if size (524,339,616 bytes) and SHA-256 match the pinned values. |
| 2. Decoded-pixel hashing | SHA-256 over image size + EXIF-oriented RGB pixels (not file bytes). File-byte hash is also stored. |
| 3. Duplicate / label-conflict resolution | Unreadable files excluded; **all members** of an identical-image group with conflicting labels excluded; otherwise one copy kept (original test copy first, then lexical path). |
| 4. Verified release build | Retained images copied byte-for-byte (no resize, recompression or metadata stripping); every copy re-hashed; manifests, data dictionary, dataset card, change log and `SHA256SUMS.txt` written. |
| 5. Descriptive statistics | Brightness, contrast, entropy, Laplacian-variance sharpness; review-only quality flags. |
| 6. Near-duplicate review | 64-bit perceptual hash, Hamming distance ≤ 4; candidates flagged (cross-split, label disagreement) but **never auto-deleted**. |
| 7. Fixed split | Stratified 80/20 train/validation on the cleaned train folder (seed 42); cleaned test folder held out. |
| 8. Locked benchmark | ResNet18, ResNet50, DenseNet121, EfficientNet-B0, MobileNetV3-Large; 3 seeds; validation-only checkpoint selection; majority-class baseline. |

## Key results (source v5)

| Item | Count |
|---|---|
| Raw images | 2,626 |
| Same-label duplicate copies removed | 25 |
| Images in conflicting-label groups (excluded) | 96 |
| Unreadable files | 0 |
| **Retained images** | **2,505** (1,994 train / 511 test) |
| Near-duplicate candidate pairs | 115 (41 cross train/test) |
| Test accuracy (fine-tuned CNNs) | 91.65 – 92.50 % (majority baseline 49.90 %) |

The expected-versus-observed counts are asserted in code; the run stops on any mismatch rather than forcing totals.

## Repository contents

```
.
├── chest-xray-curation-eda-benchmarks.ipynb   # Full pipeline (download → audit → release → EDA → benchmark → manuscript tables)
├── README.md
├── LICENSE
└── requirements.txt                            # optional, see below
```

> Rename the notebook file as you prefer and update the tree above.

## Quick start (Kaggle — recommended)

1. Create a new Kaggle notebook and upload the `.ipynb` file.
2. Turn **Internet ON**. Turn **GPU ON** (T4) if you want to run the benchmarks.
3. Run all cells in order. Read the expected-versus-observed audit before continuing.
4. After **Section 8**, run **Section 8b** to copy the cleaned ZIP to `/kaggle/working/` and download it from the **Output** panel (or *Save Version → Save & Run All*, then download from the version's Output tab).

Set `RUN_BENCHMARKS = False` in the configuration cell to export the cleaned dataset without training.

## Configuration flags (Section 1)

| Flag | Default | Meaning |
|---|---|---|
| `STRICT_EXPECTED_AUDIT` | `True` | Stop if archive hash or counts differ from the audited values |
| `EXPORT_CLEAN_DATASET` | `True` | Build the cleaned release ZIP |
| `RUN_NEAR_DUPLICATE_REVIEW` | `True` | Perceptual-hash review |
| `PHASH_MAX_DISTANCE` | `4` | Candidate threshold (not proof of same patient/image) |
| `RUN_BENCHMARKS` | `True` | Train and evaluate all models |
| `BENCHMARK_SEEDS` | `[42, 43, 44]` | Training seeds |
| `EPOCHS`, `BATCH_SIZE` | `12`, `16` | Training budget (reduce batch to 8 on GPU out-of-memory; this creates a new protocol fingerprint) |

## Benchmark protocol

- **Input:** 224 × 224 RGB, aspect-preserving black padding, ImageNet normalization
- **Augmentation:** random rotation ±5° (training only)
- **Optimizer:** AdamW, lr 1e-4, weight decay 1e-4; cosine annealing over 12 epochs; cross-entropy
- **Selection:** highest validation macro-F1 (ties → lower validation loss); test set used once per run
- **Metrics:** accuracy, macro precision/recall/F1, balanced accuracy, sensitivity, specificity, ROC-AUC, average precision, Brier score, 15-bin ECE
- **Reporting:** mean ± sample SD over seeds; ROC/PR/confusion figures use seed 42 (not the best seed)
- **Reproducibility:** seeds for Python/NumPy/PyTorch, deterministic cuDNN, SHA-256 run fingerprint

## Outputs

- **Cleaned dataset ZIP:** `images/{train,test}/{normal,pneumonia}/`, `manifest.csv`, `data_dictionary.csv`, `benchmark_splits.csv`, `source_inventory_and_decisions.csv`, exclusion lists, `near_duplicate_candidates_REVIEW_ONLY.csv`, `dataset_card.json`, `CHANGELOG.md`, `SHA256SUMS.txt`
- **Benchmark results:** per-model/seed metrics, training histories, saved predictions, tables, ROC/PR/confusion figures (PNG + PDF)
- **Manuscript support files:** specifications table, methods and limitations text

## Environment

Python 3.13 · PyTorch 2.11.0 · torchvision 0.26.0 · ImageHash 4.3.2 · scikit-learn · Pillow · SciPy · pandas · matplotlib · seaborn · tqdm · Kaggle 2 × NVIDIA T4 (single GPU used per run).
The notebook installs missing packages itself and keeps Kaggle's matched torch/torchvision.

## Limitations

- Labels come from source folders; they were **not** re-adjudicated or corrected.
- Splits are **image-level**; no patient identifiers exist, so patient independence is not established.
- Exact deduplication does not remove crops, recompressions or repeat examinations; perceptual-hash candidates need manual review.
- Images keep source bytes, embedded metadata and visible annotations — review privacy before redistribution.
- The source repository text states 3,355 images, but the archive contains 2,626; the archive and computed manifests govern this work.
- Models are validation instruments, **not** clinically validated diagnostic tools.

## Citation

If you use this code or the cleaned release, please cite the paper and the source dataset:

```bibtex
@article{hira_lcaphc,
  title   = {A Label-Conflict-Aware, Pixel-Hash Curation and Reproducible Benchmarking Protocol for a Bangladeshi Chest X-ray Dataset},
  author  = {Hira, Md Irfanul Kabir and Bithee, Mst Moriom Akter},
  journal = {MethodsX},
  year    = {2026},
  note    = {[PENDING: volume, pages, DOI]}
}

@misc{epic_chittagong_cxr_v5,
  title        = {A Primary Chest X-ray Dataset of Normal Bangladesh, Version 5},
  howpublished = {Mendeley Data},
  doi          = {10.17632/wndbd5r26y.5},
  url          = {https://data.mendeley.com/datasets/wndbd5r26y/5}
}
```

## License

- **Code:** MIT License (add a `LICENSE` file; change if you prefer another license).
- **Source images:** CC BY 4.0 — attribution to the original dataset authors is required.

## Contact

Md Irfanul Kabir Hira — irfanhira11@niter.edu.bd
Department of CSE, National Institute of Textile Engineering and Research (NITER), Savar, Dhaka-1350, Bangladesh
