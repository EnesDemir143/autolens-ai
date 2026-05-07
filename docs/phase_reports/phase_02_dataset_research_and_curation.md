# Phase 02 Report — Dataset Research and Curation

## Goal

Build a documented local dataset pipeline for the eight AutoLens AI car body classes, producing source manifests, EDA evidence, duplicate/review lists, and train/validation/internal-test split files before training.

## Implementation

Phase 2 was executed against the local `datasets/` directory because the user already supplied dataset folders. The implementation added a small, explicit Python command:

```bash
uv run python -m autolens_ai.data.build_phase2_dataset --dataset-root datasets --output-dir artifacts/dataset
```

The command scans local images, infers source labels from folder names or supplied CSV metadata where available, normalizes safe labels into the eight assignment classes, records file metadata and SHA-256 hashes, excludes duplicate/corrupt images, and writes report-ready CSV artifacts.

## Outputs

- `src/autolens_ai/data/labels.py`
- `src/autolens_ai/data/build_phase2_dataset.py`
- `configs/data/phase2_sources.json`
- `docs/dataset_manifest.md`
- `artifacts/dataset/source_catalog.csv`
- `artifacts/dataset/manifests/*.csv`
- `artifacts/dataset/eda/*.csv`
- `artifacts/dataset/splits/*.csv`
- `artifacts/dataset/dataset_decision_note.md`

## Dataset result

- Raw images scanned: **64,302**
- Split-eligible unique accepted images: **31,612**
- Target: all safe accepted unique images, while reporting the 4,000/class target

Weak classes are documented in `artifacts/dataset/eda/balance_audit.csv`. MICRO is populated only from the user-approved FIAT 500 / Smart Fortwo model whitelist; generic City Car remains excluded/review-only.

## Verification

Passed:

```bash
uv run python -m autolens_ai.data.build_phase2_dataset --dataset-root datasets --output-dir artifacts/dataset
uv run pytest
uv run ruff check src tests
uv run mypy src
```

## Handoff to Phase 3

Phase 3 should consume `artifacts/dataset/splits/all_splits.csv`, apply class balancing or weighted sampling, and avoid training directly from raw dataset folders. Weak classes need targeted manual audit or supplementary collection before final model selection. Because usable data is limited, outlier removal is low priority; if later baseline metrics or leakage review justify it, run it as an explicit before/after experiment: keep pre-outlier counts and metrics, create a post-outlier artifact version, then compare class counts, source coverage, and validation metrics before selecting the training manifest.

## Extra MICRO model-name audit

A model-name filter was added for local sources that expose vehicle model names. It writes `artifacts/dataset/eda/micro_model_candidates.csv` and `micro_model_candidate_counts.csv`. The final MICRO policy uses only user-approved whitelist matches; local accepted matches are FIAT 500 and Smart Fortwo.
### User-approved MICRO model search whitelist

The MICRO model-name audit is now restricted to the user-approved whitelist only: Smart Fortwo, other Smart micro models (Forfour, Roadster, Crossblade), Citroën Ami, Renault Twizy, Toyota/Scion iQ, Tata Nano, Aixam City, Ligier JS50, Peel P50, BMW Isetta, and Fiat 500/500e. Generic `City Car` is intentionally not included.

## Final MICRO policy

MICRO is not sourced from generic `City Car`. It is populated only from the user-approved model whitelist. In the current local datasets, the matched unique-valid MICRO models are FIAT 500 (126 images) and Smart Fortwo (80 images), for **206 MICRO images** total.

## Stanford source exclusion

`stanford-car-body-type-data` is excluded from the final Phase 2 dataset to reduce overlap/leakage risk with the other Stanford-derived source. `stanford-car-dataset-by-classes-folder` remains as the single Stanford-derived source. The exclusion is recorded in `artifacts/dataset/excluded_source_catalog.csv`.

## Perceptual near-duplicate audit

Phase 2 now computes a lightweight 64-bit average perceptual hash (`ahash64`) for accepted images and exports cross-source near-duplicate candidates. This is a conservative review signal, not an automatic deletion rule. Evidence files: `artifacts/dataset/eda/near_duplicate_candidates.csv`, `near_duplicate_group_counts.csv`, and `near_duplicate_label_source_counts.csv`. Current audit found 196 candidate rows across 38 cross-source aHash groups after excluding `stanford-car-body-type-data`; these are low-priority optional review inputs and should be used in a pre/post comparison only if baseline results or leakage evidence justify the data loss.

## Final class counts after Stanford body-type exclusion

| Class | Final kullanılabilir veri |
|---|---:|
| SUV | 6,689 |
| VAN | 4,539 |
| STATION WAGON | 651 |
| MICRO | 206 |
| OPEN WHEEL / F1 | 5,846 |
| SEDAN | 8,351 |
| HATCHBACK | 2,651 |
| PICK UP | 2,679 |
