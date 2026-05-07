# Phase 2 Verification — Dataset Research and Curation

**Verified:** 2026-05-07

## Required evidence checklist

- [x] Source catalog contains local source IDs/paths and license notes: `artifacts/dataset/source_catalog.csv`.
- [x] Class mapping covers exactly 8 labels: `src/autolens_ai/data/labels.py`.
- [x] Manifest records source, mapping, counts inputs, hash, quality fields, and known gaps: `artifacts/dataset/manifests/merged_manifest.csv`.
- [x] Dedup/split command exists: `uv run python -m autolens_ai.data.build_phase2_dataset --dataset-root datasets --output-dir artifacts/dataset`.
- [x] Balance audit exists: `artifacts/dataset/eda/balance_audit.csv`.
- [x] Per-dataset CSV manifests exist under `artifacts/dataset/manifests/` for every inspected source.
- [x] EDA outputs exist under `artifacts/dataset/eda/`.
- [x] EDA includes class distributions, source-by-class coverage, status/missing/corrupt/duplicate/review tables, and image quality summaries.
- [x] Outlier/anomaly/manual review artifacts exist: `artifacts/dataset/eda/review_candidates.csv`.
- [x] Dataset decision note records accepted/filter/excluded/deferred policy before Phase 3: `artifacts/dataset/dataset_decision_note.md`.

## Validation commands

```bash
uv run python -m autolens_ai.data.build_phase2_dataset --dataset-root datasets --output-dir artifacts/dataset
uv run pytest
uv run ruff check src tests
uv run mypy src
```

## Result

Phase 2 artifacts were generated and code quality gates passed. The phase is usable for Phase 3, with documented data gaps for MICRO, STATION WAGON, HATCHBACK, and PICK UP. Outlier removal must be evaluated as a before/after comparison, not applied silently.

## MICRO model-name audit evidence

- [x] Model-name MICRO candidate list exists: `artifacts/dataset/eda/micro_model_candidates.csv`.
- [x] Model-name candidate counts exist: `artifacts/dataset/eda/micro_model_candidate_counts.csv`.
- [x] Strong vs review-only candidates are separated so strict labels are not silently changed.

## Final MICRO policy

MICRO is not sourced from generic `City Car`. It is populated only from the user-approved model whitelist. In the current local datasets, the matched unique-valid MICRO models are FIAT 500 (126 images) and Smart Fortwo (80 images), for **206 MICRO images** total.

## Near-duplicate evidence

- [x] Perceptual hash column exists in manifests: `ahash64`.
- [x] Cross-source near-duplicate candidates exist: `artifacts/dataset/eda/near_duplicate_candidates.csv`.
- [x] Near-duplicate group/count summaries exist for pre/post comparison before final training selection.
