# Phase 2 Validation Strategy

**Phase:** 2 — Dataset Research and Curation
**Created:** 2026-05-06
**Branch:** `feat/dataset-curation-eda`

## Required Evidence

- source catalog contains Kaggle and HF IDs/URLs.
- class mapping covers exactly 8 labels.
- manifest records counts and known gaps.
- dedup/split command exists.
- balance audit file exists.
- per-dataset CSV manifests exist under `artifacts/dataset/manifests/` for every inspected source.
- EDA outputs exist under `artifacts/dataset/eda/` for each candidate dataset and for the merged candidate pool.
- EDA includes class distributions, source-by-class coverage, missing/null label and metadata counts, missing referenced image-file counts, image dimensions/aspect ratios/file sizes, invalid/corrupt image counts, duplicate/hash results, and ambiguous-label counts.
- outlier/anomaly review artifacts exist and the final dataset decision note explains which sources/classes/images are accepted, filtered, excluded, or deferred before Phase 3 preprocessing/training. The note must preserve the unfiltered baseline and mark pre-outlier vs post-outlier comparison as low-priority optional work to run only if later metrics/leakage review justify the data loss.
