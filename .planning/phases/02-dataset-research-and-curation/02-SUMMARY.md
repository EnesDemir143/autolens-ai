# Phase 2 Execution Summary — Dataset Research and Curation

**Executed:** 2026-05-07  
**Mode:** `/gsd-execute-phase 2` equivalent, using local datasets supplied under `datasets/`  
**Internet research/download:** Skipped per user instruction because candidate datasets are already local.

## Completed plan coverage

- **02-01 Source catalog:** Created `artifacts/dataset/source_catalog.csv` and `.json` with local source IDs, paths, and license notes.
- **02-02 Ingestion scripts:** Added `autolens_ai.data.build_phase2_dataset` command to scan local images and emit one CSV manifest per source.
- **02-03 Label curation:** Added explicit 8-class label normalization rules in `autolens_ai.data.labels`; unsafe mappings are routed to review/exclusion.
- **02-04 Audit and splits:** Generated merged manifest, EDA tables, review/duplicate/missing lists, balance audit, decision note, and train/val/internal-test split CSV files.

## Key evidence

- Raw local images scanned: **64,302**
- Split-eligible unique accepted images: **31,612**
- Evidence root: `artifacts/dataset/`
- Dataset decision note: `artifacts/dataset/dataset_decision_note.md`
- Balance audit: `artifacts/dataset/eda/balance_audit.csv`
- Split manifests: `artifacts/dataset/splits/*.csv`

## Known gaps before Phase 3

- `MICRO`: 206 accepted images from the approved FIAT 500 / Smart Fortwo model whitelist; generic `City Car` remains excluded/review-only.
- `MICRO`, `STATION WAGON`, `HATCHBACK`, and `PICK UP` remain below the 4k/class target.
- F1 images are source-biased to Formula 1 team folders.
- Phase 3 should use class balancing/sampling and revisit weak-class collection before final model selection.
- Outlier handling is a low-priority optional comparison gate, not a silent deletion step: preserve the current pre-outlier dataset counts, and only if later metrics/leakage review justify the data loss create a second manual/outlier-filtered artifact version and compare class counts, source coverage, and Phase 3 metrics before deciding whether to use it.

## Extra MICRO model-name audit

Added model-name based MICRO candidate artifacts under `artifacts/dataset/eda/micro_model_candidates.csv` and `micro_model_candidate_counts.csv`. Final strict split includes only user-approved MICRO model whitelist matches as MICRO; local matches are FIAT 500 and Smart Fortwo.
### User-approved MICRO model search whitelist

The MICRO model-name audit is now restricted to the user-approved whitelist only: Smart Fortwo, other Smart micro models (Forfour, Roadster, Crossblade), Citroën Ami, Renault Twizy, Toyota/Scion iQ, Tata Nano, Aixam City, Ligier JS50, Peel P50, BMW Isetta, and Fiat 500/500e. Generic `City Car` is intentionally not included.

## Final MICRO policy

MICRO is not sourced from generic `City Car`. It is populated only from the user-approved model whitelist. In the current local datasets, the matched unique-valid MICRO models are FIAT 500 (126 images) and Smart Fortwo (80 images), for **206 MICRO images** total.

## Stanford source exclusion

`stanford-car-body-type-data` is excluded from the final Phase 2 dataset to reduce overlap/leakage risk with the other Stanford-derived source. `stanford-car-dataset-by-classes-folder` remains as the single Stanford-derived source. The exclusion is recorded in `artifacts/dataset/excluded_source_catalog.csv`.

## Perceptual near-duplicate audit

Phase 2 now computes a lightweight 64-bit average perceptual hash (`ahash64`) for accepted images and exports cross-source near-duplicate candidates. This is a conservative review signal, not an automatic deletion rule. Evidence files: `artifacts/dataset/eda/near_duplicate_candidates.csv`, `near_duplicate_group_counts.csv`, and `near_duplicate_label_source_counts.csv`. Current audit found 196 candidate rows across 38 cross-source aHash groups after excluding `stanford-car-body-type-data`; because weak classes have limited data, use these for low-priority optional review or a later pre/post comparison only if baseline results or leakage evidence make it worthwhile.

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
