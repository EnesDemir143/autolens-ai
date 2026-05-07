# Phase 2 Dataset Decision Note

Generated: 2026-05-07T16:14:30.994544+00:00

## Execution decision

Internet source research/download was skipped because the user placed candidate datasets under `datasets/`. Phase 2 therefore scanned local source mirrors and produced reproducible manifests, EDA evidence, and split files.

## Dataset decision gate

- Excluded source for leakage risk: `stanford-car-body-type-data`; kept `stanford-car-dataset-by-classes-folder` as the single Stanford-derived source.
- Raw local images scanned: 56158
- Split-eligible unique accepted images: 31612
- Curated split policy: use all safe accepted unique images unless --max-per-class is set.
- Accepted images require explicit target-class labels and valid image files.
- Duplicate hashes, corrupt images, missing labels, generic trucks, City Car/MPV/Coupe/Convertible/Sports/Limousine/Car/Taxi ambiguity, and plain Cab labels are excluded or routed to manual review before training.
- Phase 3 may train from `artifacts/dataset/splits/*.csv`, but weak/gap classes must be considered in sampling or additional collection.

## Weak/gap classes
- SUV: 6689 eligible vs 4000 target; gap -2689.
- VAN: 4539 eligible vs 4000 target; gap -539.
- STATION WAGON: 651 eligible vs 4000 target; gap 3349.
- MICRO: 206 eligible vs 4000 target; gap 3794.
- OPEN WHEEL / F1: 5846 eligible vs 4000 target; gap -1846.
- SEDAN: 8351 eligible vs 4000 target; gap -4351.
- HATCHBACK: 2651 eligible vs 4000 target; gap 1349.
- PICK UP: 2679 eligible vs 4000 target; gap 1321.

## Evidence files

- Source catalog: `artifacts/dataset/source_catalog.csv`
- Per-source manifests: `artifacts/dataset/manifests/*.csv`
- Merged manifest: `artifacts/dataset/manifests/merged_manifest.csv`
- Class/source/status EDA: `artifacts/dataset/eda/*.csv`
- Model-name MICRO candidates: `artifacts/dataset/eda/micro_model_candidates.csv` and `micro_model_candidate_counts.csv`
- Perceptual-hash near-duplicate candidates: `artifacts/dataset/eda/near_duplicate_candidates.csv`
- Split manifests: `artifacts/dataset/splits/*.csv`

## Phase 3 preprocessing risks

- MICRO is populated only from the user-approved model whitelist (currently FIAT 500 and Smart Fortwo found locally); generic City Car remains excluded/review-only.
- STATION WAGON and PICK UP are below the 4k/class target and need class-weighted sampling or additional data.
- OPEN WHEEL / F1 is source-biased to F1 team folders; validation should monitor source/domain bias.
- Stanford-derived classes and cropped body-type images may differ in framing; augmentation/normalization should be chosen after inspecting EDA distributions.

## Stanford source exclusion

`stanford-car-body-type-data` is excluded from the final Phase 2 dataset to reduce overlap/leakage risk with the other Stanford-derived source. `stanford-car-dataset-by-classes-folder` remains as the single Stanford-derived source. The exclusion is recorded in `artifacts/dataset/excluded_source_catalog.csv`.

## Perceptual near-duplicate audit

Phase 2 now computes a lightweight 64-bit average perceptual hash (`ahash64`) for accepted images and exports cross-source near-duplicate candidates. This is a conservative review signal, not an automatic deletion rule. Evidence files: `artifacts/dataset/eda/near_duplicate_candidates.csv`, `near_duplicate_group_counts.csv`, and `near_duplicate_label_source_counts.csv`. Current audit found 196 candidate rows across 38 cross-source aHash groups after excluding `stanford-car-body-type-data`; these should be reviewed or used in the pre/post comparison gate before final training selection.

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

