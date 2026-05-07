# AutoLens AI Dataset Manifest — Phase 2

Phase 2 uses local user-provided datasets under `datasets/`; internet download/research was intentionally skipped for execution because the candidate datasets are already present locally.

## Target classes

Exactly eight assignment labels are supported:

1. SUV
2. VAN
3. STATION WAGON
4. MICRO
5. OPEN WHEEL / F1
6. SEDAN
7. HATCHBACK
8. PICK UP

## Reproducible command

```bash
uv run python -m autolens_ai.data.build_phase2_dataset --dataset-root datasets --output-dir artifacts/dataset
```

## Manifest schema

Per-source manifests are written to `artifacts/dataset/manifests/`. Each row contains:

- `source_id`
- `relative_path`
- `original_path_or_url`
- `source_split_hint`
- `raw_label`
- `normalized_label`
- `mapping_status`
- `review_or_exclusion_reason`
- `split_eligible`
- `width`
- `height`
- `aspect_ratio`
- `file_size_bytes`
- `sha256`
- `ahash64`
- `near_duplicate_group`
- `near_duplicate_group_size`
- `near_duplicate_sources`
- `duplicate_of`
- `license_note`

## Evidence outputs

- Source catalog: `artifacts/dataset/source_catalog.csv`
- Per-source manifests: `artifacts/dataset/manifests/*.csv`
- Merged manifest: `artifacts/dataset/manifests/merged_manifest.csv`
- EDA tables: `artifacts/dataset/eda/*.csv`
- Split manifests: `artifacts/dataset/splits/*.csv`
- Dataset decision gate: `artifacts/dataset/dataset_decision_note.md`

## Curation policy

Unsafe label mappings are not forced. `City Car` is not automatically MICRO, generic `Truck` is not PICK UP, and `Crossover`/`MPV`/`Coupe`/`Convertible` labels are routed to review or exclusion. Phase 3 must use the split files plus this decision note rather than raw folders directly. Outlier removal must preserve a pre-outlier baseline and compare it against the post-outlier dataset before changing the training manifest.

## Model-name MICRO candidate artifact

Some local sources include model names in CSV labels, folder names, or filenames. Phase 2 now exports `artifacts/dataset/eda/micro_model_candidates.csv` and `micro_model_candidate_counts.csv` for micro-style model filtering. Final strict split includes only user-approved MICRO model whitelist matches as MICRO; local matches are FIAT 500 and Smart Fortwo.
### User-approved MICRO model search whitelist

The MICRO model-name audit is now restricted to the user-approved whitelist only: Smart Fortwo, other Smart micro models (Forfour, Roadster, Crossblade), Citroën Ami, Renault Twizy, Toyota/Scion iQ, Tata Nano, Aixam City, Ligier JS50, Peel P50, BMW Isetta, and Fiat 500/500e. Generic `City Car` is intentionally not included.

## Final MICRO policy

MICRO is not sourced from generic `City Car`. It is populated only from the user-approved model whitelist. In the current local datasets, the matched unique-valid MICRO models are FIAT 500 (126 images) and Smart Fortwo (80 images), for **206 MICRO images** total.

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

