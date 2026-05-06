# Phase 2 Research: Dataset Research and Curation
**Branch:** `feat/dataset-research-and-curation`

    ## Research Question

    What needs to be known to plan Phase 2 well?

    ## Source Research Notes

Kaggle candidates to validate/download:
- `ademboukhris/cars-body-type-cropped`: 7 body classes, approximately 7,000 images, CC0; useful for Hatchback, Pick-Up, SUV, Sedan, VAN; missing Station Wagon, Micro, Open Wheel/F1.
- `mayurmahurkar/stanford-car-body-type-data`: assignment-referenced Stanford body type source; exact accessible structure must be verified with Kaggle CLI.
- `jutrera/stanford-car-dataset-by-classes-folder`: 16,185 images / 196 make-model-year classes; can support body-style mapping only with metadata/manual mapping.

Hugging Face candidates to validate/download:
- `kitrofimov/cbsc`: Car Body Style Classification; 595 rows, 10 classes including hatchback, van, pickup, sedan, SUV.
- `DrBimmer/vehicle-classification`: imagefolder dataset, 2,733 rows, 16 labels, CC-BY-NC-4.0; may help with vehicle type coverage.

Critical research conclusion: no single source satisfies all 8 required labels. Phase 2 must include a class-gap audit and targeted collection strategy for Micro, Station Wagon, Open Wheel/F1, and clean Pick-Up. Final dataset target is 36k–40k raw candidates and about 32k clean near-balanced images (~4k/class). Ambiguous labels must be filtered: `City Car` is not automatically `MICRO`; generic `Truck` is not automatically `PICK_UP`; `Crossover`, `MPV/Minibus`, and `Coupe` are not directly mapped unless audited.


    ## Validation Architecture

    Phase 2 validation checks:
1. source catalog contains Kaggle and HF IDs/URLs.
2. class mapping covers exactly 8 labels.
3. manifest records counts, known gaps, and target-vs-actual counts for the ~4k/class goal.
4. dedup/split command exists.
5. balance audit file exists and flags ambiguous-label exclusions.
