# AutoLens AI — Dataset Documentation

## Overview

The AutoLens AI dataset is a custom-curated collection of vehicle images assembled from multiple public sources for the task of 8-class car body type classification. The dataset was created following strict curation policies to ensure label quality, minimize leakage, and reflect real-world distribution.

## Target Classes (8)

| Class | Description | Example Vehicles |
|---|---|---|
| **SUV** | Sport Utility Vehicle | Jeep Wrangler, Toyota RAV4, BMW X5 |
| **VAN** | Van / Minibus | Mercedes Sprinter, Ford Transit, Volkswagen Caravelle |
| **STATION WAGON** | Station Wagon / Estate | Volvo VW, Skoda Octavia Combi, Audi A4 Avant |
| **MICRO** | Microcar / City Car | Fiat 500, Smart Fortwo, Toyota Aygo |
| **OPEN WHEEL / F1** | Open-wheel / Formula-style | Formula 1 cars, go-karts, IndyCar |
| **SEDAN** | Sedan / Saloon | Toyota Camry, Honda Accord, BMW 3 Series |
| **HATCHBACK** | Hatchback | Volkswagen Golf, Ford Focus, Renault Clio |
| **PICK UP** | Pickup Truck | Ford F-150, Toyota Hilux, Chevrolet Silverado |

## Data Sources

The dataset combines images from multiple public sources. Each source was audited for licensing, label consistency, and potential overlap with internal test sets used for final evaluation.

| Source | Type | Notes |
|---|---|---|
| Stanford Car Dataset | Academic | Contains side-view car images; partially excluded to reduce leakage risk |
| CompCars | Multi-view | Includes vision, sketch, and attribute data |
| CarDM | Make/Model classification | Large-scale web-scraped dataset |
| COCO Car Subset | Object detection | Vehicle annotations from COCO |
| Custom Web Scrapes | Targeted | Specific makes/models from manufacturer sites |
| Kaggle Competitions | Various | Vehicle classification and attribute datasets |
| HuggingFace Datasets | Community | Publicly shared vehicle image collections |

All sources are documented in the source catalog: `artifacts/dataset/source_catalog.csv`

## Curation Policy

### Label Normalization
- Raw labels from sources were mapped to the 8 target classes using a many-to-one mapping.
- Ambiguous or unsafe mappings were rejected (e.g., "City Car" → MICRO is NOT allowed unless explicitly from an approved model whitelist).
- "Truck" → PICK UP only for actual pickup trucks; dump trucks, semi-trailers excluded.
- "Coupe", "Convertible", "MPV" routed to review; not automatically assigned.

### MICRO Class Specific Rules
- MICRO is populated **only** from a user-approved model whitelist:
  - Fiat 500/500e
  - Smart Fortwo / Forfour / Roadster / Crossblade
  - Citroën Ami
  - Renault Twizy
  - Toyota/Scion iQ
  - Tata Nano
  - Aixam City
  - Ligier JS50
  - Peel P50, P55, P56
  - BMW Isetta
  - Other confirmed microcars ≤1000cc engine displacement
- Generic "City Car" labels from sources are **not** used for MICRO.

### Quality Filters
- Images below 64×64 pixels discarded
- Corrupted/unreadable images removed
- Near-duplicate detection via perceptual hashing (aHash64) – groups flagged for review, not auto-deleted
- Watermarked or text-overlay images excluded when detectable

### Split Strategy
After curation, images were split stratified by class:
- **Train**: 80% (25,285 images)
- **Validation**: 10% (3,157 images) – used for calibration fitting and early stopping
- **Internal Test**: 10% (3,170 images) – held-out for final model selection, no tuning allowed

The split uses a fixed seed (42) for reproducibility. No image appears in more than one split.

## Evidence Artifacts

All curation decisions and intermediate outputs are preserved in `artifacts/dataset/`:

| Artifact | Purpose |
|---|---|
| `source_catalog.csv` | One row per source: name, URL, license, image count |
| `manifests/` | Per-source CSV manifests with metadata (path, original label, normalized label, review status) |
| `merged_manifest.csv` | Combined manifest of all accepted images |
| `splits/` | train.csv, val.csv, internal_test.csv, all_splits.csv (with SHA256 for verification) |
| `eda/` | Exploratory data analysis: class distribution, source stats, quality metrics |
| `dataset_decision_note.md` | Final curation rationale and policy decisions |
| `excluded_source_catalog.csv` | Sources excluded from final dataset and why |
| `phase2_metadata.json` | Processing timestamps, software versions, curator notes |

## Class Distribution (Final)

| Class | Train | Val | Internal Test | Total |
|---|---|---|---|---|
| SUV | 5,351 | 669 | 670 | 6,689 |
| VAN | 3,631 | 453 | 455 | 4,539 |
| STATION WAGON | 521 | 64 | 66 | 651 |
| MICRO | 165 | 21 | 22 | 206 |
| OPEN WHEEL / F1 | 4,677 | 583 | 586 | 5,846 |
| SEDAN | 6,681 | 834 | 836 | 8,351 |
| HATCHBACK | 2,121 | 264 | 266 | 2,651 |
| PICK UP | 2,143 | 265 | 269 | 2,679 |
| **Total** | **25,285** | **3,157** | **3,170** | **31,612** |

## Reproducibility

The entire dataset can be reproduced from the original sources using the provided scripts (see `scripts/create_splits.py` and `scripts/compute_dataset_stats.py`). However, due to licensing restrictions, the raw merged image set is **not** redistributable. Researchers should:

1. Download the source datasets using the links in `source_catalog.csv`
2. Run the Phase 2 curation pipeline to generate the splits and manifests
3. Use the manifest files to copy only accepted images into a local `datasets/` folder

This approach respects the original licenses while enabling full reproducibility of the training/evaluation splits.

## Usage in Training

The `AutolensAIDataModule` (PyTorch Lightning) reads the split CSV files and applies:
- Resize to 256×256 (DINOv3) or 224×224 (CNNs)
- RGB conversion
- Normalization with dataset-specific mean/std (computed from train split)
- Optional albumentations augmentations (horizontal shift, blur, etc.)

Mean/std values are cached in `artifacts/dataset/stats.json` after the first computation.