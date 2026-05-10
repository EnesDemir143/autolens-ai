"""Build Phase 2 dataset manifests, EDA tables, and deterministic splits.

This command scans local files under ``datasets/``. It does not download data and
therefore supports the user-provided local dataset workflow for Phase 2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, UnidentifiedImageError

from autolens_ai.data.labels import TARGET_CLASSES, normalize_label

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIFEST_COLUMNS = [
    "source_id",
    "relative_path",
    "original_path_or_url",
    "source_split_hint",
    "raw_label",
    "normalized_label",
    "mapping_status",
    "review_or_exclusion_reason",
    "split_eligible",
    "width",
    "height",
    "aspect_ratio",
    "file_size_bytes",
    "sha256",
    "ahash64",
    "near_duplicate_group",
    "near_duplicate_group_size",
    "near_duplicate_sources",
    "duplicate_of",
    "license_note",
]

SOURCE_LICENSE_NOTES = {
    "DrBimmer-vehicle-classification": "Local Kaggle-style mirror; verify Kaggle page/license before publication.",
    "cars-body-type-cropped": "Local Kaggle-style mirror; verify Kaggle page/license before publication.",
    "cbsc": "Local Hugging Face/Kaggle-style mirror with README; verify original license before publication.",
    "f1-image-classification-updated": "Local Kaggle-style F1 image mirror; verify source license before publication.",
    "stanford-car-body-type-data": "Derived from Stanford Cars metadata/body-type folders; research/education dataset lineage.",
    "stanford-car-dataset-by-classes-folder": "Stanford Cars class-folder mirror; research/education dataset lineage.",
    "vehicle-classification-dataset": "Mixed local vehicle classification mirror; unlabeled train folder is excluded until metadata is supplied.",
    "vehicle-images-dataset": "Local vehicle image mirror; ambiguous City Car/MPV/Truck labels require review or exclusion.",
    "vehicle-type-10-competition": "Local 10-class vehicle competition mirror; sample_submission supplies test labels.",
}

# Model-name filter for MICRO candidates. These names are checked against source
# model labels and filenames, not against generic body-type folder labels alone.
# `strong` candidates are small city/micro models close to the assignment example;
# `review` candidates are small cars that need human confirmation before relabeling.
MICRO_MODEL_PATTERNS: tuple[tuple[str, str, str], ...] = (
    # User-approved MICRO model whitelist. Do not add generic `City Car`.
    ("Smart Fortwo", "smart fortwo", "strong"),
    ("Smart Forfour", "smart forfour", "strong"),
    ("Smart Roadster", "smart roadster", "strong"),
    ("Smart Crossblade", "smart crossblade", "strong"),
    ("Citroen Ami", "citroen ami", "strong"),
    ("Renault Twizy", "renault twizy", "strong"),
    ("Toyota iQ", "toyota iq", "strong"),
    ("Scion iQ", "scion iq", "strong"),
    ("Tata Nano", "tata nano", "strong"),
    ("Aixam City", "aixam city", "strong"),
    ("Ligier JS50", "ligier js50", "strong"),
    ("Peel P50", "peel p50", "strong"),
    ("BMW Isetta", "bmw isetta", "strong"),
    ("FIAT 500", "fiat 500", "strong"),
    ("FIAT 500e", "fiat 500e", "strong"),
)

EXCLUDED_SOURCE_IDS: set[str] = {
    # Excluded to avoid Stanford-derived source overlap/leakage risk.
    "stanford-car-body-type-data",
}


@dataclass(frozen=True)
class ImageRecord:
    source_id: str
    relative_path: str
    original_path_or_url: str
    source_split_hint: str
    raw_label: str
    normalized_label: str
    mapping_status: str
    review_or_exclusion_reason: str
    split_eligible: str
    width: str
    height: str
    aspect_ratio: str
    file_size_bytes: str
    sha256: str
    ahash64: str
    near_duplicate_group: str
    near_duplicate_group_size: str
    near_duplicate_sources: str
    duplicate_of: str
    license_note: str


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def infer_split(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "train" in parts:
        return "source_train"
    if "valid" in parts or "validation" in parts:
        return "source_validation"
    if "test" in parts:
        return "source_test"
    return "source_unspecified"


def infer_raw_label(source_root: Path, image_path: Path, sample_labels: dict[str, str]) -> str:
    rel_parts = image_path.relative_to(source_root).parts
    if image_path.name in sample_labels:
        return sample_labels[image_path.name]

    # Known unlabeled bulk folder: keep rows, but exclude until metadata is supplied.
    rel_text = "/".join(rel_parts).lower()
    if "car-classification-data-with-labels-ihasib/train" in rel_text:
        return ""

    parent = image_path.parent.name
    if parent.lower() in {"train", "test", "valid", "validation"}:
        return ""
    return parent


def load_sample_submission(source_root: Path) -> dict[str, str]:
    csv_path = source_root / "sample_submission.csv"
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return {row["image_name"]: row["ground_truth"] for row in csv.DictReader(handle)}


def load_vehicle_classification_labels(source_root: Path) -> dict[str, str]:
    csv_path = source_root / "Car-Classification-Data-with-Labels-iHasib/type-labels.csv"
    if not csv_path.exists():
        return {}
    labels: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image_id = row.get("ID", "").strip()
            label = row.get("Type", "").strip()
            if image_id and label and image_id.lower() != "id":
                labels[f"{image_id}.jpg"] = label
    return labels


def image_facts(path: Path) -> tuple[str, str, str, str, str, str, str]:
    size = path.stat().st_size
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with Image.open(path) as image:
            width, height = image.size
            ahash = average_hash64(image)
    except (OSError, UnidentifiedImageError):
        return "", "", "", str(size), digest, "", "invalid_or_corrupt_image"
    ratio = round(width / height, 4) if height else 0
    return str(width), str(height), str(ratio), str(size), digest, ahash, ""


def average_hash64(image: Image.Image) -> str:
    gray = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())  # Pillow <14 compatible pixel access
    mean = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | int(pixel >= mean)
    return f"{bits:016x}"


def scan_source(
    source_root: Path, dataset_root: Path, seen_hashes: dict[str, str]
) -> list[ImageRecord]:
    source_id = source_root.name
    sample_labels = load_sample_submission(source_root)
    metadata_labels = load_vehicle_classification_labels(source_root)
    rows: list[ImageRecord] = []
    for image_path in iter_images(source_root):
        raw_label = metadata_labels.get(image_path.name) or infer_raw_label(
            source_root, image_path, sample_labels
        )
        decision = normalize_label(raw_label, source_id)
        width, height, ratio, size, digest, ahash, image_error = image_facts(image_path)
        rel = image_path.relative_to(dataset_root).as_posix()
        duplicate_of = seen_hashes.get(digest, "") if digest else ""
        if digest and not duplicate_of:
            seen_hashes[digest] = rel
        status = decision.status
        reason = decision.reason
        if image_error:
            status = "excluded"
            reason = image_error
        elif duplicate_of:
            status = "excluded"
            reason = "duplicate_image_hash"

        normalized_label = decision.normalized_label if status == "accepted" else ""
        approved_micro_model = approved_micro_model_name(
            ImageRecord(
                source_id=source_id,
                relative_path=rel,
                original_path_or_url=rel,
                source_split_hint=infer_split(image_path),
                raw_label=raw_label,
                normalized_label=normalized_label,
                mapping_status=status,
                review_or_exclusion_reason=reason,
                split_eligible="no",
                width=width,
                height=height,
                aspect_ratio=ratio,
                file_size_bytes=size,
                sha256=digest,
                ahash64=ahash,
                near_duplicate_group="",
                near_duplicate_group_size="",
                near_duplicate_sources="",
                duplicate_of=duplicate_of,
                license_note=SOURCE_LICENSE_NOTES.get(
                    source_id, "Local user-provided dataset; verify source/license."
                ),
            )
        )
        if approved_micro_model and not image_error and not duplicate_of:
            status = "accepted"
            reason = f"approved_micro_model:{approved_micro_model}"
            normalized_label = "MICRO"

        split_eligible = (
            "yes" if status == "accepted" and normalized_label in TARGET_CLASSES else "no"
        )
        rows.append(
            ImageRecord(
                source_id=source_id,
                relative_path=rel,
                original_path_or_url=rel,
                source_split_hint=infer_split(image_path),
                raw_label=raw_label,
                normalized_label=normalized_label,
                mapping_status=status,
                review_or_exclusion_reason=reason,
                split_eligible=split_eligible,
                width=width,
                height=height,
                aspect_ratio=ratio,
                file_size_bytes=size,
                sha256=digest,
                ahash64=ahash,
                near_duplicate_group="",
                near_duplicate_group_size="",
                near_duplicate_sources="",
                duplicate_of=duplicate_of,
                license_note=SOURCE_LICENSE_NOTES.get(
                    source_id, "Local user-provided dataset; verify source/license."
                ),
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_counts(path: Path, counts: Mapping[tuple[str, ...], int], columns: list[str]) -> None:
    rows = [
        {**{columns[i]: key[i] for i in range(len(key))}, "count": str(value)}
        for key, value in sorted(counts.items())
    ]
    write_csv(path, rows, [*columns, "count"])


def _searchable_text(row: ImageRecord) -> str:
    return f"{row.raw_label} {row.relative_path}".lower().replace("-", " ").replace("_", " ")


def micro_model_matches(row: ImageRecord) -> list[tuple[str, str]]:
    text = _searchable_text(row)
    matches: list[tuple[str, str]] = []
    for model_name, pattern, tier in MICRO_MODEL_PATTERNS:
        if pattern in text:
            matches.append((model_name, tier))
    return matches


def approved_micro_model_name(row: ImageRecord) -> str:
    for model_name, tier in micro_model_matches(row):
        if tier == "strong":
            return model_name
    return ""


def write_micro_model_candidates(rows: list[ImageRecord], output_dir: Path) -> None:
    candidate_rows: list[dict[str, str]] = []
    for row in rows:
        matches = micro_model_matches(row)
        if not matches:
            continue
        is_duplicate = row.review_or_exclusion_reason == "duplicate_image_hash"
        is_valid_unique = row.width != "" and not is_duplicate
        for model_name, tier in matches:
            candidate_rows.append(
                {
                    **asdict(row),
                    "micro_model_name": model_name,
                    "micro_candidate_tier": tier,
                    "micro_candidate_unique_valid": "yes" if is_valid_unique else "no",
                    "micro_candidate_note": (
                        "model_name_filter_candidate_for_manual_micro_relabel"
                        if tier == "strong"
                        else "small_car_review_only_not_auto_micro"
                    ),
                }
            )

    columns = [
        *MANIFEST_COLUMNS,
        "micro_model_name",
        "micro_candidate_tier",
        "micro_candidate_unique_valid",
        "micro_candidate_note",
    ]
    write_csv(output_dir / "eda/micro_model_candidates.csv", candidate_rows, columns)

    counts: Counter[tuple[str, ...]] = Counter()
    for candidate_row in candidate_rows:
        counts[
            (
                candidate_row["micro_model_name"],
                candidate_row["micro_candidate_tier"],
                candidate_row["micro_candidate_unique_valid"],
            )
        ] += 1
    write_counts(
        output_dir / "eda/micro_model_candidate_counts.csv",
        counts,
        ["micro_model_name", "micro_candidate_tier", "micro_candidate_unique_valid"],
    )


def summarize_numeric(rows: list[ImageRecord], field: str) -> dict[str, str]:
    values = [float(getattr(row, field)) for row in rows if getattr(row, field)]
    if not values:
        return {"field": field, "count": "0", "min": "", "max": "", "mean": ""}
    return {
        "field": field,
        "count": str(len(values)),
        "min": str(round(min(values), 4)),
        "max": str(round(max(values), 4)),
        "mean": str(round(sum(values) / len(values), 4)),
    }


def annotate_near_duplicates(rows: list[ImageRecord]) -> list[ImageRecord]:
    by_ahash: dict[str, list[ImageRecord]] = defaultdict(list)
    for row in rows:
        if row.ahash64 and row.split_eligible == "yes":
            by_ahash[row.ahash64].append(row)

    group_metadata: dict[str, tuple[str, str, str]] = {}
    group_index = 1
    for ahash, group_rows in sorted(by_ahash.items()):
        unique_paths = {row.relative_path for row in group_rows}
        if len(unique_paths) <= 1:
            continue
        sources = sorted({row.source_id for row in group_rows})
        if len(sources) <= 1:
            continue
        group_id = f"ahash-{group_index:05d}"
        group_index += 1
        group_metadata[ahash] = (group_id, str(len(unique_paths)), "|".join(sources))

    annotated: list[ImageRecord] = []
    for row in rows:
        group_id, group_size, group_sources = group_metadata.get(row.ahash64, ("", "", ""))
        annotated.append(
            ImageRecord(
                source_id=row.source_id,
                relative_path=row.relative_path,
                original_path_or_url=row.original_path_or_url,
                source_split_hint=row.source_split_hint,
                raw_label=row.raw_label,
                normalized_label=row.normalized_label,
                mapping_status=row.mapping_status,
                review_or_exclusion_reason=row.review_or_exclusion_reason,
                split_eligible=row.split_eligible,
                width=row.width,
                height=row.height,
                aspect_ratio=row.aspect_ratio,
                file_size_bytes=row.file_size_bytes,
                sha256=row.sha256,
                ahash64=row.ahash64,
                near_duplicate_group=group_id,
                near_duplicate_group_size=group_size,
                near_duplicate_sources=group_sources,
                duplicate_of=row.duplicate_of,
                license_note=row.license_note,
            )
        )
    return annotated


def write_near_duplicate_reports(rows: list[ImageRecord], output_dir: Path) -> None:
    near_rows = [asdict(row) for row in rows if row.near_duplicate_group]
    write_csv(output_dir / "eda/near_duplicate_candidates.csv", near_rows, MANIFEST_COLUMNS)

    counts: Counter[tuple[str, ...]] = Counter()
    label_counts: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        if not row.near_duplicate_group:
            continue
        counts[
            (row.near_duplicate_group, row.near_duplicate_group_size, row.near_duplicate_sources)
        ] += 1
        label_counts[(row.normalized_label, row.near_duplicate_sources)] += 1
    write_counts(
        output_dir / "eda/near_duplicate_group_counts.csv",
        counts,
        ["near_duplicate_group", "near_duplicate_group_size", "near_duplicate_sources"],
    )
    write_counts(
        output_dir / "eda/near_duplicate_label_source_counts.csv",
        label_counts,
        ["normalized_label", "near_duplicate_sources"],
    )


def build_splits(
    rows: list[ImageRecord], output_dir: Path, max_per_class: int | None = None
) -> dict[str, Counter[str]]:
    by_class: dict[str, list[ImageRecord]] = defaultdict(list)
    for row in rows:
        if row.split_eligible == "yes":
            by_class[row.normalized_label].append(row)

    split_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "internal_test": Counter(),
    }
    split_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    split_columns = ["split", *MANIFEST_COLUMNS]

    for label in TARGET_CLASSES:
        class_rows = sorted(by_class[label], key=lambda item: (item.sha256, item.relative_path))
        if max_per_class is not None:
            class_rows = class_rows[:max_per_class]
        n = len(class_rows)
        train_end = int(n * 0.8)
        val_end = train_end + int(n * 0.1)
        for index, row in enumerate(class_rows):
            split = "train" if index < train_end else "val" if index < val_end else "internal_test"
            split_counts[split][label] += 1
            split_rows[split].append({"split": split, **asdict(row)})

    for split, records in split_rows.items():
        write_csv(output_dir / f"splits/{split}.csv", records, split_columns)
    all_split_rows = [
        record for split in ("train", "val", "internal_test") for record in split_rows[split]
    ]
    write_csv(output_dir / "splits/all_splits.csv", all_split_rows, split_columns)
    return split_counts


def write_catalog(dataset_root: Path, source_dirs: list[Path], output_dir: Path) -> None:
    rows = []
    for source_dir in source_dirs:
        source_id = source_dir.name
        rows.append(
            {
                "source_id": source_id,
                "local_path": source_dir.as_posix(),
                "source_family": "local_user_provided",
                "source_url_or_id": "local datasets/ mirror; internet research skipped per user instruction",
                "license_note": SOURCE_LICENSE_NOTES.get(
                    source_id, "Verify original source/license before publication."
                ),
                "phase2_use": "scan_local_manifest_then_eda_decision_gate",
            }
        )
    write_csv(
        output_dir / "source_catalog.csv",
        rows,
        [
            "source_id",
            "local_path",
            "source_family",
            "source_url_or_id",
            "license_note",
            "phase2_use",
        ],
    )
    (output_dir / "source_catalog.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def run(dataset_root: Path, output_dir: Path, max_per_class: int | None = None) -> None:
    all_source_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    source_dirs = [path for path in all_source_dirs if path.name not in EXCLUDED_SOURCE_IDS]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests").mkdir(exist_ok=True)
    (output_dir / "eda").mkdir(exist_ok=True)
    write_catalog(dataset_root, source_dirs, output_dir)
    excluded_rows = [
        {
            "source_id": path.name,
            "local_path": path.as_posix(),
            "exclusion_reason": "excluded_to_reduce_stanford_source_overlap_leakage_risk",
        }
        for path in all_source_dirs
        if path.name in EXCLUDED_SOURCE_IDS
    ]
    write_csv(
        output_dir / "excluded_source_catalog.csv",
        excluded_rows,
        ["source_id", "local_path", "exclusion_reason"],
    )

    all_rows: list[ImageRecord] = []
    seen_hashes: dict[str, str] = {}
    for source_dir in source_dirs:
        rows = scan_source(source_dir, dataset_root, seen_hashes)
        all_rows.extend(rows)

    all_rows = annotate_near_duplicates(all_rows)
    for source_dir in source_dirs:
        source_rows = [row for row in all_rows if row.source_id == source_dir.name]
        write_csv(
            output_dir / f"manifests/{source_dir.name}.csv",
            [asdict(row) for row in source_rows],
            MANIFEST_COLUMNS,
        )

    write_csv(
        output_dir / "manifests/merged_manifest.csv",
        [asdict(row) for row in all_rows],
        MANIFEST_COLUMNS,
    )

    write_counts(
        output_dir / "eda/class_distribution.csv",
        Counter((row.normalized_label or "UNMAPPED",) for row in all_rows),
        ["normalized_label"],
    )
    write_counts(
        output_dir / "eda/status_counts.csv",
        Counter((row.mapping_status, row.review_or_exclusion_reason) for row in all_rows),
        ["mapping_status", "reason"],
    )
    write_counts(
        output_dir / "eda/source_by_class.csv",
        Counter((row.source_id, row.normalized_label or "UNMAPPED") for row in all_rows),
        ["source_id", "normalized_label"],
    )
    write_counts(
        output_dir / "eda/source_status_counts.csv",
        Counter(
            (row.source_id, row.mapping_status, row.review_or_exclusion_reason) for row in all_rows
        ),
        ["source_id", "mapping_status", "reason"],
    )
    write_csv(
        output_dir / "eda/image_quality_summary.csv",
        [
            summarize_numeric(all_rows, field)
            for field in ("width", "height", "aspect_ratio", "file_size_bytes")
        ],
        ["field", "count", "min", "max", "mean"],
    )
    write_micro_model_candidates(all_rows, output_dir)

    review_rows = [asdict(row) for row in all_rows if row.mapping_status == "review"]
    duplicate_rows = [
        asdict(row) for row in all_rows if row.review_or_exclusion_reason == "duplicate_image_hash"
    ]
    missing_rows = [
        asdict(row)
        for row in all_rows
        if row.review_or_exclusion_reason in {"missing_raw_label", "invalid_or_corrupt_image"}
    ]
    write_csv(output_dir / "eda/review_candidates.csv", review_rows, MANIFEST_COLUMNS)
    write_csv(output_dir / "eda/duplicate_hashes.csv", duplicate_rows, MANIFEST_COLUMNS)
    write_csv(output_dir / "eda/missing_or_corrupt.csv", missing_rows, MANIFEST_COLUMNS)
    write_near_duplicate_reports(all_rows, output_dir)

    split_counts = build_splits(all_rows, output_dir, max_per_class=max_per_class)
    accepted_counts = Counter(
        row.normalized_label for row in all_rows if row.split_eligible == "yes"
    )
    target_rows = [
        {
            "class": label,
            "target_clean_count": "4000",
            "eligible_unique_count": str(accepted_counts[label]),
            "gap_to_target": str(4000 - accepted_counts[label]),
            "train_count": str(split_counts["train"][label]),
            "val_count": str(split_counts["val"][label]),
            "internal_test_count": str(split_counts["internal_test"][label]),
        }
        for label in TARGET_CLASSES
    ]
    write_csv(
        output_dir / "eda/balance_audit.csv",
        target_rows,
        [
            "class",
            "target_clean_count",
            "eligible_unique_count",
            "gap_to_target",
            "train_count",
            "val_count",
            "internal_test_count",
        ],
    )

    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_root": dataset_root.as_posix(),
        "total_images_scanned": len(all_rows),
        "target_classes": TARGET_CLASSES,
        "internet_research": "skipped; user provided datasets under datasets/",
        "manifests": "artifacts/dataset/manifests/",
        "eda": "artifacts/dataset/eda/",
        "splits": "artifacts/dataset/splits/",
        "max_per_class": max_per_class,
        "split_policy": "use all safe accepted unique images unless --max-per-class is set",
    }
    (output_dir / "phase2_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    accepted_total = sum(accepted_counts.values())
    decision = [
        "# Phase 2 Dataset Decision Note",
        "",
        f"Generated: {metadata['generated_at']}",
        "",
        "## Execution decision",
        "",
        "Internet source research/download was skipped because the user placed candidate datasets under `datasets/`. Phase 2 therefore scanned local source mirrors and produced reproducible manifests, EDA evidence, and split files.",
        "",
        "## Dataset decision gate",
        "",
        "- Excluded source for leakage risk: `stanford-car-body-type-data`; kept `stanford-car-dataset-by-classes-folder` as the single Stanford-derived source.",
        f"- Raw local images scanned: {len(all_rows)}",
        f"- Split-eligible unique accepted images: {accepted_total}",
        f"- Curated split policy: {metadata['split_policy']}.",
        "- Accepted images require explicit target-class labels and valid image files.",
        "- Duplicate hashes, corrupt images, missing labels, generic trucks, City Car/MPV/Coupe/Convertible/Sports/Limousine/Car/Taxi ambiguity, and plain Cab labels are excluded or routed to manual review before training.",
        "- Phase 3 may train from `artifacts/dataset/splits/*.csv`, but weak/gap classes must be considered in sampling or additional collection.",
        "",
        "## Weak/gap classes",
    ]
    for row in target_rows:
        decision.append(
            f"- {row['class']}: {row['eligible_unique_count']} eligible vs 4000 target; gap {row['gap_to_target']}."
        )
    decision.extend(
        [
            "",
            "## Evidence files",
            "",
            "- Source catalog: `artifacts/dataset/source_catalog.csv`",
            "- Per-source manifests: `artifacts/dataset/manifests/*.csv`",
            "- Merged manifest: `artifacts/dataset/manifests/merged_manifest.csv`",
            "- Class/source/status EDA: `artifacts/dataset/eda/*.csv`",
            "- Model-name MICRO candidates: `artifacts/dataset/eda/micro_model_candidates.csv` and `micro_model_candidate_counts.csv`",
            "- Perceptual-hash near-duplicate candidates: `artifacts/dataset/eda/near_duplicate_candidates.csv`",
            "- Split manifests: `artifacts/dataset/splits/*.csv`",
            "",
            "## Phase 3 preprocessing risks",
            "",
            "- MICRO is populated only from the user-approved model whitelist (currently FIAT 500 and Smart Fortwo found locally); generic City Car remains excluded/review-only.",
            "- STATION WAGON and PICK UP are below the 4k/class target and need class-weighted sampling or additional data.",
            "- OPEN WHEEL / F1 is source-biased to F1 team folders; validation should monitor source/domain bias.",
            "- Stanford-derived classes and cropped body-type images may differ in framing; augmentation/normalization should be chosen after inspecting EDA distributions.",
        ]
    )
    (output_dir / "dataset_decision_note.md").write_text(
        "\n".join(decision) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dataset"))
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional cap per class. Default 0 keeps all safe accepted images.",
    )
    args = parser.parse_args()
    max_per_class = args.max_per_class if args.max_per_class > 0 else None
    run(args.dataset_root, args.output_dir, max_per_class=max_per_class)


if __name__ == "__main__":
    main()
