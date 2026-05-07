"""Label normalization rules for the eight AutoLens body-type classes."""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_CLASSES: tuple[str, ...] = (
    "SUV",
    "VAN",
    "STATION WAGON",
    "MICRO",
    "OPEN WHEEL / F1",
    "SEDAN",
    "HATCHBACK",
    "PICK UP",
)

AMBIGUOUS_TERMS: tuple[str, ...] = (
    "city car",
    "crossover",
    "multi purpose vehicle",
    "mpv",
    "coupe",
    "convertible",
    "cabriolet",
    "sports",
    "limousine",
    "car",
    "taxi",
)


@dataclass(frozen=True)
class LabelDecision:
    """Result of mapping a raw label into the assignment class space."""

    normalized_label: str
    status: str
    reason: str


def _clean(text: str) -> str:
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_label(raw_label: str, source_id: str = "") -> LabelDecision:
    """Map a source label to a target class or return a review/exclusion reason.

    The rules intentionally avoid unsafe mappings called out in the Phase 2 plan:
    generic trucks are not pickups, city cars are not automatically MICRO, and
    coupe/convertible/MPV labels are not forced into nearby assignment classes.
    """

    label = _clean(raw_label)
    source = _clean(source_id)
    if not label:
        return LabelDecision("", "excluded", "missing_raw_label")

    if "f1" in label or "formula 1" in label or "formula one" in label or "f1" in source:
        return LabelDecision("OPEN WHEEL / F1", "accepted", "formula_one_source_or_label")

    if "station wagon" in label or re.search(r"\bwagon\b", label):
        return LabelDecision("STATION WAGON", "accepted", "explicit_wagon_label")

    if "hatchback" in label:
        return LabelDecision("HATCHBACK", "accepted", "explicit_hatchback_label")

    if re.search(r"\bsuv\b", label) or "sport utility vehicle" in label or "hummer" in label:
        return LabelDecision("SUV", "accepted", "explicit_suv_label")

    if "sedan" in label:
        return LabelDecision("SEDAN", "accepted", "explicit_sedan_label")

    if "minivan" in label or re.search(r"\bvan\b", label):
        return LabelDecision("VAN", "accepted", "explicit_van_or_minivan_label")

    if "pick up" in label or "pickup" in label:
        return LabelDecision("PICK UP", "accepted", "explicit_pickup_label")

    # Stanford class names encode pickup variants as cab styles. Plain "Cab" folders
    # remain review-only because they are too broad without the model name context.
    if any(term in label for term in ("crew cab", "extended cab", "regular cab", "club cab")):
        return LabelDecision("PICK UP", "accepted", "pickup_cab_style_label")

    if label == "cab":
        return LabelDecision("", "review", "plain_cab_needs_manual_pickup_review")

    if "truck" in label or "big truck" in label or "trucks" in label:
        return LabelDecision("", "excluded", "generic_truck_not_mapped_to_pickup")

    for term in AMBIGUOUS_TERMS:
        if term in label:
            return LabelDecision("", "review", f"ambiguous_{term.replace(' ', '_')}")

    return LabelDecision("", "excluded", "non_target_or_unknown_label")
