from autolens_ai.data.labels import TARGET_CLASSES, normalize_label


def test_target_classes_are_exact_assignment_labels() -> None:
    assert TARGET_CLASSES == (
        "SUV",
        "VAN",
        "STATION WAGON",
        "MICRO",
        "OPEN WHEEL / F1",
        "SEDAN",
        "HATCHBACK",
        "PICK UP",
    )


def test_safe_label_mappings() -> None:
    assert normalize_label("BMW X5 SUV 2007").normalized_label == "SUV"
    assert normalize_label("Audi 100 Wagon 1994").normalized_label == "STATION WAGON"
    assert normalize_label("Ferrari F1 car", "f1-image-classification-updated").normalized_label == "OPEN WHEEL / F1"
    assert normalize_label("Chevrolet Silverado 1500 Crew Cab 2012").normalized_label == "PICK UP"


def test_ambiguous_labels_are_not_forced_into_target_classes() -> None:
    assert normalize_label("City Car").status == "review"
    assert normalize_label("Truck").status == "excluded"
    assert normalize_label("Coupe").status == "review"
    assert normalize_label("MPV").status == "review"


def test_vehicle_metadata_csv_with_bom_is_loaded() -> None:
    from pathlib import Path

    from autolens_ai.data.build_phase2_dataset import load_vehicle_classification_labels

    labels = load_vehicle_classification_labels(Path("datasets/vehicle-classification-dataset"))
    if labels:  # local dataset is gitignored, so keep the package test portable
        assert labels["4841.jpg"] == "Pickup"
        assert labels["2829.jpg"] == "SEDAN"
