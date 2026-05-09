from pathlib import Path
import sys

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from autolens_ai.evaluation import expected_calibration_split
from scripts.checkpoint_to_safetensors import strip_lightning_prefix


def test_strip_lightning_prefix_keeps_only_model_weights() -> None:
    state = {
        "model.classifier.weight": torch.ones(2, 2),
        "_class_weights": torch.ones(2),
    }

    assert list(strip_lightning_prefix(state).keys()) == ["classifier.weight"]


def test_calibration_rejects_non_validation_split_names() -> None:
    with pytest.raises(ValueError):
        expected_calibration_split(Path("artifacts/dataset/splits/internal_test.csv"))

    expected_calibration_split(Path("artifacts/dataset/splits/val.csv"))
