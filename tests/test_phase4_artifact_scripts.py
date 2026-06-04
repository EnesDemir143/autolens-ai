from pathlib import Path
import sys

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from autolens_ai.evaluation import expected_calibration_split
from scripts.generate_plots import parse_epoch_metrics_from_csv
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


def test_parse_epoch_metrics_from_csv_reads_epoch_level_metrics(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "epoch,train/loss_epoch,train/acc,val/loss,val/acc,val/f1_macro\n"
        ",,,,,\n"
        "0,1.20,,,,\n"
        "0,1.10,0.55,,,\n"
        "0,0.90,0.62,0.80,0.60,0.58\n"
        "1,0.70,0.72,0.60,0.74,0.71\n"
    )

    metrics = parse_epoch_metrics_from_csv(csv_path)

    assert metrics["epoch"] == [0, 1]
    assert metrics["train_loss"] == [0.9, 0.7]
    assert metrics["train_acc"] == [0.62, 0.72]
    assert metrics["val_loss"] == [0.8, 0.6]
    assert metrics["val_acc"] == [0.6, 0.74]
    assert metrics["val_f1_macro"] == [0.58, 0.71]
