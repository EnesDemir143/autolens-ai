"""Generate multi-class AUROC (ROC) curves for the active model.

Usage: uv run python scripts/generate_auroc.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from PIL import Image

from autolens_ai.inference import ONNXPredictor
from autolens_ai.data.labels import TARGET_CLASSES

DATASETS_DIR = Path("datasets")
SPLITS_DIR = Path("artifacts/dataset/splits")
OUTPUT_DIR = Path("artifacts/export/dinov3_safe_weighted_latest")


def main() -> None:
    csv_path = SPLITS_DIR / "internal_test.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} test samples")

    class_names = TARGET_CLASSES
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    print("Loading ONNX predictor...")
    predictor = ONNXPredictor.from_default_config()

    all_probs = []
    all_true = []

    for i, row in enumerate(rows):
        rel_path = row["relative_path"]
        img_path = DATASETS_DIR / rel_path
        true_label = row["normalized_label"]

        if not img_path.exists():
            print(f"  SKIP (missing): {rel_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            result = predictor.predict(img)
            probs_dict = result["probabilities"]
            probs = np.array([probs_dict[cls] for cls in class_names], dtype=np.float32)
            all_probs.append(probs)
            all_true.append(label_to_idx[true_label])
        except Exception as e:
            print(f"  ERROR {rel_path}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(rows)}")

    y_true = np.array(all_true)
    y_score = np.vstack(all_probs)

    print(f"\nInference complete: {len(y_true)} samples")
    print(f"Score matrix shape: {y_score.shape}")

    fpr: dict[int, np.ndarray] = {}
    tpr: dict[int, np.ndarray] = {}
    roc_auc: dict[int, float] = {}

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cls in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f"  {cls:20s} AUROC = {roc_auc[i]:.4f}")
        ax.plot(fpr[i], tpr[i], label=f"{cls} (AUC={roc_auc[i]:.3f})", linewidth=1.2)

    macro_auroc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    weighted_auroc = roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")

    print(f"\n  {'MACRO OVR':20s} AUROC = {macro_auroc:.4f}")
    print(f"  {'WEIGHTED OVR':20s} AUROC = {weighted_auroc:.4f}")

    y_true_oh = np.zeros_like(y_score)
    y_true_oh[np.arange(len(y_true)), y_true] = 1
    fpr_micro, tpr_micro, _ = roc_curve(y_true_oh.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    print(f"  {'MICRO OVR':20s} AUROC = {roc_auc_micro:.4f}")

    ax.plot(fpr_micro, tpr_micro, "k--", label=f"Micro-avg (AUC={roc_auc_micro:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "gray", linestyle=":", alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-class ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.3)

    output_path = OUTPUT_DIR / "roc_curves.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"\nROC plot saved to {output_path}")

    auroc_data = {
        "per_class": {cls: float(roc_auc[i]) for i, cls in enumerate(class_names)},
        "macro_ovr": float(macro_auroc),
        "weighted_ovr": float(weighted_auroc),
        "micro_ovr": float(roc_auc_micro),
    }
    json_path = OUTPUT_DIR / "auroc_scores.json"
    with open(json_path, "w") as f:
        json.dump(auroc_data, f, indent=2)
    print(f"AUROC scores saved to {json_path}")

    txt_path = OUTPUT_DIR / "per_class_auroc.txt"
    with open(txt_path, "w") as f:
        f.write(f"{'Class':20s} AUROC\n")
        f.write("-" * 30 + "\n")
        for cls in class_names:
            f.write(f"{cls:20s} {roc_auc[class_names.index(cls)]:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Macro OVR':20s} {macro_auroc:.4f}\n")
        f.write(f"{'Weighted OVR':20s} {weighted_auroc:.4f}\n")
        f.write(f"{'Micro OVR':20s} {roc_auc_micro:.4f}\n")
    print(f"Per-class AUROC saved to {txt_path}")


if __name__ == "__main__":
    main()
