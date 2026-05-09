"""Run inference on internal test set and evaluate metrics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from autolens_ai.inference.artifact import load_artifact_metadata, load_model_from_artifact


def preprocess_image(image_path: Path, metadata: dict) -> torch.Tensor:
    """Preprocess a single image using artifact metadata."""
    from torchvision import transforms
    
    preprocess = metadata["preprocessing"]
    
    transform = transforms.Compose([
        transforms.Resize(preprocess["resize_size"]),
        transforms.CenterCrop(preprocess["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=preprocess["mean"], std=preprocess["std"]),
    ])
    
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def main():
    artifact_dir = Path("artifacts/export/efficientnet_b2_latest")
    test_csv = Path("artifacts/dataset/splits/internal_test.csv")
    data_root = Path("datasets")
    
    # Load metadata
    metadata = load_artifact_metadata(artifact_dir / "metadata.json")
    class_names = metadata["classes"]["labels"]
    
    # Load model from safetensors
    print("Loading model...")
    model = load_model_from_artifact(
        safetensors_path=artifact_dir / "model.safetensors",
        metadata_path=artifact_dir / "metadata.json",
    )
    model.eval()
    
    # Load test CSV
    test_df = pd.read_csv(test_csv)
    print(f"Internal test samples: {len(test_df)}")
    
    # Run inference
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Map label strings to indices
    class_to_idx = metadata["classes"]["class_to_idx"]
    
    print("Running inference...")
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            img_path = data_root / row["relative_path"]
            if not img_path.exists():
                continue
            
            try:
                img_tensor = preprocess_image(img_path, metadata).to(device)
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                
                all_preds.append(pred)
                all_labels.append(class_to_idx[row["normalized_label"]])
                all_probs.append(probs[0].cpu().numpy())
            except Exception as e:
                print(f"Error on {img_path}: {e}")
                continue
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"\n=== Internal Test Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Per-class metrics
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(8)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    
    print("\nNormalized Confusion Matrix:")
    print(np.round(cm_norm, 4))
    
    # Save results
    results = {
        "split": "internal_test",
        "num_samples": len(all_preds),
        "accuracy": float(accuracy),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }
    
    # Per-class accuracy
    for i, cls in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == all_labels[mask]).mean()
            results["per_class"][cls] = {
                "accuracy": float(cls_acc),
                "support": int(mask.sum())
            }
    
    output_path = artifact_dir / "internal_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()