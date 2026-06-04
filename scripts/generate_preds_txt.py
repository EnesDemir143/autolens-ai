"""Batch prediction script — runs ONNX predictor on a folder of test images
and writes preds.txt in the format expected by PredictionScript.txt."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from autolens_ai.inference import ONNXPredictor

from autolens_ai.data.labels import TARGET_CLASSES

CLASS_TO_INDEX: dict[str, int] = {
    cls: idx for idx, cls in enumerate(TARGET_CLASSES, start=1)
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_preds_txt.py <test_images_dir> [output_path]")
        sys.exit(1)

    test_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("preds.txt")

    if not test_dir.exists() or not test_dir.is_dir():
        print(f"Error: directory not found: {test_dir}")
        sys.exit(1)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    image_files = sorted(
        f for f in test_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions
    )

    if not image_files:
        print(f"No image files found in {test_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images")
    print("Loading ONNX predictor...")
    predictor = ONNXPredictor.from_default_config()

    lines: list[str] = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            result = predictor.predict(img)
            predicted_class = result["predicted_class"]
            class_idx = CLASS_TO_INDEX[predicted_class]

            line = f"{img_path.name} | Value:{class_idx}"
            lines.append(line)

            print(f"  {img_path.name} -> {predicted_class} (Value:{class_idx})")
        except Exception as exc:
            print(f"  ERROR {img_path.name}: {exc}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {len(lines)} predictions to {output_path}")


if __name__ == "__main__":
    main()
