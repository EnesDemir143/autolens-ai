#!/usr/bin/env python
"""Smoke test for the AutoLens AI Gradio demo.

Runs the ONNX predictor directly against a synthetic image and verifies
outputs match the expected artifact contract.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure project src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autolens_ai.inference import ONNXPredictor


def create_test_image(size: int = 224) -> Image.Image:
    """Create a synthetic RGB image for smoke testing."""
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def main() -> int:
    """Run smoke tests and return exit code."""
    print("=" * 50)
    print("AutoLens AI Demo Smoke Test")
    print("=" * 50)

    # 1. Artifact config exists
    config_path = Path("artifacts/demo/active_model.json")
    if not config_path.exists():
        print(f"FAIL: Artifact config not found: {config_path}")
        print("  Run: make prepare-demo-artifact")
        return 1
    print("PASS: Artifact config exists")

    # 2. Predictor loads
    try:
        predictor = ONNXPredictor.from_default_config()
    except Exception as exc:
        print(f"FAIL: Could not load predictor: {exc}")
        return 1
    print("PASS: Predictor loaded")

    # 3. Class labels count
    if len(predictor.class_labels) != 8:
        print(f"FAIL: Expected 8 classes, got {len(predictor.class_labels)}")
        return 1
    print(f"PASS: 8 class labels loaded: {predictor.class_labels}")

    # 4. Inference runs and returns expected structure
    image = create_test_image()
    try:
        result = predictor.predict(image)
    except Exception as exc:
        print(f"FAIL: Inference failed: {exc}")
        return 1

    required_keys = {
        "predicted_class",
        "predicted_display",
        "confidence",
        "probabilities",
        "latency_ms",
        "temperature",
    }
    missing = required_keys - set(result.keys())
    if missing:
        print(f"FAIL: Missing keys in result: {missing}")
        return 1
    print("PASS: Inference result structure valid")

    # 5. Probabilities sum to ~1
    probs = result["probabilities"]
    prob_sum = sum(probs.values())
    if not (0.99 <= prob_sum <= 1.01):
        print(f"FAIL: Probabilities sum to {prob_sum:.4f} (expected ~1.0)")
        return 1
    print(f"PASS: Probabilities sum to {prob_sum:.4f}")

    # 6. Latency is reasonable (< 5s for CPU)
    latency = result["latency_ms"]
    if latency > 5000:
        print(f"WARN: Latency {latency:.0f} ms is high (> 5s)")
    else:
        print(f"PASS: Latency {latency:.0f} ms")

    # 7. Predicted class is in label set
    if result["predicted_class"] not in predictor.class_labels:
        print(f"FAIL: Predicted class '{result['predicted_class']}' not in labels")
        return 1
    print(f"PASS: Predicted class: {result['predicted_class']}")

    # 8. Confidence is in [0, 1]
    conf = result["confidence"]
    if not (0.0 <= conf <= 1.0):
        print(f"FAIL: Confidence {conf} out of range [0, 1]")
        return 1
    print(f"PASS: Confidence: {conf:.1%}")

    # 9. Run a few iterations and record average latency
    print("\nBenchmarking 5 iterations...")
    latencies = []
    for _ in range(5):
        img = create_test_image()
        start = time.perf_counter()
        predictor.predict(img)
        latencies.append((time.perf_counter() - start) * 1000)
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average latency: {avg_latency:.1f} ms")

    # Save evidence
    evidence = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_config": str(config_path),
        "model_path": str(predictor.model_path),
        "num_classes": len(predictor.class_labels),
        "class_labels": predictor.class_labels,
        "single_inference_ms": latency,
        "avg_inference_ms_5_runs": round(avg_latency, 2),
        "status": "PASS",
    }
    evidence_path = Path("artifacts/demo/smoke_test_evidence.json")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"\nEvidence saved: {evidence_path}")

    print("\n" + "=" * 50)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
