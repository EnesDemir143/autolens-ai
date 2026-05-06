# Phase 3 Research: Baseline Training Pipeline
**Branch:** `feat/baseline-training-pipeline`

    ## Research Question

    What needs to be known to plan Phase 3 well?

    ## Training Research Notes

Use Lightning to keep training code explainable: Datamodule -> Model factory -> LightningModule -> callbacks/checkpoints. Use Albumentations for train augmentation and deterministic validation preprocessing. Device selection must prefer `mps` when available and fall back to CPU. Keep model factory small: MobileNetV4 Conv Medium, EfficientNet-B2, and ResNet baseline/comparison models only. Do not treat these CNNs as the intended final architecture by default; DINOv3 is the main path in Phase 4.


    ## Validation Architecture

    Phase 3 validation checks:
1. datamodule import succeeds.
2. model factory lists MobileNetV4/EfficientNet-B2/ResNet.
3. Lightning trainer config has early stopping/checkpointing.
4. MPS/CPU fallback code exists.
5. baseline runbook exists.
