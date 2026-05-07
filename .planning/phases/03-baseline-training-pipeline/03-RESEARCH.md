# Phase 3 Research: Baseline Training Pipeline
**Branch:** `feat/baseline-training-pipeline`

    ## Research Question

    What needs to be known to plan Phase 3 well?

    ## Training Research Notes

Use Lightning to keep training code explainable: Datamodule -> Model factory -> LightningModule -> callbacks/checkpoints. Device selection must prefer `mps` when available and fall back to CPU. Keep model factory small: MobileNetV4 Conv Medium, EfficientNet-B2, and ResNet18 baseline/comparison models only. Do not treat these CNNs as the intended final architecture by default; DINOv3 is the fourth model family and main path in Phase 4.

Preprocessing/augmentation research decision: start with a deterministic no-augmentation baseline before enabling train augmentation. Baseline 0 should use the same safe preprocessing for train, validation, and internal test: resize to 256, center crop to 224, convert to tensor, and train-split mean/std normalization. This preserves an interpretable reference point for label quality, class imbalance, and model-capacity debugging. Light augmentation belongs in Baseline 1 after Baseline 0 metrics exist; suggested light augmentation is horizontal flip plus mild color jitter and conservative random resized crop only if the no-augmentation reference is stable.

Class imbalance decision: because MICRO and STATION WAGON are weak classes, the first baseline should use class-weighted CrossEntropy rather than silently oversampling. Compute class weights from the training split only with inverse-frequency balancing, e.g. `weight_c = train_total / (num_classes * train_count_c)`, then apply a configurable max cap (default target: 5.0, tune only if unstable) so MICRO does not dominate the gradients. WeightedRandomSampler should be implemented/configurable as a follow-up comparison, not mixed into the first baseline by default.

Outlier decision: the first baseline must train on the Phase 2 outlier-unfiltered split. Because MICRO and STATION WAGON already have limited data, outlier/near-duplicate filtering is a low-priority optional comparison, not a required early training task. If time remains or leakage/quality evidence warrants it, generate a separate filtered artifact variant near the end and compare it against Baseline 0 using class counts, source coverage, macro/weighted F1, per-class F1, and normalized confusion matrix. Outlier filtering must not silently replace the baseline split.


    ## Validation Architecture

    Phase 3 validation checks:
1. datamodule import succeeds.
2. model factory lists MobileNetV4/EfficientNet-B2/ResNet18.
3. Lightning trainer config has early stopping/checkpointing.
4. MPS/CPU fallback code exists.
5. baseline runbook exists.
6. baseline runbook records Baseline 0 no-augmentation config, required follow-up variants for light augmentation and weighted sampler, and outlier-filtered comparison as low-priority optional work.
