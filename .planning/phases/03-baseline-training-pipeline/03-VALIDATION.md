# Phase 3 Validation Strategy

**Phase:** 3 — Baseline Training Pipeline
**Created:** 2026-05-06
**Branch:** `feat/baseline-training-pipeline`

## Required Evidence

- datamodule import succeeds.
- model factory lists MobileNetV4/EfficientNet-B2/ResNet.
- Lightning trainer config has early stopping/checkpointing.
- MPS/CPU fallback code exists.
- baseline runbook exists.
- Baseline 0 config uses no train augmentation and deterministic resize 256 + center crop 224 + ImageNet normalization.
- Follow-up configs/runbook entries separate light augmentation, weighted sampler comparison, and outlier-filtered comparison from the first baseline.
