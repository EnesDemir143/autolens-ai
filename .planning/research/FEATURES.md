# Feature Research

## Table Stakes

- Dataset ingestion from Kaggle and Hugging Face sources.
- Class normalization into exactly 8 assignment labels.
- Dataset audit showing source, class count, image count, and imbalance.
- Train/validation split with no final test leakage.
- Augmentation and preprocessing pipeline for 224x224 or model-specific input size.
- Main DINOv3 training/evaluation path plus baseline model training/evaluation runs.
- Macro and weighted metrics: F1, Accuracy, balanced accuracy, MCC, Precision, Recall.
- Per-class classification report.
- Training/validation loss graph, accuracy graph, normalized 8x8 confusion matrix.
- Saved model artifact under 95 MB.
- Gradio UI with upload, preview, predicted class, confidence, and class probability chart.

## Differentiators

- Grad-CAM visual explanations for CNN baselines.
- ONNX export and ONNX Runtime inference for faster UI prediction.
- Optuna tuning for learning rate, dropout, weight decay, augmentation strength.
- W&B experiment tracking for report-ready plots.
- DINOv3 non-LoRA main path and optional LoRA fine-tuning extension; CNN models remain baselines.

## Anti-Features

- Training on instructor final test images.
- Using one dataset blindly without class/source audit.
- Selecting a model only by validation accuracy while ignoring F1, size, speed, and explainability.
- Overbuilding a custom web app when Gradio satisfies the assignment faster.
