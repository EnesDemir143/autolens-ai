# Phase 4 Validation Strategy

**Phase:** 4 — Export, Calibration, Overnight Model Selection
**Created:** 2026-05-06
**Revised:** 2026-05-09
**Branch:** `feat/model-comparison-and-selection`

## Required Evidence

### Immediate deployability lane

- Current best EfficientNet-B2 checkpoint path is recorded.
- Lightning `.ckpt` is converted to `model.safetensors` plus explicit metadata, or the exact conversion blocker is documented.
- Safetensors metadata is sufficient to rebuild the model for inference/export and excludes optimizer/callback/training-only state.
- ONNX export exists from the safetensors+metadata artifact, or the exact blocker/fallback to `.ckpt` export is documented.
- ONNX Runtime smoke inference passes against a non-final-test sample or synthetic tensor where appropriate.
- ONNX simplification is attempted when supported; if skipped/failing, rationale is documented.
- Artifact size is recorded and checked against the 95 MB submission limit.
- Temperature scaling is fit on validation predictions only.
- Calibration metadata is saved beside the exported model.
- Calibration evidence records before/after confidence behavior using NLL/ECE or equivalent diagnostics when available.

### Overnight comparison lane

- DINOv3 ViT-S non-LoRA run command/config is available and queued/executed before LoRA.
- DINOv3 LoRA is a separate optional comparison run after non-LoRA evidence exists.
- EfficientNet-B2 augmentation+EMA+focal-loss config/run command is available as the enhanced CNN candidate.
- Long-running experiment status is recorded without blocking the immediate Gradio demo path.

### Makefile orchestration lane

- `make help` lists post-training deployment targets and required variables.
- A chosen run/checkpoint can be passed to a documented Make target sequence for safetensors conversion, export, size check, calibration, active Gradio artifact update, and smoke validation.
- Make targets call small scripts/commands and do not hide complex Python logic inside Makefile recipes.
- Final model replacement is a config/artifact pointer change, not a Gradio code edit.

### Final selection lane

- Metric suite computes required metrics.
- Plots are generated to artifacts directory.
- Benchmark table contains all completed model-family results: MobileNetV4 Conv Medium, EfficientNet-B2 current/baseline, ResNet18, DINOv3 non-LoRA, optional DINOv3 LoRA, and enhanced EfficientNet-B2 if run.
- Metrics include macro F1, weighted F1, balanced accuracy, MCC, per-class metrics, normalized confusion matrix, size, and latency.
- Selected final artifact is under 95 MB or mitigation is documented.
- If an overnight model beats the current EfficientNet-B2 candidate, export/calibration evidence is repeated for that winner before Phase 6.
