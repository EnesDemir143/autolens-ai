# Graph Report - AutoLens AI

Generated: 2026-05-09T14:36:09.914671+00:00

## Summary

- Nodes: 99
- Edges: 192
- Hyperedges: 8
- Source corpus: `.planning/`, `docs/`, project skills
- Refresh command run: `node "$HOME/.codex/get-shit-done/bin/gsd-tools.cjs" graphify build .` and `make graphify-update`

## Communities

1. Project governance and constraints
2. Dataset curation
3. Baseline training
4. Export, calibration, and model comparison
5. Gradio interface with swappable model artifact
6. Hugging Face Spaces deploy phase
7. Final IEEE report

## 2026-05-09 Planning Pivot Captured

- Immediate lane: current best EfficientNet-B2 checkpoint -> ONNX export -> optional simplification -> 95 MB size check -> validation-only temperature scaling -> Gradio-ready inference metadata.
- Overnight lane: DINOv3 ViT-S non-LoRA, DINOv3 LoRA, and EfficientNet-B2 augmentation + EMA + focal-loss runs.
- Final lane: compare completed runs by macro F1 first, then balanced accuracy, MCC, per-class behavior, size, and latency; repeat export/calibration only for the selected winner.
- Makefile orchestration: selected run/checkpoint should flow through documented targets for export, size check, calibration, active Gradio artifact pointer update, and demo smoke check.
- UI loading state: Gradio must show prediction progress/status so slow CPU inference does not appear frozen.
- HF Spaces deploy phase: after local demo, package Gradio + ONNX Runtime for CPU Basic using Gradio SDK, no Docker/API scope by default.
- HF model publishing: completed model artifacts can be uploaded to the student HF account as model repos with model cards; raw merged datasets are not published by default due to mixed-source license/redistribution risk.
- Safetensors conversion: Lightning `.ckpt` remains the training-resume artifact; `model.safetensors + metadata` is the preferred HF/share/export source before ONNX.

## Notes

The installed Graphify command returned a spawn-agent response but did not rewrite the report file contents automatically in this runtime. This report was refreshed manually from the current GSD planning artifacts after running the project Graphify update commands, so downstream planning readers see the current execution strategy.
