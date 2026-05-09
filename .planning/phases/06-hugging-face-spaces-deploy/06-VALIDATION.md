# Phase 6 Validation Strategy

**Phase:** 6 — Hugging Face Spaces Demo Deploy
**Created:** 2026-05-09
**Branch:** `feat/hugging-face-spaces-deploy`

## Required Evidence

- Local Gradio demo passes before deploy packaging starts.
- Deploy package excludes training datasets, final instructor test images, secrets, `.env`, W&B tokens, and large unnecessary checkpoints.
- Deploy package includes the active ONNX artifact/config/metadata/calibration needed for CPU inference.
- `requirements.txt` is minimal and includes Gradio, ONNX Runtime, Pillow, NumPy, YAML/config dependencies, and only required local package code.
- README or deploy runbook explains manual HF Spaces setup: create Space, choose Gradio SDK, CPU Basic hardware, upload/push files, inspect build logs, open link.
- Makefile exposes deploy packaging and/or CLI deploy commands where safe.
- If CLI deploy is attempted, missing Hugging Face auth is documented with exact `hf auth login`/manual fallback steps.
- Public Space link or local deploy package smoke evidence is recorded for Phase 7 report.
- No Dockerfile is added unless Gradio SDK deploy is documented as blocked.


## Model publishing evidence

- `docs/hf_model_publishing.md` exists and explains model repo publishing, model card contents, CLI/manual upload options, and dataset license caveats.
- Completed model artifact folders can be packaged for HF model repos without raw datasets, secrets, or instructor final test images.
- Model cards include dataset source links/count summaries instead of redistributing third-party image files.
- Optional CLI publish command documents required variables such as `REPO=owner/model` and `ARTIFACT_DIR=...`; missing HF auth falls back to manual upload steps.
