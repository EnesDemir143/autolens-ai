# Next Steps After Final Model Selection

**Current state:** Phase 4 final model selection is complete. Phase 5 UI is complete and model-agnostic. The active demo pointer has now been switched from EfficientNet-B2 to the selected DINOv3 non-LoRA artifact and local smoke/latency evidence passes.

**Current phase:** Phase 6 — Hugging Face Spaces Demo Deploy and Model Publishing.

---

## Progress Checklist

| Phase | Status | What is done | What remains |
|---|---|---|---|
| Phase 1 — Project Foundation | Complete | Python/uv project, quality gates, baseline structure | None |
| Phase 2 — Dataset Curation | Complete | Dataset manifests, labels, splits | None |
| Phase 3 — Baseline Training | Complete | ResNet18, MobileNetV4, EfficientNet-B2 baseline training | None |
| Phase 4 — Model Comparison & Selection | **Complete** | DINOv3 non-LoRA selected; LoRA rejected; split-explicit comparison written; export/calibration evidence exists | None for selection; keep final artefacts immutable except fixes |
| Phase 5 — Demo Interface | Complete | Model-agnostic ONNX predictor/UI exists; active artifact now points to DINOv3; smoke test passes | None |
| Phase 6 — HF Deploy & Publishing | **In progress** | DINOv3/EfficientNet model repos uploaded; model cards/calibration evidence published; DINOv3 local demo smoke passes | Package/update Space with final DINOv3 artifact; record public URL |
| Phase 7 — Final Evidence & IEEE Report | Pending | Report plans exist | Freeze evidence, map report inputs, generate IEEE LaTeX report, final package checklist |

---

## Execution Plan From Here

### Step 1 — Repoint demo to final DINOv3 artifact — DONE

Update the active demo config to use:

- ONNX: `artifacts/export/dinov3_vits16_latest/model.onnx`
- Metadata: `artifacts/export/dinov3_vits16_latest/metadata.json`
- Calibration: `artifacts/export/dinov3_vits16_latest/calibration.json`

Validation completed 2026-05-10:

- `uv run python scripts/smoke_test_demo.py` → PASS
- Single inference: 44.21 ms
- Average latency over 5 runs: 39.64 ms

Command used:

```bash
uv run python scripts/prepare_demo_artifact.py \
  --metadata artifacts/export/dinov3_vits16_latest/metadata.json \
  --onnx artifacts/export/dinov3_vits16_latest/model.onnx \
  --calibration artifacts/export/dinov3_vits16_latest/calibration.json \
  --output artifacts/demo/active_model.json

uv run python scripts/smoke_test_demo.py
```

### Step 2 — Update local and HF demo package

Package the Gradio/HF Space using the final DINOv3 ONNX artifact. Do not upload raw datasets, credentials, or instructor final-test data.

Validation:

- Space/package includes app code, ONNX model, metadata, calibration, requirements, and README.
- Local smoke test passes before upload.
- Public Space URL is recorded.

### Step 3 — Record final evidence

Create/fill final evidence inventory with:

- Selected model rationale: `docs/dinov3_non_lora_selection.md`
- Full model comparison: `docs/model_comparison.md`
- Internal-test metrics: `artifacts/export/dinov3_vits16_latest/internal_test_results.json`
- Calibration: `artifacts/export/dinov3_vits16_latest/calibration.json`
- Size check: `artifacts/export/dinov3_vits16_latest/size_check.json`
- HF model repo URL: `https://huggingface.co/JiEunA1/autolens-dinov3-vits16`
- HF Space URL once deployed

### Step 4 — Generate IEEE report

Use the local `awesome-ieee-report` skill after evidence freeze. The report should cite the DINOv3 non-LoRA decision and explicitly state why LoRA was not selected.

### Step 5 — Final package checklist

Before submission:

- Verify app predicts from image upload.
- Verify probability visualization works.
- Verify final model artifact is under 95 MB.
- Verify no raw merged dataset images are published.
- Verify no secrets or `.env` files are included.
- Verify report is at least 4 pages and includes required metrics.
